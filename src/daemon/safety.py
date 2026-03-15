"""Safety and alignment monitor - p-hacking detection, HARKing prevention, audit trails."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.safety")


class SafetyMonitor:
    def __init__(self, config: Any, db: Any) -> None:
        self.config = config
        self.db = db

    def run_audit(self) -> dict[str, Any]:
        """Full audit: check for p-hacking, HARKing, cherry-picking."""
        flags: list[dict[str, Any]] = []

        phacking = self.check_phacking()
        flags.extend(phacking)

        harking = self.check_harking()
        flags.extend(harking)

        cherry = self.check_cherry_picking()
        flags.extend(cherry)

        dredging = self._check_data_dredging()
        flags.extend(dredging)

        # Store new flags
        for flag in flags:
            self.db.insert_safety_flag(flag)

        return {
            "flags_created": len(flags),
            "flags": [
                {"id": f["id"], "flag_type": f["flag_type"],
                 "severity": f.get("severity", "warning"),
                 "description": f["description"]}
                for f in flags
            ],
            "clean": len(flags) == 0,
        }

    def check_phacking(self) -> list[dict[str, Any]]:
        """Detect multiple testing on same dataset without correction."""
        flags: list[dict[str, Any]] = []
        threshold = self.config.phacking_test_threshold

        # Get all statistical tests
        try:
            rows = self.db._conn.execute(
                "SELECT linked_execution_id, COUNT(*) as test_count, "
                "GROUP_CONCAT(id) as test_ids "
                "FROM statistical_tests "
                "WHERE linked_execution_id IS NOT NULL "
                "GROUP BY linked_execution_id "
                "HAVING test_count > ?",
                (threshold,),
            ).fetchall()
        except Exception:
            return flags

        for row in rows:
            test_ids = (row["test_ids"] or "").split(",")
            flags.append({
                "id": str(ULID()),
                "flag_type": "p_hacking",
                "severity": "warning",
                "description": (
                    f"Multiple testing alert: {row['test_count']} statistical tests "
                    f"on execution {row['linked_execution_id']}. "
                    f"Consider Bonferroni or FDR correction."
                ),
                "evidence": {
                    "execution_id": row["linked_execution_id"],
                    "test_count": row["test_count"],
                    "test_ids": test_ids[:10],
                },
                "related_ids": test_ids[:10],
            })

        return flags

    def check_harking(self) -> list[dict[str, Any]]:
        """Detect hypothesizing after results are known."""
        flags: list[dict[str, Any]] = []
        threshold_s = self.config.harking_time_threshold_s

        try:
            # Find hypotheses that were created AFTER linked experiments
            rows = self.db._conn.execute(
                "SELECT h.id as hyp_id, h.content as hyp_content, "
                "h.created_at as hyp_created, "
                "e.id as exp_id, e.created_at as exp_created "
                "FROM knowledge h "
                "JOIN knowledge e ON e.parent_id = h.id "
                "WHERE h.entry_type = 'hypothesis' "
                "AND e.entry_type = 'experiment' "
                "AND julianday(h.created_at) > julianday(e.created_at)"
            ).fetchall()
        except Exception:
            return flags

        for row in rows:
            flags.append({
                "id": str(ULID()),
                "flag_type": "harking",
                "severity": "warning",
                "description": (
                    f"HARKing alert: hypothesis '{row['hyp_content'][:80]}...' "
                    f"was created after its linked experiment. "
                    f"Hypothesis: {row['hyp_created']}, Experiment: {row['exp_created']}"
                ),
                "evidence": {
                    "hypothesis_id": row["hyp_id"],
                    "experiment_id": row["exp_id"],
                    "hypothesis_created": row["hyp_created"],
                    "experiment_created": row["exp_created"],
                },
                "related_ids": [row["hyp_id"], row["exp_id"]],
            })

        return flags

    def check_cherry_picking(self) -> list[dict[str, Any]]:
        """Detect selective reporting of positive results."""
        flags: list[dict[str, Any]] = []

        try:
            # Get all hypotheses
            hypotheses = self.db._conn.execute(
                "SELECT id, content, status FROM knowledge "
                "WHERE entry_type = 'hypothesis'"
            ).fetchall()

            concluded_positive = 0
            concluded_negative = 0
            open_with_findings = 0

            for h in hypotheses:
                # Check for conclusions
                conclusions = self.db._conn.execute(
                    "SELECT content, status FROM knowledge "
                    "WHERE entry_type = 'conclusion' AND parent_id = ?",
                    (h["id"],),
                ).fetchall()

                if conclusions:
                    for c in conclusions:
                        if c["status"] == "confirmed":
                            concluded_positive += 1
                        elif c["status"] == "refuted":
                            concluded_negative += 1
                else:
                    # Check if there are findings but no conclusion
                    experiments = self.db._conn.execute(
                        "SELECT id FROM knowledge "
                        "WHERE entry_type = 'experiment' AND parent_id = ?",
                        (h["id"],),
                    ).fetchall()
                    for exp in experiments:
                        findings = self.db._conn.execute(
                            "SELECT id FROM knowledge "
                            "WHERE entry_type = 'finding' AND parent_id = ?",
                            (exp["id"],),
                        ).fetchall()
                        if findings:
                            open_with_findings += 1
                            break

            total_concluded = concluded_positive + concluded_negative
            if total_concluded >= 5 and concluded_negative == 0 and open_with_findings > 0:
                flags.append({
                    "id": str(ULID()),
                    "flag_type": "cherry_picking",
                    "severity": "warning",
                    "description": (
                        f"Potential cherry-picking: {concluded_positive} confirmed hypotheses, "
                        f"0 refuted, and {open_with_findings} open hypotheses with findings. "
                        f"Consider concluding open hypotheses even if results are negative."
                    ),
                    "evidence": {
                        "confirmed": concluded_positive,
                        "refuted": concluded_negative,
                        "open_with_findings": open_with_findings,
                    },
                    "related_ids": [],
                })
        except Exception as exc:
            log.warning("cherry_picking check failed: %s", exc)

        return flags

    def _check_data_dredging(self) -> list[dict[str, Any]]:
        """Flag high ratio of tests to findings."""
        flags: list[dict[str, Any]] = []

        try:
            test_count = self.db._conn.execute(
                "SELECT COUNT(*) as c FROM statistical_tests"
            ).fetchone()["c"]

            finding_count = self.db._conn.execute(
                "SELECT COUNT(*) as c FROM knowledge WHERE entry_type = 'finding'"
            ).fetchone()["c"]

            if test_count > 10 and finding_count > 0:
                ratio = test_count / finding_count
                if ratio > 5:
                    flags.append({
                        "id": str(ULID()),
                        "flag_type": "data_dredging",
                        "severity": "info",
                        "description": (
                            f"High test-to-finding ratio: {test_count} tests for "
                            f"{finding_count} findings (ratio={ratio:.1f}). "
                            f"Consider whether all tests were pre-registered."
                        ),
                        "evidence": {
                            "test_count": test_count,
                            "finding_count": finding_count,
                            "ratio": round(ratio, 2),
                        },
                        "related_ids": [],
                    })
        except Exception as exc:
            log.warning("data_dredging check failed: %s", exc)

        return flags

    def get_audit_trail(self, entity_id: str) -> dict[str, Any]:
        """Full provenance trace: hypothesis -> experiments -> findings -> conclusion."""
        entity = self.db.get_knowledge_by_id(entity_id)
        if not entity:
            return {"error": f"Entity not found: {entity_id}"}

        chain: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []

        def _trace(eid: str, depth: int = 0) -> None:
            if depth > 10:
                return
            e = self.db.get_knowledge_by_id(eid)
            if not e:
                return
            chain.append({
                "id": e["id"],
                "entry_type": e.get("entry_type", ""),
                "content": e.get("content", "")[:200],
                "status": e.get("status", ""),
                "created_at": e.get("created_at", ""),
                "depth": depth,
            })
            timeline.append({
                "id": e["id"],
                "entry_type": e.get("entry_type", ""),
                "created_at": e.get("created_at", ""),
            })

            # Find children
            try:
                children = self.db._conn.execute(
                    "SELECT id FROM knowledge WHERE parent_id = ? ORDER BY created_at",
                    (eid,),
                ).fetchall()
                for child in children:
                    _trace(child["id"], depth + 1)
            except Exception:
                pass

        # Walk up to root
        root_id = entity_id
        try:
            current = entity
            while current and current.get("parent_id"):
                parent = self.db.get_knowledge_by_id(current["parent_id"])
                if not parent:
                    break
                root_id = parent["id"]
                current = parent
        except Exception:
            pass

        _trace(root_id)

        # Sort timeline
        timeline.sort(key=lambda x: x.get("created_at", ""))

        # Calculate integrity score
        has_hypothesis = any(c["entry_type"] == "hypothesis" for c in chain)
        has_experiment = any(c["entry_type"] == "experiment" for c in chain)
        has_finding = any(c["entry_type"] == "finding" for c in chain)
        has_conclusion = any(c["entry_type"] == "conclusion" for c in chain)
        steps = sum([has_hypothesis, has_experiment, has_finding, has_conclusion])
        integrity_score = steps / 4.0

        # Check for safety flags on related entities
        related_flags = []
        entity_ids = [c["id"] for c in chain]
        existing_flags = self.db.list_safety_flags(include_dismissed=False)
        for flag in existing_flags:
            if any(eid in flag.get("related_ids", []) for eid in entity_ids):
                related_flags.append({
                    "id": flag["id"],
                    "flag_type": flag["flag_type"],
                    "description": flag["description"][:200],
                })

        return {
            "entity_id": entity_id,
            "root_id": root_id,
            "chain": chain,
            "timeline": timeline,
            "integrity_score": round(integrity_score, 2),
            "safety_flags": related_flags,
        }

    def list_flags(self, include_dismissed: bool = False) -> list[dict[str, Any]]:
        return self.db.list_safety_flags(include_dismissed=include_dismissed)

    def dismiss_flag(self, flag_id: str, reason: str) -> None:
        self.db.dismiss_safety_flag(flag_id, reason)
