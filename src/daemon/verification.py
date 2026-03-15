"""Result verification and reproducibility checking.

Re-runs sandbox executions and compares metrics to detect
non-reproducible results. Stores verification runs with match scores
and discrepancy details.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.verification")


class ResultVerifier:
    """Verify reproducibility of sandbox execution results."""

    def __init__(self, config: Any, db: Any, sandbox_runner: Any) -> None:
        self._config = config
        self._db = db
        self._runner = sandbox_runner

    # -- Public API ----------------------------------------------------------

    def verify_result(self, execution_id: str) -> dict[str, Any]:
        """Re-run an execution and compare metrics for reproducibility.

        Steps:
        1. Get original execution from db.
        2. Read the original script from work_dir/script.py.
        3. Re-run it via sandbox_runner.execute().
        4. Compare metrics within tolerance.
        5. Store verification_run in db.

        Returns:
            {"verification_id": str, "reproducible": bool,
             "match_score": float, "discrepancies": list}
        """
        original = self._db.get_sandbox_execution(execution_id)
        if original is None:
            raise ValueError(f"Execution {execution_id} not found")

        verification_id = str(ULID())

        # Create initial verification record
        original_metrics = original.get("metrics") or {}
        self._db.insert_verification_run({
            "id": verification_id,
            "original_execution_id": execution_id,
            "verification_type": "reproducibility",
            "status": "running",
            "original_metrics": original_metrics,
            "reproduced_metrics": {},
            "discrepancies": [],
        })

        # Read the original script
        work_dir = original.get("work_dir", "")
        script_path = Path(work_dir) / "script.py"
        if not script_path.exists():
            self._db.update_verification_run(
                verification_id,
                status="failed",
                notes=f"Script not found at {script_path}",
            )
            log.error("Verification %s: script not found at %s", verification_id, script_path)
            return {
                "verification_id": verification_id,
                "reproducible": False,
                "match_score": 0.0,
                "discrepancies": [{"error": f"Script not found at {script_path}"}],
            }

        script = script_path.read_text(encoding="utf-8")

        # Read requirements if present
        req_path = Path(work_dir) / "requirements.txt"
        requirements: list[str] = []
        if req_path.exists():
            raw = req_path.read_text(encoding="utf-8").strip()
            if raw:
                requirements = [line.strip() for line in raw.splitlines() if line.strip()]

        # Re-run the script
        log.info(
            "Verification %s: re-running execution %s",
            verification_id, execution_id,
        )
        rerun_result = self._runner.execute(
            script=script,
            requirements=requirements,
            timeout_s=None,  # use default
            description=f"Verification re-run of {execution_id}",
        )

        if rerun_result.returncode != 0:
            self._db.update_verification_run(
                verification_id,
                status="failed",
                notes=f"Re-run failed (rc={rerun_result.returncode}): {rerun_result.stderr[:500]}",
            )
            log.warning(
                "Verification %s: re-run failed (rc=%d)",
                verification_id, rerun_result.returncode,
            )
            return {
                "verification_id": verification_id,
                "reproducible": False,
                "match_score": 0.0,
                "discrepancies": [{
                    "error": "re-run failed",
                    "returncode": rerun_result.returncode,
                    "stderr": rerun_result.stderr[:500],
                }],
            }

        # Compare metrics
        reproduced_metrics = rerun_result.metrics or {}
        tolerance = self._config.reproducibility_tolerance
        match_score, discrepancies = self._compare_metrics(
            original_metrics, reproduced_metrics, tolerance
        )

        reproducible = match_score >= (1.0 - tolerance)
        status = "passed" if reproducible else "failed"

        self._db.update_verification_run(
            verification_id,
            status=status,
            reproduced_metrics=reproduced_metrics,
            match_score=match_score,
            discrepancies=discrepancies,
            notes=f"Match score: {match_score:.3f}, tolerance: {tolerance}",
        )

        log.info(
            "Verification %s: match_score=%.3f reproducible=%s discrepancies=%d",
            verification_id, match_score, reproducible, len(discrepancies),
        )
        return {
            "verification_id": verification_id,
            "reproducible": reproducible,
            "match_score": match_score,
            "discrepancies": discrepancies,
        }

    def check_reproducibility(self, verification_id: str) -> dict[str, Any]:
        """Get verification run details and analysis."""
        run = self._db.get_verification_run(verification_id)
        if run is None:
            raise ValueError(f"Verification run {verification_id} not found")

        return {
            "verification_id": run["id"],
            "original_execution_id": run["original_execution_id"],
            "status": run["status"],
            "match_score": run.get("match_score"),
            "reproducible": run["status"] == "passed",
            "original_metrics": run.get("original_metrics", {}),
            "reproduced_metrics": run.get("reproduced_metrics", {}),
            "discrepancies": run.get("discrepancies", []),
            "notes": run.get("notes", ""),
            "created_at": run.get("created_at"),
        }

    # -- Private helpers -----------------------------------------------------

    def _compare_metrics(
        self,
        original: dict[str, Any],
        reproduced: dict[str, Any],
        tolerance: float,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Compare two metric dicts.

        Returns:
            (match_score, discrepancies)
            - match_score: fraction of metrics within tolerance (0.0-1.0)
            - discrepancies: list of metrics that differ beyond tolerance
        """
        if not original and not reproduced:
            return 1.0, []

        all_keys = set(original.keys()) | set(reproduced.keys())
        if not all_keys:
            return 1.0, []

        matched = 0
        discrepancies: list[dict[str, Any]] = []

        for key in all_keys:
            orig_val = original.get(key)
            repr_val = reproduced.get(key)

            if key not in original:
                discrepancies.append({
                    "metric": key,
                    "issue": "missing_in_original",
                    "reproduced_value": repr_val,
                })
                continue

            if key not in reproduced:
                discrepancies.append({
                    "metric": key,
                    "issue": "missing_in_reproduced",
                    "original_value": orig_val,
                })
                continue

            # Compare numeric values with tolerance
            if isinstance(orig_val, (int, float)) and isinstance(repr_val, (int, float)):
                if orig_val == 0 and repr_val == 0:
                    matched += 1
                elif orig_val == 0:
                    # Absolute comparison when original is zero
                    if abs(repr_val) <= tolerance:
                        matched += 1
                    else:
                        discrepancies.append({
                            "metric": key,
                            "issue": "value_mismatch",
                            "original_value": orig_val,
                            "reproduced_value": repr_val,
                            "relative_diff": None,
                            "absolute_diff": abs(repr_val),
                        })
                else:
                    relative_diff = abs(repr_val - orig_val) / abs(orig_val)
                    if relative_diff <= tolerance:
                        matched += 1
                    else:
                        discrepancies.append({
                            "metric": key,
                            "issue": "value_mismatch",
                            "original_value": orig_val,
                            "reproduced_value": repr_val,
                            "relative_diff": round(relative_diff, 6),
                            "absolute_diff": round(abs(repr_val - orig_val), 6),
                        })
            else:
                # Non-numeric: exact comparison
                if orig_val == repr_val:
                    matched += 1
                else:
                    discrepancies.append({
                        "metric": key,
                        "issue": "value_mismatch",
                        "original_value": orig_val,
                        "reproduced_value": repr_val,
                    })

        match_score = matched / len(all_keys) if all_keys else 1.0
        return round(match_score, 4), discrepancies
