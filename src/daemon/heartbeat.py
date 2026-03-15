"""Heartbeat daemon - cron-based periodic task runner.

Port of automaton/src/heartbeat/daemon.ts + tasks.ts.
Runs periodic tasks even while the agent sleeps.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from croniter import croniter

from .config import DaemonConfig

log = logging.getLogger("daemon.heartbeat")

# ---------------------------------------------------------------------------
# Task function signature
# ---------------------------------------------------------------------------

HeartbeatTaskFn = Callable[
    ["HeartbeatTaskContext"],
    dict[str, Any],  # {"should_wake": bool, "message": str}
]


class HeartbeatTaskContext:
    def __init__(self, config: DaemonConfig, db: Any) -> None:
        self.config = config
        self.db = db


# ---------------------------------------------------------------------------
# Built-in research-oriented tasks
# ---------------------------------------------------------------------------


def _task_literature_fetch(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Search OpenAlex for new papers on the research topic and store them."""
    try:
        from agent.web_tools import openalex_search_works

        out = openalex_search_works(
            ctx.config.topic,
            max_results=5,
            from_publication_date="2024-01-01",
            sort="publication_date:desc",
        )
        results = out.get("results", [])
        if not results:
            return {"should_wake": False, "message": "No new papers found."}

        # Check for genuinely new papers and store in papers table
        new_count = 0
        for paper in results:
            title = paper.get("title", "")
            doi = paper.get("doi") or ""

            # Dedup: check by DOI first, then by title marker
            if doi and ctx.db.get_paper_by_doi(doi):
                continue
            key = f"paper_seen_{title[:60]}"
            if ctx.db.get_kv(key):
                continue

            # Extract author names
            authorships = paper.get("authorships", [])
            author_names = [
                a.get("author", {}).get("display_name", "")
                for a in authorships
                if a.get("author", {}).get("display_name")
            ]
            authors_str = "; ".join(author_names[:10])

            # Extract year from publication_date
            pub_date = paper.get("publication_date", "")
            year = int(pub_date[:4]) if pub_date and len(pub_date) >= 4 else None

            # Store in papers table
            paper_id = paper.get("id", "") or f"openalex:{title[:60]}"
            ctx.db.insert_paper({
                "id": paper_id,
                "title": title,
                "authors": authors_str,
                "year": year,
                "doi": doi or None,
                "abstract": _reconstruct_abstract(paper),
                "source": "openalex",
                "url": paper.get("primary_location", {}).get("landing_page_url") if paper.get("primary_location") else None,
                "cited_by_count": paper.get("cited_by_count", 0),
            })

            # Mark as seen in KV for backward compat
            ctx.db.set_kv(key, datetime.now(timezone.utc).isoformat())
            new_count += 1

        if new_count > 0:
            return {
                "should_wake": True,
                "message": f"Found {new_count} new paper(s) on '{ctx.config.topic}'.",
            }
        return {"should_wake": False, "message": "No new papers since last check."}
    except Exception as exc:
        log.warning("literature_fetch failed: %s", exc)
        return {"should_wake": False, "message": f"Error: {exc}"}


def _reconstruct_abstract(paper: dict[str, Any]) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    abstract_inv = paper.get("abstract_inverted_index")
    if not abstract_inv:
        return ""
    try:
        # OpenAlex stores abstracts as {word: [positions]}
        word_positions: list[tuple[int, str]] = []
        for word, positions in abstract_inv.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(w for _, w in word_positions)
    except Exception:
        return ""


def _task_experiment_watchdog(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Check if any running experiments have completed."""
    import subprocess

    # Check for common experiment marker files
    run_dir = ctx.config.runs_root
    if not run_dir.exists():
        return {"should_wake": False, "message": "No runs directory."}

    # Look for recently modified result files
    try:
        result = subprocess.run(
            ["find", str(run_dir), "-name", "results.json", "-mmin", "-10"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        files = [f for f in result.stdout.strip().split("\n") if f]
        if files:
            return {
                "should_wake": True,
                "message": f"Experiment results ready: {len(files)} file(s).",
            }
    except Exception:
        pass
    return {"should_wake": False, "message": "No experiment results ready."}


def _task_knowledge_consolidate(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Log knowledge base stats. Full consolidation happens in the agent loop."""
    count = len(ctx.db.get_recent_knowledge(limit=10000))
    ctx.db.set_kv(
        "knowledge_stats",
        f'{{"count": {count}, "checked_at": "{datetime.now(timezone.utc).isoformat()}"}}',
    )
    return {"should_wake": False, "message": f"Knowledge entries: {count}"}


def _task_cost_check(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Check API spend vs budget, update tier."""
    from .cost_tracker import get_total_spent, get_budget_tier

    spent = get_total_spent(ctx.db)
    tier = get_budget_tier(ctx.db, ctx.config)
    prev_tier = ctx.db.get_kv("prev_budget_tier")
    ctx.db.set_kv("prev_budget_tier", tier)
    ctx.db.set_kv(
        "last_cost_check",
        f'{{"spent_cents": {spent}, "tier": "{tier}", '
        f'"timestamp": "{datetime.now(timezone.utc).isoformat()}"}}',
    )

    if prev_tier and prev_tier != tier and tier in ("critical", "dead"):
        return {
            "should_wake": True,
            "message": f"Budget tier dropped to {tier}: ${spent / 100:.2f} spent.",
        }
    return {"should_wake": False, "message": f"Cost: ${spent / 100:.2f}, tier: {tier}"}


def _task_status_log(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Log current state to stdout."""
    state = ctx.db.get_agent_state()
    turns = ctx.db.get_turn_count()
    log.info("[STATUS] state=%s turns=%d topic=%s", state, turns, ctx.config.topic)
    return {"should_wake": False, "message": f"state={state} turns={turns}"}


def _task_check_children(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Check child agent status and read their knowledge."""
    children = ctx.db.get_children()
    alive = [c for c in children if c.get("status") not in ("dead", "stopped")]
    if not alive:
        return {"should_wake": False, "message": "No active children."}

    from .children import check_child_status, read_child_knowledge

    new_knowledge = 0
    for child in alive:
        status = check_child_status(ctx.db, child["id"])
        # Read child knowledge
        try:
            knowledge = read_child_knowledge(Path(child["config_path"]))
            for k in knowledge:
                key = f"child_knowledge_seen_{k.get('id', '')}"
                if not ctx.db.get_kv(key):
                    ctx.db.set_kv(key, "1")
                    new_knowledge += 1
        except Exception:
            pass

    if new_knowledge > 0:
        return {
            "should_wake": True,
            "message": f"Children produced {new_knowledge} new finding(s).",
        }
    return {"should_wake": False, "message": f"{len(alive)} children alive, no new findings."}


def _task_check_social_inbox(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Poll inbox for new messages from peers/children."""
    messages = ctx.db.get_unprocessed_inbox(limit=10)
    if not messages:
        return {"should_wake": False, "message": "Inbox empty."}
    senders = list({m["from_agent"] for m in messages})
    return {
        "should_wake": True,
        "message": f"{len(messages)} unread message(s) from: {', '.join(senders[:3])}",
    }


def _task_review_hypotheses(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Check for hypotheses that have findings but no conclusion, and stale hypotheses."""
    # Hypotheses with findings but no conclusion - should wake
    needs_conclusion = ctx.db.get_hypotheses_with_findings_no_conclusion()
    if needs_conclusion:
        names = [h["content"][:80] for h in needs_conclusion[:3]]
        return {
            "should_wake": True,
            "message": (
                f"{len(needs_conclusion)} hypothesis/hypotheses have findings but no conclusion: "
                + "; ".join(names)
            ),
        }

    # Stale hypotheses (>7 days old, still active) - informational only
    stale = ctx.db.get_stale_hypotheses(older_than_hours=168)
    if stale:
        names = [h["content"][:80] for h in stale[:3]]
        log.info(
            "Stale hypotheses (%d): %s",
            len(stale),
            "; ".join(names),
        )

    return {"should_wake": False, "message": "No hypotheses need immediate conclusion."}


def _task_embed_backfill(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Batch-compute embeddings for knowledge entries and papers that lack them."""
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"should_wake": False, "message": "OPENAI_API_KEY not set, skipping embed backfill."}

    try:
        from . import embeddings

        # Knowledge entries
        unembedded_k = ctx.db.get_unembedded_knowledge(limit=20)
        if unembedded_k:
            texts = [
                f"{e.get('topic', '')} {e.get('content', '')}"
                for e in unembedded_k
            ]
            vecs = embeddings.embed_texts(texts, api_key)
            for entry, vec in zip(unembedded_k, vecs):
                ctx.db.update_knowledge_embedding(
                    entry["id"], embeddings.serialize_embedding(vec)
                )
            log.info("Embedded %d knowledge entries.", len(unembedded_k))

        # Papers
        unembedded_p = ctx.db.get_unembedded_papers(limit=20)
        if unembedded_p:
            texts = [
                f"{p.get('title', '')} {p.get('abstract', '')}"
                for p in unembedded_p
            ]
            vecs = embeddings.embed_texts(texts, api_key)
            for paper, vec in zip(unembedded_p, vecs):
                ctx.db.update_paper_embedding(
                    paper["id"], embeddings.serialize_embedding(vec)
                )
            log.info("Embedded %d papers.", len(unembedded_p))

        total = len(unembedded_k) + len(unembedded_p)
        return {
            "should_wake": False,
            "message": f"Embedded {total} items ({len(unembedded_k)} knowledge, {len(unembedded_p)} papers).",
        }
    except Exception as exc:
        log.warning("embed_backfill failed: %s", exc)
        return {"should_wake": False, "message": f"Embed backfill error: {exc}"}


def _task_sandbox_cleanup(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Clean up old sandbox workspaces. Keep last 50, delete older than 7 days."""
    try:
        from .sandbox import SandboxRunner
        runner = SandboxRunner(ctx.config, ctx.db)
        deleted = runner.cleanup_old_workspaces(max_age_days=7, keep_last=50)
        if deleted > 0:
            log.info("Sandbox cleanup: removed %d old workspace(s)", deleted)
        return {"should_wake": False, "message": f"Sandbox cleanup: removed {deleted} workspace(s)"}
    except Exception as exc:
        log.warning("sandbox_cleanup failed: %s", exc)
        return {"should_wake": False, "message": f"Sandbox cleanup error: {exc}"}


def _task_plan_progress_check(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Flag stalled plans - in_progress steps untouched for >24h."""
    try:
        from .plans import PlanManager
        plan_mgr = PlanManager(ctx.db)
        active_plans = plan_mgr.get_active_plans()
        stalled: list[str] = []

        for plan in active_plans:
            plan_data = plan_mgr.get_plan(plan["plan_id"])
            if not plan_data:
                continue
            for step in plan_data.get("steps", []):
                if step["status"] != "in_progress":
                    continue
                started = step.get("started_at")
                if not started:
                    continue
                try:
                    started_dt = datetime.fromisoformat(started)
                    age_hours = (
                        datetime.now(timezone.utc) - started_dt
                    ).total_seconds() / 3600
                    if age_hours > 24:
                        stalled.append(
                            f"'{step['title']}' in plan '{plan['title']}' "
                            f"(in_progress for {age_hours:.0f}h)"
                        )
                except Exception:
                    pass

        if stalled:
            msg = f"{len(stalled)} stalled plan step(s): " + "; ".join(stalled[:3])
            log.warning("Plan progress: %s", msg)
            return {"should_wake": True, "message": msg}

        return {
            "should_wake": False,
            "message": f"{len(active_plans)} active plan(s), none stalled",
        }
    except Exception as exc:
        log.warning("plan_progress_check failed: %s", exc)
        return {"should_wake": False, "message": f"Plan check error: {exc}"}


def _task_experiment_loop_monitor(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Check for stalled experiment loops - running loops with no trial in >4h."""
    try:
        active_loops = ctx.db.get_active_experiment_loops()
        if not active_loops:
            return {"should_wake": False, "message": "No active experiment loops."}

        stalled: list[str] = []
        for loop in active_loops:
            loop_id = loop["id"]
            trials = ctx.db.get_loop_trials(loop_id)
            if not trials:
                stalled.append(f"'{loop.get('title', loop_id)}' (no trials yet)")
                continue
            # Trials are ordered by creation; check the last one
            last_trial = trials[-1]
            last_created = last_trial.get("created_at", "")
            if not last_created:
                continue
            try:
                last_dt = datetime.fromisoformat(last_created)
                age_hours = (
                    datetime.now(timezone.utc) - last_dt
                ).total_seconds() / 3600
                if age_hours > 4:
                    stalled.append(
                        f"'{loop.get('title', loop_id)}' (last trial {age_hours:.0f}h ago)"
                    )
            except Exception:
                pass

        if stalled:
            msg = f"{len(stalled)} stalled experiment loop(s): " + "; ".join(stalled[:3])
            log.warning("Experiment loop monitor: %s", msg)
            return {"should_wake": True, "message": msg}

        return {
            "should_wake": False,
            "message": f"{len(active_loops)} active loop(s), none stalled",
        }
    except Exception as exc:
        log.warning("experiment_loop_monitor failed: %s", exc)
        return {"should_wake": False, "message": f"Loop monitor error: {exc}"}


def _task_verification_scheduler(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Schedule re-verification of important results that haven't been verified recently."""
    try:
        # Get recent successful sandbox executions
        recent = ctx.db.get_recent_sandbox_executions(limit=20)
        successful = [e for e in recent if e.get("returncode") == 0]
        if not successful:
            return {"should_wake": False, "message": "No successful executions to verify."}

        # Filter out those already verified
        unverified: list[dict[str, Any]] = []
        for ex in successful:
            verifications = ctx.db.get_verifications_for_execution(ex["id"])
            if not verifications:
                unverified.append(ex)

        if not unverified:
            return {"should_wake": False, "message": "All recent results have been verified."}

        # Don't auto-verify unless config allows it
        if not ctx.config.auto_verify_results:
            return {
                "should_wake": False,
                "message": f"{len(unverified)} result(s) await verification (auto_verify disabled).",
            }

        # Flag for verification
        descs = [r.get("description", "")[:60] for r in unverified[:3]]
        return {
            "should_wake": True,
            "message": (
                f"{len(unverified)} successful result(s) need verification: "
                + "; ".join(descs)
            ),
        }
    except Exception as exc:
        log.warning("verification_scheduler failed: %s", exc)
        return {"should_wake": False, "message": f"Verification scheduler error: {exc}"}


def _task_meta_learning_extract(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Extract meta-learning patterns from accumulated strategy outcomes."""
    try:
        from .meta_learning import MetaLearner
        meta = MetaLearner(ctx.db, ctx.config)

        # Check if there are enough outcomes for pattern extraction
        strategy_log = ctx.db.get_strategy_log(limit=100)
        total_outcomes = len(strategy_log)

        if total_outcomes < 5:
            return {
                "should_wake": False,
                "message": f"Only {total_outcomes} strategy outcome(s), need 5+ for pattern extraction.",
            }

        patterns = meta.extract_patterns()
        ctx.db.set_kv(
            "meta_learning_last_extract",
            datetime.now(timezone.utc).isoformat(),
        )

        new_patterns = len(patterns) if patterns else 0
        if new_patterns > 0:
            log.info("Meta-learning: extracted %d pattern(s)", new_patterns)
            return {
                "should_wake": False,
                "message": f"Extracted {new_patterns} meta-learning pattern(s).",
            }
        return {"should_wake": False, "message": "No new patterns extracted."}
    except Exception as exc:
        log.warning("meta_learning_extract failed: %s", exc)
        return {"should_wake": False, "message": f"Meta-learning error: {exc}"}


def _task_benchmark_check(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Auto-validate recent sandbox execution metrics against registered benchmarks."""
    try:
        from .benchmark import BenchmarkValidator
        validator = BenchmarkValidator(ctx.config, ctx.db)

        # Seed defaults on first run
        seeded = validator.seed_defaults()
        if seeded > 0:
            log.info("Seeded %d default benchmarks.", seeded)

        # Get recent successful executions
        recent = ctx.db.get_recent_sandbox_executions(limit=10)
        successful = [e for e in recent if e.get("returncode") == 0]
        if not successful:
            return {"should_wake": False, "message": "No recent executions to benchmark."}

        total_checks = 0
        total_failed = 0
        for ex in successful:
            result = validator.validate_execution(ex["id"])
            total_checks += len(result.get("checks", []))
            total_failed += result.get("failed_count", 0)

        if total_failed > 0:
            return {
                "should_wake": True,
                "message": (
                    f"Benchmark validation: {total_failed}/{total_checks} checks failed "
                    f"across {len(successful)} execution(s)."
                ),
            }
        return {
            "should_wake": False,
            "message": f"Benchmark check: {total_checks} check(s) passed across {len(successful)} execution(s).",
        }
    except Exception as exc:
        log.warning("benchmark_check failed: %s", exc)
        return {"should_wake": False, "message": f"Benchmark check error: {exc}"}


def _task_safety_audit(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Run periodic safety audit for p-hacking, HARKing, cherry-picking."""
    try:
        from .safety import SafetyMonitor
        monitor = SafetyMonitor(ctx.config, ctx.db)
        result = monitor.run_audit()

        flags_created = result.get("flags_created", 0)
        if flags_created > 0:
            # Check if any are critical
            critical = [
                f for f in result.get("flags", [])
                if f.get("severity") == "critical"
            ]
            if critical:
                return {
                    "should_wake": True,
                    "message": (
                        f"CRITICAL safety flag(s): {len(critical)} critical, "
                        f"{flags_created} total new flag(s)."
                    ),
                }
            return {
                "should_wake": False,
                "message": f"Safety audit: {flags_created} new flag(s) created.",
            }
        return {"should_wake": False, "message": "Safety audit: clean."}
    except Exception as exc:
        log.warning("safety_audit failed: %s", exc)
        return {"should_wake": False, "message": f"Safety audit error: {exc}"}


def _task_coordination_check(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Check specialist children: read completed tasks, flag stuck ones."""
    try:
        pending = ctx.db.get_pending_coordination_tasks()
        if not pending:
            return {"should_wake": False, "message": "No pending coordination tasks."}

        stuck: list[str] = []
        for task in pending:
            assigned = task.get("assigned_at", "")
            if not assigned:
                continue
            try:
                assigned_dt = datetime.fromisoformat(assigned)
                age_hours = (
                    datetime.now(timezone.utc) - assigned_dt
                ).total_seconds() / 3600
                if age_hours > 4:
                    stuck.append(
                        f"task '{task['description'][:60]}' "
                        f"for child {task['child_id'][:12]} ({age_hours:.0f}h)"
                    )
            except Exception:
                pass

        if stuck:
            msg = f"{len(stuck)} stuck coordination task(s): " + "; ".join(stuck[:3])
            return {"should_wake": True, "message": msg}

        return {
            "should_wake": False,
            "message": f"{len(pending)} pending coordination task(s), none stuck.",
        }
    except Exception as exc:
        log.warning("coordination_check failed: %s", exc)
        return {"should_wake": False, "message": f"Coordination check error: {exc}"}


def _task_paper_fulltext_fetch(ctx: HeartbeatTaskContext) -> dict[str, Any]:
    """Fetch PDFs and extract full text for papers that have a pdf_url but no full_text."""
    try:
        papers = ctx.db.get_papers_without_full_text(limit=3)
        if not papers:
            return {"should_wake": False, "message": "No papers need full-text fetch."}

        from .literature import LiteratureManager
        lit = LiteratureManager(ctx.config, ctx.db)

        fetched = 0
        for paper in papers:
            result = lit.fetch_paper_pdf(paper["id"])
            if result.get("ok"):
                fetched += 1

        if fetched > 0:
            log.info("Paper fulltext fetch: got %d/%d papers", fetched, len(papers))
        return {
            "should_wake": False,
            "message": f"Fetched full text for {fetched}/{len(papers)} paper(s).",
        }
    except Exception as exc:
        log.warning("paper_fulltext_fetch failed: %s", exc)
        return {"should_wake": False, "message": f"Error: {exc}"}


# Registry of built-in tasks
BUILTIN_TASKS: dict[str, HeartbeatTaskFn] = {
    "literature_fetch": _task_literature_fetch,
    "experiment_watchdog": _task_experiment_watchdog,
    "knowledge_consolidate": _task_knowledge_consolidate,
    "cost_check": _task_cost_check,
    "status_log": _task_status_log,
    "check_children": _task_check_children,
    "check_social_inbox": _task_check_social_inbox,
    "review_hypotheses": _task_review_hypotheses,
    "embed_backfill": _task_embed_backfill,
    "sandbox_cleanup": _task_sandbox_cleanup,
    "plan_progress_check": _task_plan_progress_check,
    "experiment_loop_monitor": _task_experiment_loop_monitor,
    "verification_scheduler": _task_verification_scheduler,
    "meta_learning_extract": _task_meta_learning_extract,
    "benchmark_check": _task_benchmark_check,
    "safety_audit": _task_safety_audit,
    "coordination_check": _task_coordination_check,
    "paper_fulltext_fetch": _task_paper_fulltext_fetch,
}


# ---------------------------------------------------------------------------
# Default heartbeat schedule
# ---------------------------------------------------------------------------

DEFAULT_HEARTBEAT_ENTRIES: list[dict[str, Any]] = [
    {"name": "literature_fetch", "schedule": "0 */4 * * *", "task": "literature_fetch", "enabled": True},
    {"name": "experiment_watchdog", "schedule": "*/10 * * * *", "task": "experiment_watchdog", "enabled": True},
    {"name": "knowledge_consolidate", "schedule": "0 */12 * * *", "task": "knowledge_consolidate", "enabled": True},
    {"name": "cost_check", "schedule": "*/30 * * * *", "task": "cost_check", "enabled": True},
    {"name": "status_log", "schedule": "*/15 * * * *", "task": "status_log", "enabled": True},
    {"name": "check_children", "schedule": "*/5 * * * *", "task": "check_children", "enabled": True},
    {"name": "check_social_inbox", "schedule": "*/2 * * * *", "task": "check_social_inbox", "enabled": True},
    {"name": "review_hypotheses", "schedule": "0 */8 * * *", "task": "review_hypotheses", "enabled": True},
    {"name": "embed_backfill", "schedule": "0 */6 * * *", "task": "embed_backfill", "enabled": True},
    {"name": "sandbox_cleanup", "schedule": "0 3 * * *", "task": "sandbox_cleanup", "enabled": True},
    {"name": "plan_progress_check", "schedule": "0 */8 * * *", "task": "plan_progress_check", "enabled": True},
    {"name": "experiment_loop_monitor", "schedule": "0 */4 * * *", "task": "experiment_loop_monitor", "enabled": True},
    {"name": "verification_scheduler", "schedule": "0 6 * * *", "task": "verification_scheduler", "enabled": True},
    {"name": "meta_learning_extract", "schedule": "0 */12 * * *", "task": "meta_learning_extract", "enabled": True},
    {"name": "benchmark_check", "schedule": "0 8 * * *", "task": "benchmark_check", "enabled": True},
    {"name": "safety_audit", "schedule": "0 */6 * * *", "task": "safety_audit", "enabled": True},
    {"name": "coordination_check", "schedule": "*/15 * * * *", "task": "coordination_check", "enabled": True},
    {"name": "paper_fulltext_fetch", "schedule": "0 */6 * * *", "task": "paper_fulltext_fetch", "enabled": True},
]


# ---------------------------------------------------------------------------
# HeartbeatDaemon
# ---------------------------------------------------------------------------


class HeartbeatDaemon:
    """Periodic task runner using threading.Timer."""

    def __init__(
        self,
        config: DaemonConfig,
        db: Any,
        on_wake_request: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.on_wake_request = on_wake_request
        self._running = False
        self._timer: threading.Timer | None = None
        self._ctx = HeartbeatTaskContext(config, db)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        log.info("Heartbeat daemon started. Interval: %ds", self.config.heartbeat_interval_s)
        # Run first tick immediately
        self._tick()
        self._schedule_next()

    def stop(self) -> None:
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
        log.info("Heartbeat daemon stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    def force_run(self, task_name: str) -> None:
        """Manually trigger a specific task."""
        entries = self.db.get_heartbeat_entries()
        for entry in entries:
            if entry["name"] == task_name:
                self._execute_task(entry)
                return

    def _schedule_next(self) -> None:
        if not self._running:
            return
        interval = self.config.heartbeat_interval_s
        self._timer = threading.Timer(interval, self._tick_and_reschedule)
        self._timer.daemon = True
        self._timer.start()

    def _tick_and_reschedule(self) -> None:
        self._tick()
        self._schedule_next()

    def _tick(self) -> None:
        """Main tick: check all entries, run due ones."""
        entries = self.db.get_heartbeat_entries()

        # In low compute mode, only run essential tasks
        from .cost_tracker import get_budget_tier

        tier = get_budget_tier(self.db, self.config)
        essential_tasks = {"cost_check", "status_log", "check_social_inbox"}
        is_low = tier in ("low_compute", "critical", "dead")

        for entry in entries:
            if not entry.get("enabled", True):
                continue
            if is_low and entry["task"] not in essential_tasks:
                continue
            if self._is_due(entry):
                self._execute_task(entry)

    def _is_due(self, entry: dict[str, Any]) -> bool:
        schedule = entry.get("schedule", "")
        if not schedule:
            return False
        last_run = entry.get("last_run")
        try:
            base_time = (
                datetime.fromisoformat(last_run)
                if last_run
                else datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            )
            cron = croniter(schedule, base_time)
            next_run = cron.get_next(datetime)
            return next_run <= datetime.now(timezone.utc)
        except Exception:
            return False

    def _execute_task(self, entry: dict[str, Any]) -> None:
        task_fn = BUILTIN_TASKS.get(entry["task"])
        if not task_fn:
            return

        try:
            result = task_fn(self._ctx)
            now = datetime.now(timezone.utc).isoformat()
            self.db.update_heartbeat_last_run(entry["name"], now)

            if result.get("should_wake") and self.on_wake_request:
                msg = result.get("message", f"Task '{entry['name']}' requested wake")
                self.on_wake_request(msg)
        except Exception as exc:
            log.error("Heartbeat task '%s' failed: %s", entry["name"], exc)


# ---------------------------------------------------------------------------
# Heartbeat config sync
# ---------------------------------------------------------------------------


def sync_heartbeat_to_db(db: Any, config_path: Path | None = None) -> None:
    """Sync heartbeat entries from YAML config (or defaults) into the DB."""
    import yaml

    entries = DEFAULT_HEARTBEAT_ENTRIES
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if "entries" in raw:
            entries = raw["entries"]

    existing = {e["name"] for e in db.get_heartbeat_entries()}
    for entry in entries:
        if entry["name"] not in existing:
            db.upsert_heartbeat_entry(entry)
