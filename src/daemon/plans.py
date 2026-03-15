"""Research plan management with dependency tracking.

Plans supersede research_cycles for new structured work. Each plan has
ordered steps with dependencies, typed links to knowledge/executions,
and progress tracking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.plans")


VALID_STEP_TYPES = {
    "literature_review",
    "data_acquisition",
    "experiment",
    "analysis",
    "writing",
    "custom",
}

VALID_STEP_STATUSES = {
    "pending",
    "in_progress",
    "completed",
    "blocked",
    "skipped",
}

TERMINAL_STATUSES = {"completed", "skipped"}


@dataclass
class PlanStep:
    id: str
    plan_id: str
    title: str
    description: str
    step_type: str
    status: str
    order: int
    depends_on: list[str]
    notes: str
    started_at: str | None
    completed_at: str | None


class PlanManager:
    """Manage multi-step research plans with dependency tracking."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def create_plan(
        self,
        title: str,
        description: str = "",
        steps: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create a plan with optional initial steps. Returns plan_id."""
        plan_id = str(ULID())
        self._db.insert_research_plan({
            "id": plan_id,
            "title": title,
            "description": description,
            "status": "active",
        })

        if steps:
            for i, step_def in enumerate(steps):
                step_id = str(ULID())
                self._db.insert_plan_step({
                    "id": step_id,
                    "plan_id": plan_id,
                    "title": step_def.get("title", f"Step {i}"),
                    "description": step_def.get("description", ""),
                    "step_type": step_def.get("step_type", "custom"),
                    "status": "pending",
                    "step_order": i,
                    "depends_on": step_def.get("depends_on", []),
                    "notes": "",
                })

        log.info("Created plan '%s' (%s) with %d steps", title, plan_id, len(steps or []))
        return plan_id

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        """Get plan with all steps."""
        plan = self._db.get_research_plan(plan_id)
        if not plan:
            return None
        steps = self._db.get_plan_steps(plan_id)
        plan["steps"] = steps
        return plan

    def update_step(
        self,
        plan_id: str,
        step_id: str,
        status: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Update step status and/or notes. Validates dependency constraints."""
        step = self._db.get_plan_step(step_id)
        if not step:
            return {"ok": False, "error": f"step not found: {step_id}"}
        if step["plan_id"] != plan_id:
            return {"ok": False, "error": "step does not belong to this plan"}

        updates: dict[str, Any] = {}
        now = datetime.now(timezone.utc).isoformat()

        if status:
            if status not in VALID_STEP_STATUSES:
                return {"ok": False, "error": f"invalid status: {status}"}

            # Check dependencies before marking completed
            if status == "completed":
                deps = step.get("depends_on", [])
                if deps:
                    all_steps = self._db.get_plan_steps(plan_id)
                    steps_by_id = {s["id"]: s for s in all_steps}
                    for dep_idx in deps:
                        # depends_on can be step IDs or indices
                        dep_step = _resolve_dep(dep_idx, all_steps, steps_by_id)
                        if dep_step and dep_step["status"] not in TERMINAL_STATUSES:
                            return {
                                "ok": False,
                                "error": (
                                    f"Cannot complete: dependency '{dep_step['title']}' "
                                    f"is {dep_step['status']}"
                                ),
                            }

            updates["status"] = status
            if status == "in_progress" and not step.get("started_at"):
                updates["started_at"] = now
            if status in TERMINAL_STATUSES:
                updates["completed_at"] = now

        if notes is not None:
            existing_notes = step.get("notes", "")
            if existing_notes:
                updates["notes"] = existing_notes + "\n" + notes
            else:
                updates["notes"] = notes

        if updates:
            self._db.update_plan_step(step_id, **updates)

        # Check if plan is now complete
        self._check_plan_completion(plan_id)

        return {"ok": True, "step_id": step_id, "updates": updates}

    def link_to_step(
        self, step_id: str, link_type: str, target_id: str
    ) -> None:
        """Link a knowledge entry, execution, or dataset to a plan step."""
        self._db.insert_plan_step_link(step_id, link_type, target_id)

    def get_progress(self, plan_id: str) -> dict[str, Any]:
        """Get plan progress summary."""
        plan = self._db.get_research_plan(plan_id)
        if not plan:
            return {"error": f"plan not found: {plan_id}"}

        steps = self._db.get_plan_steps(plan_id)
        total = len(steps)
        completed = sum(1 for s in steps if s["status"] in TERMINAL_STATUSES)
        in_progress = sum(1 for s in steps if s["status"] == "in_progress")
        blocked = sum(1 for s in steps if s["status"] == "blocked")
        pct = (completed / total * 100) if total > 0 else 0

        next_action = self.suggest_next_action(plan_id)
        blockers = [
            {"step": s["title"], "status": s["status"]}
            for s in steps
            if s["status"] == "blocked"
        ]

        return {
            "plan_id": plan_id,
            "title": plan["title"],
            "status": plan["status"],
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "pct": round(pct, 1),
            "next_action": next_action,
            "blockers": blockers,
        }

    def suggest_next_action(self, plan_id: str) -> dict[str, Any] | None:
        """Return the lowest-order pending step whose dependencies are met."""
        steps = self._db.get_plan_steps(plan_id)
        if not steps:
            return None

        steps_by_id = {s["id"]: s for s in steps}

        for step in steps:
            if step["status"] != "pending":
                continue

            # Check all dependencies are met
            deps = step.get("depends_on", [])
            all_met = True
            for dep_ref in deps:
                dep_step = _resolve_dep(dep_ref, steps, steps_by_id)
                if dep_step and dep_step["status"] not in TERMINAL_STATUSES:
                    all_met = False
                    break

            if all_met:
                return {
                    "step_id": step["id"],
                    "title": step["title"],
                    "step_type": step["step_type"],
                    "description": step["description"],
                }

        return None

    def get_active_plans(self) -> list[dict[str, Any]]:
        """Get all active plans with progress summaries."""
        plans = self._db.get_research_plans_by_status("active")
        results = []
        for plan in plans:
            progress = self.get_progress(plan["id"])
            results.append(progress)
        return results

    def _check_plan_completion(self, plan_id: str) -> None:
        """Mark plan as completed if all steps are done."""
        steps = self._db.get_plan_steps(plan_id)
        if not steps:
            return
        if all(s["status"] in TERMINAL_STATUSES for s in steps):
            self._db.update_research_plan(plan_id, status="completed")
            log.info("Plan %s completed (all steps done)", plan_id)

    def format_active_plans_summary(self, max_plans: int = 5) -> str:
        """Format active plans for inclusion in system prompt."""
        active = self.get_active_plans()
        if not active:
            return ""

        lines = ["--- ACTIVE PLANS ---"]
        for p in active[:max_plans]:
            next_str = ""
            if p.get("next_action"):
                next_str = f', next: "{p["next_action"]["title"]}"'
            lines.append(
                f'Plan: "{p["title"]}" ({p["completed"]}/{p["total"]} steps done{next_str})'
            )
        lines.append("--- END PLANS ---")
        return "\n".join(lines)


def _resolve_dep(
    dep_ref: Any,
    all_steps: list[dict[str, Any]],
    steps_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve a dependency reference - can be step ID or integer index."""
    # Try as step ID first
    if isinstance(dep_ref, str) and dep_ref in steps_by_id:
        return steps_by_id[dep_ref]
    # Try as integer index
    try:
        idx = int(dep_ref)
        if 0 <= idx < len(all_steps):
            return all_steps[idx]
    except (ValueError, TypeError):
        pass
    return None
