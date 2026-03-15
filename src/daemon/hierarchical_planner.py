"""Hierarchical planning with conditional branching and backtracking.

Extends the base PlanManager with:
- Conditional branches (metric thresholds, hypothesis verdicts, custom LLM-evaluated)
- Backtracking to re-run plan segments
- Tree-structured plan views
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.hierarchical_planner")


class HierarchicalPlanner:
    """Manage plan branches, conditional execution, and backtracking."""

    def __init__(self, db: Any, config: Any) -> None:
        self._db = db
        self._config = config

    def add_branch(
        self,
        plan_id: str,
        from_step_id: str,
        condition: str,
        condition_type: str = "metric_threshold",
        then_steps: list[dict[str, Any]] | None = None,
        else_steps: list[dict[str, Any]] | None = None,
    ) -> str:
        """Add a conditional branch to a plan step.

        condition_type:
            "metric_threshold" - e.g. "rmse < 0.5" or "accuracy > 0.8"
            "hypothesis_verdict" - e.g. "hypothesis_id:confirmed"
            "custom" - free text evaluated by LLM

        then_steps / else_steps: list of step dicts with keys
            {"title": ..., "step_type": ..., "depends_on": [...]}

        Returns the branch_id.
        """
        branch_id = str(ULID())
        self._db.insert_plan_branch({
            "id": branch_id,
            "plan_id": plan_id,
            "from_step_id": from_step_id,
            "condition": condition,
            "condition_type": condition_type,
            "then_steps": then_steps or [],
            "else_steps": else_steps or [],
        })
        log.info(
            "Added branch %s to plan %s (type=%s, condition=%s)",
            branch_id, plan_id, condition_type, condition,
        )
        return branch_id

    def evaluate_branch(self, branch_id: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate a branch condition and create the appropriate steps.

        For metric_threshold: parse condition, compare against latest execution metrics.
        For hypothesis_verdict: check knowledge base for hypothesis status.
        For custom: use inference client to evaluate (requires context with inference client).

        Returns {"result": bool, "steps_created": [step_ids], "reason": str}.
        """
        # Fetch the branch
        # get_plan_branches returns all branches for a plan, so we need to find ours
        # We will search across all branches - not ideal but the DB API gives us plan-level access
        branch = self._find_branch(branch_id)
        if not branch:
            return {"result": False, "steps_created": [], "reason": f"Branch {branch_id} not found"}

        condition = branch["condition"]
        condition_type = branch.get("condition_type", "metric_threshold")

        # Evaluate the condition
        if condition_type == "metric_threshold":
            result, reason = self._evaluate_metric_condition(condition)
        elif condition_type == "hypothesis_verdict":
            result, reason = self._evaluate_hypothesis_condition(condition)
        elif condition_type == "custom":
            result, reason = self._evaluate_custom_condition(condition, context)
        else:
            result = False
            reason = f"Unknown condition_type: {condition_type}"

        # Select which steps to create
        step_defs = branch["then_steps"] if result else branch["else_steps"]
        steps_created = self._create_steps_from_defs(
            branch["plan_id"], branch["from_step_id"], step_defs,
        )

        # Mark branch as evaluated
        self._db.update_plan_branch(branch_id, evaluated=1, result=1 if result else 0)

        log.info(
            "Branch %s evaluated: result=%s, created %d steps. Reason: %s",
            branch_id, result, len(steps_created), reason,
        )
        return {"result": result, "steps_created": steps_created, "reason": reason}

    def backtrack(self, plan_id: str, to_step_id: str, reason: str = "") -> dict[str, Any]:
        """Reset all steps after to_step_id back to pending.

        Finds the target step's order, then resets all subsequent steps.
        Returns {"steps_reset": N, "reason": str}.
        """
        steps = self._db.get_plan_steps(plan_id)
        if not steps:
            return {"steps_reset": 0, "reason": "No steps found"}

        # Find the target step and its order
        target_step = None
        for s in steps:
            if s["id"] == to_step_id:
                target_step = s
                break

        if not target_step:
            return {"steps_reset": 0, "reason": f"Step {to_step_id} not found"}

        target_order = target_step.get("step_order", 0)
        reset_count = 0
        now = datetime.now(timezone.utc).isoformat()
        backtrack_note = f"[BACKTRACK {now}] {reason}" if reason else f"[BACKTRACK {now}]"

        for s in steps:
            step_order = s.get("step_order", 0)
            if step_order > target_order and s["status"] not in ("pending",):
                self._db.update_plan_step(
                    s["id"],
                    status="pending",
                    notes=backtrack_note,
                )
                reset_count += 1

        log.info(
            "Backtracked plan %s to step %s: reset %d steps. Reason: %s",
            plan_id, to_step_id, reset_count, reason,
        )
        return {"steps_reset": reset_count, "reason": reason or "backtrack requested"}

    def get_plan_tree(self, plan_id: str) -> dict[str, Any]:
        """Get full plan structure including steps and branches as a tree.

        Returns plan dict with "steps" list and "branches" list,
        where each branch references from_step_id and contains
        then_steps / else_steps definitions plus evaluation status.
        """
        plan = self._db.get_research_plan(plan_id)
        if not plan:
            return {"error": f"Plan {plan_id} not found"}

        steps = self._db.get_plan_steps(plan_id)
        branches = self._db.get_plan_branches(plan_id)

        # Index branches by from_step_id for tree construction
        branches_by_step: dict[str, list[dict[str, Any]]] = {}
        for b in branches:
            sid = b["from_step_id"]
            branches_by_step.setdefault(sid, []).append(b)

        # Build tree nodes from steps
        tree_nodes = []
        for s in steps:
            node: dict[str, Any] = {
                "id": s["id"],
                "title": s["title"],
                "step_type": s.get("step_type", "custom"),
                "status": s["status"],
                "order": s.get("step_order", 0),
                "depends_on": s.get("depends_on", []),
                "notes": s.get("notes", ""),
            }
            if s["id"] in branches_by_step:
                node["branches"] = branches_by_step[s["id"]]
            tree_nodes.append(node)

        return {
            "plan_id": plan_id,
            "title": plan.get("title", ""),
            "status": plan.get("status", ""),
            "steps": tree_nodes,
            "branches": branches,
        }

    # -- Private helpers --------------------------------------------------

    def _evaluate_metric_condition(self, condition: str) -> tuple[bool, str]:
        """Parse 'metric_name operator value' and compare against latest execution metrics.

        Supported formats: 'rmse < 0.5', 'accuracy > 0.8', 'r2 >= 0.7'
        """
        match = re.match(r"(\w+)\s*([<>=!]+)\s*([\d.]+)", condition)
        if not match:
            return False, f"Could not parse metric condition: {condition}"

        metric_name = match.group(1)
        operator = match.group(2)
        threshold = float(match.group(3))

        # Get latest sandbox execution metrics
        recent = self._db.get_recent_sandbox_executions(limit=5)
        actual_value = None
        for execution in recent:
            metrics = execution.get("metrics", {})
            if metric_name in metrics:
                actual_value = metrics[metric_name]
                break

        if actual_value is None:
            return False, f"Metric '{metric_name}' not found in recent executions"

        try:
            actual_value = float(actual_value)
        except (ValueError, TypeError):
            return False, f"Metric '{metric_name}' value '{actual_value}' is not numeric"

        result = _compare(actual_value, operator, threshold)
        reason = f"{metric_name} = {actual_value} {operator} {threshold} -> {result}"
        return result, reason

    def _evaluate_hypothesis_condition(self, condition: str) -> tuple[bool, str]:
        """Parse 'hypothesis_id:expected_verdict' and check knowledge base.

        expected_verdict: "confirmed", "refuted", "inconclusive"
        """
        parts = condition.split(":", 1)
        if len(parts) != 2:
            return False, f"Invalid hypothesis condition format: {condition} (expected 'id:verdict')"

        hypothesis_id = parts[0].strip()
        expected_verdict = parts[1].strip().lower()

        # Look up hypothesis in knowledge base
        hypotheses = self._db.get_knowledge_by_type("hypothesis", limit=100)
        target = None
        for h in hypotheses:
            if h["id"] == hypothesis_id:
                target = h
                break

        if not target:
            return False, f"Hypothesis {hypothesis_id} not found"

        # Check the content or status for the verdict
        content = (target.get("content", "") or "").lower()
        actual_verdict = "inconclusive"
        if "confirmed" in content or "supported" in content:
            actual_verdict = "confirmed"
        elif "refuted" in content or "rejected" in content:
            actual_verdict = "refuted"

        result = actual_verdict == expected_verdict
        reason = f"Hypothesis {hypothesis_id}: actual={actual_verdict}, expected={expected_verdict}"
        return result, reason

    def _evaluate_custom_condition(
        self, condition: str, context: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Evaluate a free-text condition using the inference client if available.

        Falls back to False if no inference client is provided in context.
        """
        inference = (context or {}).get("inference")
        if not inference:
            return False, "No inference client available for custom condition evaluation"

        prompt = (
            f"Evaluate whether the following condition is TRUE or FALSE "
            f"based on the provided context.\n\n"
            f"Condition: {condition}\n\n"
        )
        if context:
            extra = {k: v for k, v in context.items() if k != "inference"}
            if extra:
                prompt += f"Context: {extra}\n\n"
        prompt += "Respond with ONLY 'TRUE' or 'FALSE' followed by a brief explanation."

        try:
            resp = inference.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.content.strip().upper()
            result = text.startswith("TRUE")
            reason = resp.content.strip()
            return result, reason
        except Exception as exc:
            return False, f"Inference error evaluating custom condition: {exc}"

    def _find_branch(self, branch_id: str) -> dict[str, Any] | None:
        """Find a branch by ID across all plans.

        The DB API only provides get_plan_branches(plan_id), so we
        check recent plan steps to locate the branch's plan.
        """
        # Try to find via plan_steps that might reference this branch
        # First, check a reasonable set of recent plans
        for status in ("active", "completed"):
            try:
                plans = self._db.get_research_plans_by_status(status)
            except Exception:
                continue
            for plan in plans:
                branches = self._db.get_plan_branches(plan["id"])
                for b in branches:
                    if b["id"] == branch_id:
                        return b
        return None

    def _create_steps_from_defs(
        self,
        plan_id: str,
        from_step_id: str,
        step_defs: list[dict[str, Any]],
    ) -> list[str]:
        """Create plan steps from a list of step definitions.

        New steps are ordered after the from_step and depend on it
        unless explicit depends_on is provided.
        """
        if not step_defs:
            return []

        # Get current max order for the plan
        existing_steps = self._db.get_plan_steps(plan_id)
        max_order = max((s.get("step_order", 0) for s in existing_steps), default=0)

        created_ids = []
        for i, step_def in enumerate(step_defs):
            step_id = str(ULID())
            depends = step_def.get("depends_on", [from_step_id])
            self._db.insert_plan_step({
                "id": step_id,
                "plan_id": plan_id,
                "title": step_def.get("title", f"Branch step {i}"),
                "description": step_def.get("description", ""),
                "step_type": step_def.get("step_type", "custom"),
                "status": "pending",
                "step_order": max_order + i + 1,
                "depends_on": depends,
                "notes": f"Created from branch evaluation (from_step={from_step_id})",
            })
            created_ids.append(step_id)

        return created_ids


def _compare(actual: float, operator: str, threshold: float) -> bool:
    """Compare a value against a threshold using the given operator."""
    if operator == "<":
        return actual < threshold
    elif operator == "<=":
        return actual <= threshold
    elif operator == ">":
        return actual > threshold
    elif operator == ">=":
        return actual >= threshold
    elif operator in ("==", "="):
        return actual == threshold
    elif operator == "!=":
        return actual != threshold
    else:
        log.warning("Unknown operator '%s', defaulting to False", operator)
        return False
