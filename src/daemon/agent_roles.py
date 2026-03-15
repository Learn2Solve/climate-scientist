"""Multi-agent specialization - role-based child agent coordination."""

from __future__ import annotations

import json
import logging
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.agent_roles")


ROLE_PROFILES: dict[str, dict[str, Any]] = {
    "experimenter": {
        "genesis": (
            "You specialize in designing and running experiments. "
            "Focus on: sandbox script generation, parameter sweeps, "
            "experiment loop optimization, and metrics collection. "
            "Always validate results with statistical tests before reporting."
        ),
        "heartbeat_focus": ["experiment_watchdog", "sandbox_cleanup"],
        "tools_emphasis": [
            "sandbox_run", "start_experiment_loop", "run_next_trial",
            "run_statistical_test", "validate_experiment_rigor",
        ],
    },
    "theorist": {
        "genesis": (
            "You specialize in hypothesis generation and causal reasoning. "
            "Focus on: proposing testable hypotheses, building causal DAGs, "
            "checking novelty of ideas, and connecting findings to theory. "
            "Always check novelty before investing in new hypotheses."
        ),
        "heartbeat_focus": ["literature_fetch", "review_hypotheses"],
        "tools_emphasis": [
            "propose_hypothesis", "build_causal_dag", "check_novelty",
            "suggest_confounders", "conclude_hypothesis",
        ],
    },
    "reviewer": {
        "genesis": (
            "You specialize in critical review and verification. "
            "Focus on: verifying results, running safety audits, "
            "adversarial debate on contested claims, and ensuring "
            "reproducibility. Be skeptical and thorough."
        ),
        "heartbeat_focus": ["review_hypotheses", "safety_audit"],
        "tools_emphasis": [
            "review_paper", "verify_result", "start_debate",
            "run_safety_audit", "validate_experiment_rigor",
        ],
    },
    "data_curator": {
        "genesis": (
            "You specialize in data acquisition, cleaning, and analysis. "
            "Focus on: finding and downloading datasets, running climate "
            "analyses, ensuring data quality, and maintaining the data pipeline. "
            "Document all data provenance carefully."
        ),
        "heartbeat_focus": ["literature_fetch"],
        "tools_emphasis": [
            "acquire_dataset", "fetch_era5", "run_climate_analysis",
            "search_datasets", "sandbox_run",
        ],
    },
}


class AgentRoleManager:
    def __init__(self, config: Any, db: Any, inference: Any | None = None) -> None:
        self.config = config
        self.db = db
        self.inference = inference

    def spawn_specialist(self, role: str, topic: str, name: str | None = None,
                         budget_cents: int | None = None) -> dict[str, Any]:
        """Spawn a child agent with a specialized role."""
        if role not in ROLE_PROFILES:
            return {"error": f"Unknown role: {role}. Available: {list(ROLE_PROFILES.keys())}"}

        # Check child limit
        existing_roles = self.db.list_agent_roles()
        active_children = self.db.get_children()
        alive = [c for c in active_children if c.get("status") not in ("dead", "stopped")]
        if len(alive) >= self.config.max_specialist_children:
            return {"error": f"Max specialist children ({self.config.max_specialist_children}) reached"}

        profile = ROLE_PROFILES[role]
        child_name = name or f"{role}_{str(ULID())[:8]}"
        child_budget = budget_cents or self.config.specialist_budget_cents

        # Build enhanced genesis prompt
        genesis = (
            f"ROLE: {role.upper()}\n\n"
            f"{profile['genesis']}\n\n"
            f"TOPIC: {topic}\n\n"
            f"PRIORITY TOOLS: {', '.join(profile['tools_emphasis'])}\n\n"
            f"Report findings back to the parent agent regularly."
        )

        from .children import spawn_child

        try:
            child = spawn_child(
                parent_config=self.config,
                parent_db=self.db,
                name=child_name,
                topic=topic,
                budget_cents=child_budget,
                genesis_prompt=genesis,
            )
        except Exception as exc:
            return {"error": f"Failed to spawn child: {exc}"}

        child_id = child["id"]

        # Record role
        self.db.insert_agent_role({
            "child_id": child_id,
            "role": role,
            "specialization": topic,
            "genesis_prompt": genesis,
        })

        return {
            "child_id": child_id,
            "name": child_name,
            "role": role,
            "topic": topic,
            "budget_cents": child_budget,
        }

    def assign_task(self, child_id: str, task_type: str,
                    description: str) -> str:
        """Assign a specific task to a specialist child."""
        role = self.db.get_agent_role(child_id)
        if not role:
            return ""  # Child not found as specialist, but allow anyway

        task_id = str(ULID())
        self.db.insert_coordination_task({
            "id": task_id,
            "child_id": child_id,
            "task_type": task_type,
            "description": description,
        })

        # Send task to child
        from .children import send_to_child

        message = (
            f"TASK ASSIGNMENT [{task_type}]\n\n"
            f"{description}\n\n"
            f"When complete, save your results to the knowledge base "
            f"and report back to the parent agent."
        )
        try:
            send_to_child(self.db, child_id, message, sender_name="coordinator")
        except Exception as exc:
            log.warning("Failed to send task to child %s: %s", child_id, exc)

        return task_id

    def check_task_status(self, task_id: str) -> dict[str, Any]:
        """Check if assigned task is complete and read results."""
        task = self.db.get_coordination_task(task_id)
        if not task:
            return {"error": f"Task not found: {task_id}"}

        child_id = task["child_id"]

        # Check child status
        from .children import check_child_status, read_child_knowledge

        child_record = None
        for c in self.db.get_children():
            if c["id"] == child_id:
                child_record = c
                break

        child_status = "unknown"
        recent_knowledge: list[dict[str, Any]] = []
        if child_record:
            status_info = check_child_status(self.db, child_id)
            child_status = status_info.get("status", "unknown")
            try:
                from pathlib import Path
                knowledge = read_child_knowledge(Path(child_record["config_path"]))
                recent_knowledge = [
                    {"topic": k.get("topic", ""), "content": k.get("content", "")[:200]}
                    for k in knowledge[:5]
                ]
            except Exception:
                pass

        # Check inbox for task completion messages
        inbox = self.db.get_unprocessed_inbox(limit=50)
        task_messages = [
            m for m in inbox
            if child_id in m.get("from_agent", "")
        ]

        return {
            "task_id": task_id,
            "status": task["status"],
            "child_status": child_status,
            "description": task["description"],
            "result": task.get("result", {}),
            "recent_knowledge": recent_knowledge,
            "messages": [
                {"content": m["content"][:300]} for m in task_messages[:3]
            ],
        }

    def synthesize_results(self, task_ids: list[str]) -> dict[str, Any]:
        """Use LLM to synthesize results from multiple specialist tasks."""
        results: list[dict[str, Any]] = []
        for tid in task_ids:
            task = self.db.get_coordination_task(tid)
            if task:
                results.append({
                    "task_id": tid,
                    "task_type": task["task_type"],
                    "description": task["description"],
                    "status": task["status"],
                    "result": task.get("result", {}),
                })

        if not results:
            return {"error": "No tasks found"}

        if not self.inference:
            # Simple aggregation without LLM
            return {
                "synthesis": "LLM not available for synthesis",
                "task_count": len(results),
                "completed": sum(1 for r in results if r["status"] == "completed"),
                "results": results,
            }

        results_json = json.dumps(results, indent=2, default=str)
        resp = self.inference.chat(
            messages=[{
                "role": "user",
                "content": (
                    f"Synthesize the following results from specialist research agents:\n\n"
                    f"{results_json}\n\n"
                    f"Identify: 1) Key agreements across agents, "
                    f"2) Conflicts or contradictions, "
                    f"3) Recommendations for next steps.\n\n"
                    f"Return JSON with keys: synthesis (string), agreements (list), "
                    f"conflicts (list), recommendations (list)"
                ),
            }],
        )

        content = resp.content.strip()
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "synthesis": content[:2000],
            "agreements": [],
            "conflicts": [],
            "recommendations": [],
        }

    def get_team_status(self) -> dict[str, Any]:
        """Overview of all specialist children and their tasks."""
        roles = self.db.list_agent_roles()
        children = self.db.get_children()

        team: list[dict[str, Any]] = []
        for role_info in roles:
            child_id = role_info["child_id"]
            child = next((c for c in children if c["id"] == child_id), None)
            tasks = self.db.get_tasks_for_child(child_id)

            team.append({
                "child_id": child_id,
                "role": role_info["role"],
                "specialization": role_info.get("specialization", ""),
                "status": child.get("status", "unknown") if child else "not_found",
                "name": child.get("name", "") if child else "",
                "total_tasks": len(tasks),
                "pending_tasks": sum(1 for t in tasks if t["status"] == "pending"),
                "completed_tasks": sum(1 for t in tasks if t["status"] == "completed"),
            })

        return {
            "team_size": len(team),
            "members": team,
        }

    def list_roles(self) -> list[dict[str, Any]]:
        """List available roles with descriptions."""
        return [
            {
                "role": name,
                "genesis_summary": profile["genesis"][:200],
                "tools_emphasis": profile["tools_emphasis"],
            }
            for name, profile in ROLE_PROFILES.items()
        ]
