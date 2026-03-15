"""Child agent spawning and management.

Port of automaton's child spawning (types.ts children table + replication tools).
Each child is a separate daemon process with its own config, DB, and research topic.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ulid import ULID

from .config import DaemonConfig, AGENT_DIR


def spawn_child(
    parent_config: DaemonConfig,
    parent_db: Any,
    name: str,
    topic: str,
    budget_cents: int = 500,
    genesis_prompt: str = "",
) -> dict[str, Any]:
    """Create and start a child agent.

    1. Create ~/.climate_agent/children/{name}/ directory
    2. Write child config.yml (inherits API keys, sets topic + budget)
    3. Write child identity.md with genesis_prompt
    4. Start child as subprocess
    5. Record child in parent's DB
    """
    if not name:
        raise ValueError("Child name required")
    if not topic:
        raise ValueError("Child topic required")

    child_dir = AGENT_DIR / "children" / name
    child_dir.mkdir(parents=True, exist_ok=True)

    child_id = str(ULID())

    # Write child config
    child_config = {
        "name": name,
        "topic": topic,
        "repo_root": str(parent_config.repo_root),
        "db_path": str(child_dir / "state.db"),
        "identity_path": str(child_dir / "identity.md"),
        "inference_model": parent_config.inference_model,
        "low_compute_model": parent_config.low_compute_model,
        "max_tokens_per_turn": parent_config.max_tokens_per_turn,
        "openai_api_key": parent_config.openai_api_key,
        "anthropic_api_key": parent_config.anthropic_api_key,
        "heartbeat_config_path": str(child_dir / "heartbeat.yml"),
        "heartbeat_interval_s": parent_config.heartbeat_interval_s,
        "data_dir": str(parent_config.data_dir),
        "runs_root": str(child_dir / "runs"),
        "max_api_cost_cents": budget_cents,
        "max_turns_per_wake": parent_config.max_turns_per_wake,
        "allow_web": parent_config.allow_web,
        "allow_code_edit": parent_config.allow_code_edit,
        "skills_dir": str(child_dir / "skills"),
        "log_level": parent_config.log_level,
        "version": parent_config.version,
    }

    config_path = child_dir / "config.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(child_config, f, default_flow_style=False)

    # Write identity
    identity_text = (
        f"# {name}\n\n"
        f"I am a child research agent specializing in: {topic}\n\n"
        f"My parent agent is {parent_config.name}, researching {parent_config.topic}.\n"
    )
    if genesis_prompt:
        identity_text += f"\n## Genesis Instructions\n\n{genesis_prompt}\n"

    (child_dir / "identity.md").write_text(identity_text, encoding="utf-8")

    # Create subdirectories
    (child_dir / "skills").mkdir(exist_ok=True)
    (child_dir / "runs").mkdir(exist_ok=True)

    # Start child process
    daemon_script = Path(__file__).resolve().parent.parent / "daemon.py"
    cmd = [
        sys.executable, str(daemon_script),
        "--run",
        "--config", str(config_path),
        "--topic", topic,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Record in parent DB
    child_record = {
        "id": child_id,
        "name": name,
        "topic": topic,
        "pid": proc.pid,
        "config_path": str(config_path),
        "status": "running",
        "budget_cents": budget_cents,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    parent_db.insert_child(child_record)

    # Record as modification
    parent_db.insert_modification({
        "id": str(ULID()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "child_spawn",
        "description": f"Spawned child '{name}' for topic '{topic}' with ${budget_cents / 100:.2f} budget",
        "reversible": True,
    })

    return child_record


def check_child_status(db: Any, child_id: str) -> dict[str, Any]:
    """Check if child process is alive and read its state."""
    child = db.get_child_by_id(child_id)
    if not child:
        return {"status": "unknown", "error": "child not found"}

    pid = child.get("pid")
    alive = False
    if pid:
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            alive = True
        except (OSError, ProcessLookupError):
            alive = False

    if not alive and child.get("status") == "running":
        db.update_child_status(child_id, "stopped")

    # Read child's state.db for status
    config_path = Path(child.get("config_path", ""))
    child_dir = config_path.parent
    child_db_path = child_dir / "state.db"

    child_state = "unknown"
    child_turns = 0
    if child_db_path.exists():
        try:
            conn = sqlite3.connect(str(child_db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT value FROM kv WHERE key = 'agent_state'").fetchone()
            if row:
                child_state = row["value"]
            count_row = conn.execute("SELECT COUNT(*) as c FROM turns").fetchone()
            if count_row:
                child_turns = count_row["c"]
            conn.close()
        except Exception:
            pass

    status = "running" if alive else "stopped"
    db.update_child_status(child_id, status)

    return {
        "status": status,
        "alive": alive,
        "pid": pid,
        "child_state": child_state,
        "child_turns": child_turns,
    }


def send_to_child(parent_db: Any, child_id: str, message: str, sender_name: str = "parent") -> None:
    """Write message to child's inbox_messages table (direct SQLite access)."""
    child = parent_db.get_child_by_id(child_id)
    if not child:
        raise ValueError(f"Child not found: {child_id}")

    config_path = Path(child["config_path"])
    child_dir = config_path.parent
    child_db_path = child_dir / "state.db"

    if not child_db_path.exists():
        raise ValueError(f"Child database not found: {child_db_path}")

    msg_id = str(ULID())
    conn = sqlite3.connect(str(child_db_path))
    try:
        conn.execute(
            "INSERT OR IGNORE INTO inbox_messages (id, from_agent, content, received_at) "
            "VALUES (?, ?, ?, ?)",
            (msg_id, sender_name, message, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def read_child_knowledge(config_path: Path) -> list[dict[str, Any]]:
    """Read knowledge entries from a child's state.db."""
    child_dir = config_path.parent
    child_db_path = child_dir / "state.db"

    if not child_db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(child_db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM knowledge ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def stop_child(db: Any, child_id: str) -> None:
    """Send SIGTERM to child process."""
    child = db.get_child_by_id(child_id)
    if not child:
        raise ValueError(f"Child not found: {child_id}")

    pid = child.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    db.update_child_status(child_id, "stopped")
