#!/usr/bin/env python3
"""Climate Research Agent Daemon - main entry point.

Port of automaton/src/index.ts.
Runs as a long-lived daemon that autonomously researches a given topic.

Usage:
    uv run python src/daemon.py --run --topic "rapid intensification"
    uv run python src/daemon.py --run --max-cycles 3
    uv run python src/daemon.py --status
    uv run python src/daemon.py --init
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VERSION = "0.1.0"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("daemon")


def main() -> None:
    parser = argparse.ArgumentParser(description="Climate Research Agent Daemon")
    parser.add_argument("--run", action="store_true", help="Start the daemon")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--init", action="store_true", help="Initialize ~/.climate_agent/")
    parser.add_argument("--topic", type=str, help="Research topic")
    parser.add_argument("--max-cycles", type=int, default=0, help="Max wake cycles (0=unlimited)")
    parser.add_argument("--max-turns", type=int, default=0, help="Max turns per wake cycle")
    parser.add_argument("--heartbeat-only", action="store_true", help="Run one heartbeat tick then exit")
    parser.add_argument("--config", type=str, help="Path to config.yml")
    parser.add_argument("--version", action="store_true", help="Show version")
    args = parser.parse_args()

    if args.version:
        print(f"Climate Research Agent Daemon v{VERSION}")
        sys.exit(0)

    if args.init:
        cmd_init()
        sys.exit(0)

    if args.status:
        cmd_status(args.config)
        sys.exit(0)

    if args.heartbeat_only:
        cmd_heartbeat_only(args.config, args.topic)
        sys.exit(0)

    if args.run:
        cmd_run(args.config, args.topic, args.max_cycles, args.max_turns)
        return

    parser.print_help()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_init() -> None:
    from daemon.config import init_agent_dir

    repo_root = Path(__file__).resolve().parent.parent
    agent_dir = init_agent_dir(repo_root)
    print(f"Initialized agent directory: {agent_dir}")

    # Create default identity.md
    identity_path = agent_dir / "identity.md"
    if not identity_path.exists():
        identity_path.write_text(
            "# Climate Research Agent\n\n"
            "I am an autonomous research agent investigating climate science topics.\n"
            "I accumulate knowledge, run experiments, and produce findings.\n",
            encoding="utf-8",
        )
        print(f"Created identity: {identity_path}")

    # Create default heartbeat.yml
    hb_path = agent_dir / "heartbeat.yml"
    if not hb_path.exists():
        import yaml
        from daemon.heartbeat import DEFAULT_HEARTBEAT_ENTRIES

        yaml.safe_dump(
            {"entries": DEFAULT_HEARTBEAT_ENTRIES},
            open(hb_path, "w", encoding="utf-8"),
            default_flow_style=False,
        )
        print(f"Created heartbeat config: {hb_path}")

    print("Done. Run with: uv run python src/daemon.py --run --topic '<your topic>'")


def cmd_status(config_path: str | None) -> None:
    from daemon.config import load_config
    from daemon.database import DaemonDatabase
    from daemon.cost_tracker import get_total_spent, get_budget_tier

    cfg = load_config(Path(config_path) if config_path else None)
    if not cfg:
        print("Agent not configured. Run: uv run python src/daemon.py --init")
        return

    db = DaemonDatabase(cfg.resolved_db_path)
    state = db.get_agent_state()
    turns = db.get_turn_count()
    spent = get_total_spent(db)
    tier = get_budget_tier(db, cfg)
    children = db.get_children()
    alive = [c for c in children if c.get("status") not in ("dead", "stopped")]
    peers = db.get_peers()
    knowledge_count = len(db.get_recent_knowledge(limit=10000))

    print(f"""
=== CLIMATE RESEARCH AGENT STATUS ===
Name:       {cfg.name}
Topic:      {cfg.topic}
State:      {state}
Turns:      {turns}
Cost:       ${spent / 100:.2f} / ${cfg.max_api_cost_cents / 100:.2f} ({tier})
Knowledge:  {knowledge_count} entries
Children:   {len(alive)} alive / {len(children)} total
Peers:      {len(peers)}
Model:      {cfg.inference_model}
Version:    {cfg.version}
DB:         {cfg.resolved_db_path}
=====================================
""")

    db.close()


def cmd_heartbeat_only(config_path: str | None, topic: str | None) -> None:
    cfg, db = _boot(config_path, topic)
    from daemon.heartbeat import HeartbeatDaemon, sync_heartbeat_to_db

    sync_heartbeat_to_db(db, cfg.resolved_heartbeat_config_path)
    daemon = HeartbeatDaemon(
        cfg, db,
        on_wake_request=lambda reason: log.info("[HEARTBEAT] Wake request: %s", reason),
    )
    log.info("Running one heartbeat tick...")
    daemon._tick()
    log.info("Done.")
    db.close()


def cmd_run(
    config_path: str | None,
    topic: str | None,
    max_cycles: int,
    max_turns_per_cycle: int,
) -> None:
    cfg, db = _boot(config_path, topic)

    log.info("Climate Research Agent v%s starting...", VERSION)
    log.info("Topic: %s", cfg.topic)
    log.info("Model: %s", cfg.inference_model)
    log.info("DB: %s", cfg.resolved_db_path)

    # Store identity
    db.set_identity("name", cfg.name)
    db.set_identity("topic", cfg.topic)

    # Create inference client
    from daemon.inference import InferenceClient

    inference = InferenceClient(
        default_model=cfg.inference_model,
        max_tokens=cfg.max_tokens_per_turn,
        openai_api_key=cfg.openai_api_key,
        anthropic_api_key=cfg.anthropic_api_key,
        low_compute_model=cfg.low_compute_model,
        cli_proxy_url=cfg.cli_proxy_url,
        cli_proxy_api_key=cfg.cli_proxy_api_key,
        reasoning_effort=cfg.reasoning_effort,
    )

    if cfg.cli_proxy_url:
        log.info("CLIProxy enabled: %s (inference costs bypassed)", cfg.cli_proxy_url)

    # Create tool executor
    from daemon.tools import ToolExecutor

    run_dir = cfg.runs_root / f"daemon_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    tool_executor = ToolExecutor(cfg, db, run_dir, run_dir.name, inference=inference)

    # Sync heartbeat config
    from daemon.heartbeat import HeartbeatDaemon, sync_heartbeat_to_db

    sync_heartbeat_to_db(db, cfg.resolved_heartbeat_config_path)

    # Start heartbeat daemon
    heartbeat = HeartbeatDaemon(
        cfg, db,
        on_wake_request=lambda reason: _handle_wake_request(db, reason),
    )
    heartbeat.start()

    # Signal handlers for graceful shutdown
    shutdown_requested = False

    def handle_signal(signum: int, frame: Any) -> None:
        nonlocal shutdown_requested
        log.info("Shutdown signal received (sig=%d).", signum)
        shutdown_requested = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Clear stale state from previous run
    db.delete_kv("sleep_until")
    db.delete_kv("wake_request")

    # Main run loop
    from daemon.loop import run_agent_loop

    cycle = 0
    try:
        while not shutdown_requested:
            if max_cycles > 0 and cycle >= max_cycles:
                log.info("Max cycles (%d) reached. Exiting.", max_cycles)
                break

            try:
                run_agent_loop(
                    config=cfg,
                    db=db,
                    inference=inference,
                    tool_executor=tool_executor,
                    on_state_change=lambda s: log.info("[STATE] %s", s),
                    on_turn_complete=_log_turn,
                    max_turns=max_turns_per_cycle or cfg.max_turns_per_wake or None,
                )
            except Exception as exc:
                log.error("Fatal error in agent loop: %s", exc)
                time.sleep(30)
                continue

            cycle += 1
            state = db.get_agent_state()

            if state == "dead":
                log.info("Agent is dead (budget exhausted). Heartbeat continues.")
                _sleep_with_wake_check(db, 300, shutdown_requested_fn=lambda: shutdown_requested)
                continue

            if state == "sleeping":
                sleep_until_str = db.get_kv("sleep_until")
                if sleep_until_str:
                    try:
                        wake_time = datetime.fromisoformat(sleep_until_str)
                        sleep_s = max((wake_time - datetime.now(timezone.utc)).total_seconds(), 10)
                    except ValueError:
                        sleep_s = 60
                else:
                    sleep_s = 60

                log.info("Sleeping for %ds...", int(sleep_s))
                woken = _sleep_with_wake_check(
                    db, sleep_s,
                    shutdown_requested_fn=lambda: shutdown_requested,
                )
                if woken:
                    log.info("Woken by heartbeat.")
                db.delete_kv("sleep_until")

    finally:
        log.info("Shutting down...")
        heartbeat.stop()
        db.set_agent_state("sleeping")
        db.close()
        log.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _boot(config_path: str | None, topic: str | None) -> tuple:
    """Load or create config and initialize database."""
    from daemon.config import load_config, create_default_config, save_config, init_agent_dir
    from daemon.database import DaemonDatabase

    cfg = load_config(Path(config_path) if config_path else None)
    if not cfg:
        repo_root = Path(__file__).resolve().parent.parent
        init_agent_dir(repo_root)
        cfg = create_default_config(
            topic=topic or "rapid intensification",
            repo_root=repo_root,
        )
        save_config(cfg)
        log.info("Created default config at %s", cfg.agent_dir / "config.yml")

    # Override topic from CLI if provided
    if topic and topic != cfg.topic:
        from dataclasses import replace
        cfg = replace(cfg, topic=topic)

    db = DaemonDatabase(cfg.resolved_db_path)
    return cfg, db


def _handle_wake_request(db: Any, reason: str) -> None:
    log.info("[HEARTBEAT] Wake request: %s", reason)
    db.set_kv("wake_request", reason)


def _sleep_with_wake_check(
    db: Any,
    total_seconds: float,
    shutdown_requested_fn: Any = None,
    check_interval: float = 30,
) -> bool:
    """Sleep for total_seconds, checking for wake requests every check_interval.
    Returns True if woken early by a wake request."""
    slept = 0.0
    while slept < total_seconds:
        chunk = min(check_interval, total_seconds - slept)
        time.sleep(chunk)
        slept += chunk

        if shutdown_requested_fn and shutdown_requested_fn():
            return False

        wake_request = db.get_kv("wake_request")
        if wake_request:
            db.delete_kv("wake_request")
            db.delete_kv("sleep_until")
            return True

    return False


def _log_turn(turn: dict[str, Any]) -> None:
    tool_count = len(turn.get("tool_calls", []))
    usage = turn.get("token_usage", {})
    tokens = usage.get("total_tokens", 0)
    log.info(
        "[TURN] %s: %d tools, %d tokens, %dc",
        turn["id"][:8],
        tool_count,
        tokens,
        turn.get("cost_cents", 0),
    )


# Fix import path when run from src/
if __name__ == "__main__":
    # Add parent of src/ to path so "from daemon.xxx" works
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    main()
