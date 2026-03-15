"""The core ReAct agent loop - think, act, observe, persist.

Port of automaton/src/agent/loop.ts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from ulid import ULID

from .config import DaemonConfig
from .context import AgentTurn, build_context_messages, trim_context
from .cost_tracker import estimate_cost_cents, get_budget_tier, get_total_spent
from .inference import InferenceClient
from .system_prompt import build_system_prompt, build_wakeup_prompt
from .tools import ToolExecutor, get_all_tool_specs, get_all_tool_descriptions

log = logging.getLogger("daemon.loop")

MAX_TOOL_CALLS_PER_TURN = 10
MAX_CONSECUTIVE_ERRORS = 5
MAX_REPEATED_CALLS = 3
FORCED_EVAL_INTERVAL = 10   # turns between forced evaluation checks
GENTLE_REMINDER_INTERVAL = 20  # turns before gentle nudge about inactive hypotheses


def run_agent_loop(
    config: DaemonConfig,
    db: Any,
    inference: InferenceClient,
    tool_executor: ToolExecutor,
    on_state_change: Callable[[str], None] | None = None,
    on_turn_complete: Callable[[dict[str, Any]], None] | None = None,
    max_turns: int | None = None,
) -> None:
    """Run the agent loop. Returns when sleeping, dead, or max turns reached."""

    # Set start time
    if not db.get_kv("start_time"):
        db.set_kv("start_time", datetime.now(timezone.utc).isoformat())

    consecutive_errors = 0
    turn_number = 0
    # Tools blocked by loop detection - cleared after one turn of compliance
    blocked_tools: set[str] = set()

    # Transition to waking
    db.set_agent_state("waking")
    _fire(on_state_change, "waking")

    is_first_run = db.get_turn_count() == 0

    # Build wakeup prompt
    wakeup_input = build_wakeup_prompt(config, db)

    # Transition to running
    db.set_agent_state("running")
    _fire(on_state_change, "running")

    spent = get_total_spent(db)
    log.info("[WAKE UP] %s is alive. Spent: $%.2f", config.name, spent / 100)

    # The pending input for the next turn
    pending_input: dict[str, str] | None = {
        "content": wakeup_input,
        "source": "wakeup",
    }

    while True:
        if max_turns and turn_number >= max_turns:
            log.info("[LOOP] Max turns (%d) reached.", max_turns)
            db.set_agent_state("sleeping")
            _fire(on_state_change, "sleeping")
            break

        try:
            # Check sleep_until
            sleep_until = db.get_kv("sleep_until")
            if sleep_until:
                try:
                    wake_time = datetime.fromisoformat(sleep_until)
                    if wake_time > datetime.now(timezone.utc):
                        log.info("[SLEEP] Sleeping until %s", sleep_until)
                        db.set_agent_state("sleeping")
                        _fire(on_state_change, "sleeping")
                        break
                except ValueError:
                    pass

            # Check for unprocessed inbox messages
            if not pending_input:
                inbox = db.get_unprocessed_inbox(limit=5)
                if inbox:
                    formatted = "\n\n".join(
                        f"[Message from {m['from_agent']}]: {m['content']}"
                        for m in inbox
                    )
                    pending_input = {"content": formatted, "source": "agent"}
                    for m in inbox:
                        db.mark_inbox_processed(m["id"])

            # Check budget tier (skipped when using CLIProxy - subscription-based)
            using_proxy = inference.using_proxy
            tier = "normal" if using_proxy else get_budget_tier(db, config)
            if tier == "dead":
                log.info("[DEAD] Budget exhausted. Entering dead state.")
                db.set_agent_state("dead")
                _fire(on_state_change, "dead")
                break
            elif tier == "critical":
                log.info("[CRITICAL] Budget critically low.")
                db.set_agent_state("critical")
                _fire(on_state_change, "critical")
                inference.set_low_compute_mode(True)
            elif tier == "low_compute":
                db.set_agent_state("low_compute")
                _fire(on_state_change, "low_compute")
                inference.set_low_compute_mode(True)
            else:
                if db.get_agent_state() != "running":
                    db.set_agent_state("running")
                    _fire(on_state_change, "running")
                inference.set_low_compute_mode(False)

            # Build context
            recent_turns_raw = db.get_recent_turns(20)
            recent_turns = [_dict_to_agent_turn(t) for t in recent_turns_raw]
            recent_turns = trim_context(recent_turns)

            tool_descriptions = get_all_tool_descriptions()
            system_prompt = build_system_prompt(
                config, db, tool_descriptions, is_first_run=is_first_run,
            )

            # Loop detection
            if len(recent_turns) >= MAX_REPEATED_CALLS and not pending_input:
                loop_result = _detect_loop(recent_turns)
                if loop_result:
                    pending_input = loop_result["message"]
                    blocked_tools.update(loop_result["blocked"])

            messages = build_context_messages(system_prompt, recent_turns, pending_input)

            # Capture input before clearing
            current_input = pending_input
            pending_input = None

            # Inference call
            model = inference.get_default_model()
            log.info("[THINK] Calling %s...", model)

            tool_defs = get_all_tool_specs()
            response = inference.chat(messages, tools=tool_defs)

            # Build turn record
            # CLIProxy inference is subscription-based, no per-call cost
            cost = 0 if using_proxy else estimate_cost_cents(response.usage, model)

            turn_dict: dict[str, Any] = {
                "id": str(ULID()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state": db.get_agent_state(),
                "input": current_input["content"] if current_input else None,
                "input_source": current_input["source"] if current_input else None,
                "thinking": response.content or "",
                "tool_calls": [],
                "token_usage": response.usage,
                "cost_cents": cost,
            }

            # Record cost as transaction
            if turn_dict["cost_cents"] > 0:
                db.insert_transaction({
                    "id": str(ULID()),
                    "type": "inference",
                    "amount_cents": turn_dict["cost_cents"],
                    "description": f"Turn {turn_dict['id'][:8]} via {model}",
                })

            # Execute tool calls
            used_blocked_tool = False
            if response.tool_calls:
                call_count = 0
                for tc in response.tool_calls:
                    if call_count >= MAX_TOOL_CALLS_PER_TURN:
                        log.info("[TOOLS] Max tool calls per turn reached (%d)", MAX_TOOL_CALLS_PER_TURN)
                        break

                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        args = {}

                    tool_name = tc["function"]["name"]

                    # Enforce loop detection block
                    if tool_name in blocked_tools:
                        log.info("[BLOCKED] %s is blocked by loop detection. Skipping.", tool_name)
                        result = {
                            "id": tc["id"],
                            "name": tool_name,
                            "arguments": args,
                            "result": json.dumps({
                                "ok": False,
                                "error": (
                                    f"BLOCKED: '{tool_name}' was called too many times in a row. "
                                    f"Try a different tool or approach."
                                ),
                            }),
                            "duration_ms": 0,
                            "error": None,
                        }
                        turn_dict["tool_calls"].append(result)
                        used_blocked_tool = True
                        call_count += 1
                        continue

                    log.info("[TOOL] %s(%s)", tool_name, json.dumps(args)[:100])

                    result = tool_executor.execute(tool_name, args)
                    # Override ID to match inference call's ID
                    result["id"] = tc["id"]
                    turn_dict["tool_calls"].append(result)

                    log.info(
                        "[TOOL RESULT] %s: %s",
                        tool_name,
                        f"ERROR: {result['error']}" if result.get("error") else result["result"][:200],
                    )
                    call_count += 1

            # Clear blocked tools after a turn where the agent didn't try them
            if not used_blocked_tool:
                blocked_tools.clear()

            # Track conclude_hypothesis calls for forced evaluation
            for tc in turn_dict["tool_calls"]:
                if tc["name"] == "conclude_hypothesis" and not tc.get("error"):
                    db.set_kv("last_conclude_turn", str(db.get_turn_count()))

            # Persist turn
            db.insert_turn(turn_dict)
            for tc in turn_dict["tool_calls"]:
                db.insert_tool_call(turn_dict["id"], tc)

            if on_turn_complete:
                on_turn_complete(turn_dict)

            if turn_dict["thinking"]:
                log.info("[THOUGHT] %s", turn_dict["thinking"][:300])

            turn_number += 1

            # Forced evaluation check
            if not pending_input:
                pending_input = _check_forced_evaluation(db, db.get_turn_count())

            # Check for sleep command
            sleep_tool = next(
                (tc for tc in turn_dict["tool_calls"] if tc["name"] == "sleep"),
                None,
            )
            if sleep_tool and not sleep_tool.get("error"):
                log.info("[SLEEP] Agent chose to sleep.")
                db.set_agent_state("sleeping")
                _fire(on_state_change, "sleeping")
                break

            # If no tool calls and finish_reason=stop -> idle, brief sleep
            if (
                not response.tool_calls
                and response.finish_reason == "stop"
            ):
                log.info("[IDLE] No pending inputs. Entering brief sleep.")
                wake_at = datetime.now(timezone.utc).timestamp() + 60
                db.set_kv(
                    "sleep_until",
                    datetime.fromtimestamp(wake_at, tz=timezone.utc).isoformat(),
                )
                db.set_agent_state("sleeping")
                _fire(on_state_change, "sleeping")
                break

            consecutive_errors = 0

        except Exception as exc:
            consecutive_errors += 1
            log.error("[ERROR] Turn failed: %s", exc)

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error(
                    "[FATAL] %d consecutive errors. Sleeping.",
                    MAX_CONSECUTIVE_ERRORS,
                )
                db.set_agent_state("sleeping")
                _fire(on_state_change, "sleeping")
                wake_at = datetime.now(timezone.utc).timestamp() + 300
                db.set_kv(
                    "sleep_until",
                    datetime.fromtimestamp(wake_at, tz=timezone.utc).isoformat(),
                )
                break

    log.info("[LOOP END] Agent loop finished. State: %s", db.get_agent_state())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fire(callback: Callable[[str], None] | None, state: str) -> None:
    if callback:
        callback(state)


def _dict_to_agent_turn(d: dict[str, Any]) -> AgentTurn:
    return AgentTurn(
        id=d["id"],
        timestamp=d["timestamp"],
        state=d["state"],
        input=d.get("input"),
        input_source=d.get("input_source"),
        thinking=d["thinking"],
        tool_calls=d.get("tool_calls", []),
        token_usage=d.get("token_usage", {}),
        cost_cents=d.get("cost_cents", 0),
    )


def _detect_loop(recent_turns: list[AgentTurn]) -> dict[str, Any] | None:
    """Detect repeated tool calls. Returns message + set of tool names to block."""
    last_n = recent_turns[-MAX_REPEATED_CALLS:]

    # Check 1: exact same tool+args repeated
    signatures = []
    for t in last_n:
        sig = "|".join(
            f"{tc['name']}:{json.dumps(tc.get('arguments', {}), sort_keys=True)}"
            for tc in t.tool_calls
        )
        signatures.append(sig)

    if signatures and all(s == signatures[0] for s in signatures) and signatures[0]:
        blocked = {
            tc["name"]
            for t in last_n
            for tc in t.tool_calls
        }
        tool_name = last_n[0].tool_calls[0]["name"] if last_n[0].tool_calls else "unknown"
        log.info("[LOOP DETECT] Repeated tool call: %s x%d - BLOCKING", tool_name, MAX_REPEATED_CALLS)
        return {
            "message": {
                "content": (
                    f"WARNING: You called {tool_name} {MAX_REPEATED_CALLS} times in a row "
                    f"with the same arguments. It is now BLOCKED for this turn. "
                    f"Try a completely different approach or move on to a different task."
                ),
                "source": "system",
            },
            "blocked": blocked,
        }

    # Check 2: same tool name called too many times in a window
    window = recent_turns[-6:]
    tool_names: list[str] = []
    for t in window:
        for tc in t.tool_calls:
            tool_names.append(tc["name"])

    counts: dict[str, int] = {}
    for name in tool_names:
        counts[name] = counts.get(name, 0) + 1

    for name, count in counts.items():
        if count >= 5:
            log.info("[LOOP DETECT] Tool '%s' called %d times in last 6 turns - BLOCKING", name, count)
            return {
                "message": {
                    "content": (
                        f"WARNING: You called '{name}' {count} times in the last 6 turns "
                        f"with little progress. It is now BLOCKED for this turn. "
                        f"Take a concrete action: run an experiment, save a finding, "
                        f"or sleep if you have no clear goal."
                    ),
                    "source": "system",
                },
                "blocked": {name},
            }

    return None


def _check_forced_evaluation(db: Any, global_turn_count: int) -> dict[str, str] | None:
    """Check if the agent should be forced to evaluate hypotheses.

    Uses global turn count (monotonic across wake cycles) to avoid the
    reset problem with local turn_number.
    """
    last_conclude_raw = db.get_kv("last_conclude_turn")
    last_conclude = int(last_conclude_raw) if last_conclude_raw else 0
    turns_since = global_turn_count - last_conclude

    if turns_since < FORCED_EVAL_INTERVAL:
        return None

    # Check if there are hypotheses with findings but no conclusion
    needs_conclusion = db.get_hypotheses_with_findings_no_conclusion()
    if needs_conclusion:
        names = [h["content"][:100] for h in needs_conclusion[:3]]
        return {
            "content": (
                "EVALUATION REQUIRED: You have not concluded any hypotheses in "
                f"{turns_since} turns. The following hypotheses have findings "
                "but no conclusion - you MUST use conclude_hypothesis on at least one:\n"
                + "\n".join(f"  - [{h['id']}] {h['content'][:100]}" for h in needs_conclusion[:3])
            ),
            "source": "system",
        }

    # Gentle reminder about active hypotheses with no findings after many turns
    if turns_since >= GENTLE_REMINDER_INTERVAL:
        active = db.get_active_hypotheses()
        if active:
            return {
                "content": (
                    f"Reminder: You have {len(active)} active hypothesis/hypotheses "
                    f"and have not concluded any in {turns_since} turns. "
                    "Consider running experiments to gather findings, or conclude "
                    "hypotheses that have sufficient evidence."
                ),
                "source": "system",
            }

    return None
