"""Context window management - port of automaton/src/agent/context.ts.

Manages conversation history for the agent loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


MAX_CONTEXT_TURNS = 20


@dataclass
class AgentTurn:
    id: str
    timestamp: str
    state: str
    thinking: str
    tool_calls: list[dict] = field(default_factory=list)
    token_usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
    cost_cents: int = 0
    input: str | None = None
    input_source: str | None = None


def build_context_messages(
    system_prompt: str,
    recent_turns: list[AgentTurn],
    pending_input: dict | None = None,
) -> list[dict]:
    """Build the message array for the next inference call.

    Includes system prompt, recent conversation history, and any pending input.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for turn in recent_turns:
        # The turn's input (if any) as a user message
        if turn.input:
            source = turn.input_source or "system"
            messages.append({
                "role": "user",
                "content": f"[{source}] {turn.input}",
            })

        # The agent's thinking as assistant message
        if turn.thinking:
            msg: dict = {
                "role": "assistant",
                "content": turn.thinking,
            }

            # If there were tool calls, include them
            if turn.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    }
                    for tc in turn.tool_calls
                ]
            messages.append(msg)

            # Add tool results
            for tc in turn.tool_calls:
                content = f"Error: {tc['error']}" if tc.get("error") else tc.get("result", "")
                messages.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tc["id"],
                })

    # Add pending input if any
    if pending_input:
        messages.append({
            "role": "user",
            "content": f"[{pending_input['source']}] {pending_input['content']}",
        })

    return messages


def trim_context(
    turns: list[AgentTurn],
    max_turns: int = MAX_CONTEXT_TURNS,
) -> list[AgentTurn]:
    """Trim context to fit within limits. Keeps the most recent turns."""
    if len(turns) <= max_turns:
        return turns
    return turns[-max_turns:]
