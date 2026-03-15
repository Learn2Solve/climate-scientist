"""Agent state machine - mirrors automaton/src/types.ts AgentState."""

from __future__ import annotations

from enum import Enum


class AgentState(str, Enum):
    SETUP = "setup"
    WAKING = "waking"
    RUNNING = "running"
    SLEEPING = "sleeping"
    LOW_COMPUTE = "low_compute"
    CRITICAL = "critical"
    DEAD = "dead"


# Valid state transitions. Key = current state, value = set of reachable states.
VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.SETUP: {AgentState.WAKING},
    AgentState.WAKING: {AgentState.RUNNING, AgentState.LOW_COMPUTE, AgentState.CRITICAL, AgentState.DEAD},
    AgentState.RUNNING: {AgentState.SLEEPING, AgentState.LOW_COMPUTE, AgentState.CRITICAL, AgentState.DEAD},
    AgentState.SLEEPING: {AgentState.WAKING, AgentState.DEAD},
    AgentState.LOW_COMPUTE: {AgentState.RUNNING, AgentState.SLEEPING, AgentState.CRITICAL, AgentState.DEAD},
    AgentState.CRITICAL: {AgentState.RUNNING, AgentState.SLEEPING, AgentState.LOW_COMPUTE, AgentState.DEAD},
    AgentState.DEAD: {AgentState.WAKING},  # resurrection possible if budget restored
}


def can_transition(current: AgentState, target: AgentState) -> bool:
    return target in VALID_TRANSITIONS.get(current, set())
