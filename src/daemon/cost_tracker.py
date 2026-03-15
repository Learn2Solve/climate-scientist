"""API cost tracking - replaces automaton Conway credits with local budget accounting.

Port of automaton/src/conway/inference.ts cost logic, stripped of Conway markup
and survival tiers. Tracks spending against a local budget ceiling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from daemon.config import DaemonConfig
    from daemon.database import DaemonDatabase

# Pricing: cents per million tokens (input, output).
PRICING: dict[str, dict[str, int]] = {
    "gpt-4o":             {"input": 250,  "output": 1000},
    "gpt-4o-mini":        {"input": 15,   "output": 60},
    "gpt-4.1":            {"input": 200,  "output": 800},
    "gpt-4.1-mini":       {"input": 40,   "output": 160},
    "gpt-4.1-nano":       {"input": 10,   "output": 40},
    "gpt-5.2":            {"input": 200,  "output": 800},
    "o1":                 {"input": 1500, "output": 6000},
    "o3-mini":            {"input": 110,  "output": 440},
    "o4-mini":            {"input": 110,  "output": 440},
    "claude-sonnet-4-5":           {"input": 300,  "output": 1500},
    "claude-sonnet-4-5-20250514":  {"input": 300,  "output": 1500},
    "claude-sonnet-4-20250514":    {"input": 300,  "output": 1500},
    "claude-haiku-4-5":            {"input": 100,  "output": 500},
    "claude-haiku-4-20250414":     {"input": 100,  "output": 500},
}

_DEFAULT_MODEL = "gpt-4o"


def estimate_cost_cents(usage: dict[str, Any], model: str) -> int:
    """Estimate cost in cents from a token usage dict and model name.

    Args:
        usage: Dict with "prompt_tokens" and "completion_tokens" keys.
        model: Model identifier string.

    Returns:
        Cost in whole cents, rounded up.
    """
    prices = PRICING.get(model, PRICING[_DEFAULT_MODEL])
    prompt_tokens: int = usage.get("prompt_tokens", 0)
    completion_tokens: int = usage.get("completion_tokens", 0)

    input_cost = prompt_tokens * prices["input"] / 1_000_000
    output_cost = completion_tokens * prices["output"] / 1_000_000
    return math.ceil(input_cost + output_cost)


def get_total_spent(db: DaemonDatabase) -> int:
    """Return total cents spent on inference from the transactions table."""
    row = db._conn.execute(
        "SELECT COALESCE(SUM(amount_cents), 0) FROM transactions WHERE type = 'inference'"
    ).fetchone()
    return int(row[0])


def get_budget_tier(db: DaemonDatabase, config: DaemonConfig) -> str:
    """Determine the current budget tier based on spending vs configured ceiling.

    Tiers:
        "normal"      - spent < 50% of budget
        "low_compute" - spent < 80% of budget
        "critical"    - spent < 95% of budget
        "dead"        - spent >= 95% of budget
    """
    spent = get_total_spent(db)
    budget = config.max_api_cost_cents

    if budget <= 0:
        return "dead"

    ratio = spent / budget
    if ratio < 0.50:
        return "normal"
    if ratio < 0.80:
        return "low_compute"
    if ratio < 0.95:
        return "critical"
    return "dead"
