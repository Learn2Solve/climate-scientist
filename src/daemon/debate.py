"""Multi-round adversarial debate for verifying research findings.

Runs a structured debate between a proposer (defends a finding) and an
opponent (challenges it), then a judge renders a verdict. Uses different
models for proposer/opponent to ensure diversity of reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.debate")


PROPOSER_SYSTEM = """You are a scientific debate proposer. Your role is to DEFEND
a research finding as valid and significant. Present evidence, reasoning, and
methodology that support this finding. Be specific and rigorous."""

OPPONENT_SYSTEM = """You are a scientific debate opponent. Your role is to CHALLENGE
a research finding. Look for:
- Methodological flaws
- Alternative explanations
- Confounding variables
- Statistical issues
- Missing controls
- Overgeneralization
Be specific and constructive."""

JUDGE_SYSTEM = """You are a scientific debate judge. Given a finding, the proposer's
defense, and the opponent's challenges, assess the finding's validity.

You MUST respond with ONLY a JSON object:
{
    "verdict": "valid" | "flawed" | "inconclusive",
    "confidence": 0.0-1.0,
    "reasoning": "Brief synthesis",
    "key_issues": ["issue1", "issue2"],
    "recommendations": ["rec1", "rec2"]
}"""


class DebateEngine:
    """Adversarial debate engine for research finding verification."""

    def __init__(self, inference: Any, db: Any, config: Any) -> None:
        self._inference = inference
        self._db = db
        self._config = config

    def start_debate(
        self,
        topic: str,
        debate_type: str = "verification",
        linked_finding_id: str | None = None,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        """Run a full multi-round debate on a topic.

        1. Create debate session in DB.
        2. Run max_rounds of proposer/opponent exchanges.
        3. Judge evaluates all arguments.
        4. Store verdict.

        Returns {"session_id": str, "verdict": str, "confidence": float, "rounds": int}.
        """
        rounds = max_rounds or self._config.debate_max_rounds
        session_id = str(ULID())

        # Create session
        self._db.insert_debate_session({
            "id": session_id,
            "topic": topic,
            "debate_type": debate_type,
            "status": "active",
            "max_rounds": rounds,
            "linked_finding_id": linked_finding_id,
        })
        log.info("Starting debate %s: '%s' (%d rounds)", session_id, topic[:80], rounds)

        # Run debate rounds
        history: list[dict[str, str]] = []
        for round_num in range(1, rounds + 1):
            proposer_arg, opponent_arg = self._run_round(
                session_id, round_num, topic, history,
            )
            history.append({"round": str(round_num), "proposer": proposer_arg, "opponent": opponent_arg})

        # Judge the debate
        verdict = self._judge_debate(session_id, topic, history)

        # Store verdict
        self._db.update_debate_session(
            session_id,
            status="completed",
            verdict=verdict.get("verdict", "inconclusive"),
            verdict_confidence=verdict.get("confidence", 0.5),
            verdict_reasoning=verdict.get("reasoning", ""),
            rounds=rounds,
        )

        log.info(
            "Debate %s completed: verdict=%s confidence=%.2f",
            session_id,
            verdict.get("verdict", "inconclusive"),
            verdict.get("confidence", 0.5),
        )

        return {
            "session_id": session_id,
            "verdict": verdict.get("verdict", "inconclusive"),
            "confidence": verdict.get("confidence", 0.5),
            "rounds": rounds,
            "key_issues": verdict.get("key_issues", []),
            "recommendations": verdict.get("recommendations", []),
            "reasoning": verdict.get("reasoning", ""),
        }

    def _run_round(
        self,
        session_id: str,
        round_number: int,
        topic: str,
        history: list[dict[str, str]],
    ) -> tuple[str, str]:
        """Run one debate round: proposer argues, then opponent rebuts.

        Returns (proposer_argument, opponent_argument).
        """
        proposer_model = self._config.debate_proposer_model
        opponent_model = self._config.debate_opponent_model

        # Build history context
        history_text = ""
        if history:
            parts = []
            for h in history:
                parts.append(
                    f"--- Round {h['round']} ---\n"
                    f"Proposer: {h['proposer']}\n"
                    f"Opponent: {h['opponent']}"
                )
            history_text = "\n\n".join(parts)

        # Proposer argues
        proposer_prompt = f"FINDING TO DEFEND:\n{topic}"
        if history_text:
            proposer_prompt += f"\n\nPREVIOUS ROUNDS:\n{history_text}"
        proposer_prompt += f"\n\nThis is round {round_number}. Present your argument."

        log.info("Debate %s round %d: proposer (%s)", session_id, round_number, proposer_model)
        proposer_resp = self._inference.chat(
            messages=[
                {"role": "system", "content": PROPOSER_SYSTEM},
                {"role": "user", "content": proposer_prompt},
            ],
            model=proposer_model,
            temperature=0.7,
            max_tokens=2000,
        )
        proposer_arg = proposer_resp.content

        # Store proposer argument
        self._db.insert_debate_argument({
            "id": str(ULID()),
            "session_id": session_id,
            "role": "proposer",
            "model": proposer_model,
            "argument": proposer_arg,
            "evidence": [],
            "round_number": round_number,
        })

        # Opponent rebuts
        opponent_prompt = f"FINDING BEING DEFENDED:\n{topic}"
        if history_text:
            opponent_prompt += f"\n\nPREVIOUS ROUNDS:\n{history_text}"
        opponent_prompt += (
            f"\n\n--- Round {round_number} ---\n"
            f"PROPOSER'S ARGUMENT:\n{proposer_arg}\n\n"
            f"Present your rebuttal."
        )

        log.info("Debate %s round %d: opponent (%s)", session_id, round_number, opponent_model)
        opponent_resp = self._inference.chat(
            messages=[
                {"role": "system", "content": OPPONENT_SYSTEM},
                {"role": "user", "content": opponent_prompt},
            ],
            model=opponent_model,
            temperature=0.3,
            max_tokens=2000,
        )
        opponent_arg = opponent_resp.content

        # Store opponent argument
        self._db.insert_debate_argument({
            "id": str(ULID()),
            "session_id": session_id,
            "role": "opponent",
            "model": opponent_model,
            "argument": opponent_arg,
            "evidence": [],
            "round_number": round_number,
        })

        return proposer_arg, opponent_arg

    def _judge_debate(
        self,
        session_id: str,
        topic: str,
        arguments: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Final judgment after all rounds. Returns parsed verdict dict."""
        # Build full argument summary for the judge
        rounds_text = []
        for h in arguments:
            rounds_text.append(
                f"--- Round {h['round']} ---\n"
                f"PROPOSER (defends finding):\n{h['proposer']}\n\n"
                f"OPPONENT (challenges finding):\n{h['opponent']}"
            )

        judge_prompt = (
            f"RESEARCH FINDING:\n{topic}\n\n"
            f"DEBATE TRANSCRIPT:\n\n" + "\n\n".join(rounds_text)
        )

        log.info("Debate %s: judge evaluating %d rounds", session_id, len(arguments))
        judge_resp = self._inference.chat(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            max_tokens=1000,
        )

        verdict = _parse_judge_verdict(judge_resp.content)

        # Store judge argument
        self._db.insert_debate_argument({
            "id": str(ULID()),
            "session_id": session_id,
            "role": "judge",
            "model": self._inference.get_default_model(),
            "argument": judge_resp.content,
            "evidence": verdict.get("key_issues", []),
            "round_number": 0,
        })

        return verdict

    def get_debate_result(self, session_id: str) -> dict[str, Any]:
        """Get full debate session with all arguments and verdict."""
        session = self._db.get_debate_session(session_id)
        if not session:
            return {"error": f"Debate session {session_id} not found"}

        arguments = self._db.get_debate_arguments(session_id)

        return {
            "session": session,
            "arguments": arguments,
            "verdict": session.get("verdict"),
            "confidence": session.get("verdict_confidence"),
            "reasoning": session.get("verdict_reasoning"),
        }


def _parse_judge_verdict(text: str) -> dict[str, Any]:
    """Parse JSON verdict from judge response, with fallback."""
    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block (allow nested braces)
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback
    log.warning("Could not parse judge verdict as JSON, using defaults")
    return {
        "verdict": "inconclusive",
        "confidence": 0.5,
        "reasoning": text[:500],
        "key_issues": [],
        "recommendations": [],
    }
