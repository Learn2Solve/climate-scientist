"""Novelty checking via adversarial multi-model reasoning.

Three inference calls per check:
1. Advocate (model A, temp 0.7): argues the idea IS novel
2. Critic (model B, temp 0.3): argues the idea is NOT novel, finds prior art
3. Judge (primary model, temp 0.0): synthesizes verdict
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.novelty")


@dataclass
class NoveltyVerdict:
    id: str
    is_novel: bool
    novelty_score: float
    advocate_argument: str
    critic_argument: str
    prior_art: list[dict[str, Any]]
    recommendation: str
    models: dict[str, str]


ADVOCATE_SYSTEM = """You are a scientific innovation advocate. Your role is to argue
that a proposed research idea is NOVEL and worth pursuing.

Analyze the idea and context, then argue FOR its novelty:
- What gaps in existing knowledge does it fill?
- What combinations of existing approaches are untested?
- What new insights could it yield?
- How does it differ from existing work?

Be specific and cite concrete reasons. Do not fabricate prior work."""

CRITIC_SYSTEM = """You are a scientific novelty critic. Your role is to argue that a
proposed research idea is NOT novel, finding prior art and precedent.

Given the idea, context, and the advocate's arguments, argue AGAINST novelty:
- What existing papers or studies have already explored this?
- What standard approaches already cover this ground?
- What incremental improvements does this represent rather than true novelty?
- Be specific about prior art. Name fields, methods, or common approaches.

Do NOT fabricate specific paper titles or DOIs. Instead, describe the type of
prior work that exists (e.g., "numerous studies in the 2010s applied gradient
boosting to tropical cyclone intensity prediction")."""

JUDGE_SYSTEM = """You are a scientific novelty judge. Given an idea, an advocate's
argument FOR novelty, and a critic's argument AGAINST novelty, provide a balanced
assessment.

You MUST respond with ONLY a JSON object (no markdown, no extra text):
{
    "is_novel": true/false,
    "novelty_score": 0.0-1.0,
    "prior_art": [{"description": "...", "relevance": "high/medium/low"}],
    "recommendation": "proceed/refine/abandon",
    "reasoning": "Brief synthesis of both arguments"
}

Scoring guide:
- 0.0-0.2: Well-trodden ground, many prior studies
- 0.3-0.5: Incremental novelty, minor variations on known approaches
- 0.6-0.7: Moderate novelty, interesting combination or application
- 0.8-1.0: Highly novel, genuinely new approach or insight"""


class NoveltyChecker:
    """Adversarial novelty verification using multiple inference calls."""

    def __init__(
        self, inference: Any, db: Any, config: Any, literature: Any = None
    ) -> None:
        self._inference = inference
        self._db = db
        self._config = config
        self._literature = literature

    def check(
        self,
        idea: str,
        idea_type: str = "hypothesis",
        context: str = "",
        linked_hypothesis_id: str | None = None,
    ) -> NoveltyVerdict:
        """Run adversarial novelty check. Returns NoveltyVerdict."""
        check_id = str(ULID())

        # Gather additional context from knowledge base
        kb_context = self._gather_context(idea)
        full_context = context
        if kb_context:
            full_context = f"{context}\n\nRelated knowledge from database:\n{kb_context}"

        # Select models
        advocate_model, critic_model = self._select_models()
        models_used = {
            "advocate": advocate_model,
            "critic": critic_model,
            "judge": self._inference.get_default_model(),
        }

        idea_prompt = f"IDEA ({idea_type}): {idea}"
        if full_context:
            idea_prompt += f"\n\nCONTEXT: {full_context}"

        # Step 1: Advocate argues FOR novelty
        log.info("Novelty check: advocate (%s)", advocate_model)
        advocate_resp = self._inference.chat(
            messages=[
                {"role": "system", "content": ADVOCATE_SYSTEM},
                {"role": "user", "content": idea_prompt},
            ],
            model=advocate_model,
            temperature=0.7,
            max_tokens=1500,
        )
        advocate_argument = advocate_resp.content

        # Step 2: Critic argues AGAINST novelty
        log.info("Novelty check: critic (%s)", critic_model)
        critic_prompt = (
            f"{idea_prompt}\n\n"
            f"ADVOCATE'S ARGUMENT FOR NOVELTY:\n{advocate_argument}"
        )
        critic_resp = self._inference.chat(
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": critic_prompt},
            ],
            model=critic_model,
            temperature=0.3,
            max_tokens=1500,
        )
        critic_argument = critic_resp.content

        # Step 3: Judge synthesizes verdict
        log.info("Novelty check: judge")
        judge_prompt = (
            f"{idea_prompt}\n\n"
            f"ADVOCATE (argues novel):\n{advocate_argument}\n\n"
            f"CRITIC (argues not novel):\n{critic_argument}"
        )
        judge_resp = self._inference.chat(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            max_tokens=1000,
        )

        # Parse judge verdict
        verdict = _parse_judge_verdict(judge_resp.content)

        result = NoveltyVerdict(
            id=check_id,
            is_novel=verdict.get("is_novel", False),
            novelty_score=verdict.get("novelty_score", 0.5),
            advocate_argument=advocate_argument,
            critic_argument=critic_argument,
            prior_art=verdict.get("prior_art", []),
            recommendation=verdict.get("recommendation", "refine"),
            models=models_used,
        )

        # Store in DB
        self._db.insert_novelty_check({
            "id": check_id,
            "idea_text": idea,
            "idea_type": idea_type,
            "is_novel": result.is_novel,
            "novelty_score": result.novelty_score,
            "advocate_argument": result.advocate_argument,
            "critic_argument": result.critic_argument,
            "prior_art": result.prior_art,
            "recommendation": result.recommendation,
            "models": result.models,
            "linked_hypothesis_id": linked_hypothesis_id,
        })

        log.info(
            "Novelty check complete: score=%.2f novel=%s recommend=%s",
            result.novelty_score,
            result.is_novel,
            result.recommendation,
        )
        return result

    def _gather_context(self, idea: str) -> str:
        """Search knowledge base for related work."""
        parts: list[str] = []

        # Search knowledge
        try:
            entries = self._db.search_knowledge_fts(idea, limit=5)
            if entries:
                for e in entries:
                    parts.append(
                        f"- [{e.get('entry_type', '?')}] {e.get('content', '')[:200]}"
                    )
        except Exception:
            pass

        # Search papers
        try:
            papers = self._db.search_papers_fts(idea, limit=5)
            if papers:
                for p in papers:
                    parts.append(
                        f"- [paper] {p.get('title', '')} ({p.get('year', '?')}): "
                        f"{p.get('abstract', '')[:150]}"
                    )
        except Exception:
            pass

        # External literature search (arXiv + Semantic Scholar)
        if self._literature is not None:
            try:
                ext_context = self._literature.gather_literature_context(idea)
                if ext_context:
                    parts.append("\nExternal literature (real-time search):")
                    parts.append(ext_context)
            except Exception as exc:
                log.debug("External literature search failed: %s", exc)

        return "\n".join(parts)

    def _select_models(self) -> tuple[str, str]:
        """Select advocate and critic models from config."""
        advocate = getattr(self._config, "novelty_advocate_model", "")
        critic = getattr(self._config, "novelty_critic_model", "")

        # If CLIProxy is configured, all models route through it
        if self._inference.using_proxy:
            return (
                advocate or "claude-opus-4-6",
                critic or "codex-5.3",
            )

        # Without proxy, fall back to same model with different prompts
        default = self._inference.get_default_model()
        return (
            advocate or default,
            critic or default,
        )


def _parse_judge_verdict(text: str) -> dict[str, Any]:
    """Parse JSON verdict from judge response."""
    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: extract what we can from text
    log.warning("Could not parse judge verdict as JSON, using defaults")
    return {
        "is_novel": True,
        "novelty_score": 0.5,
        "prior_art": [],
        "recommendation": "refine",
        "reasoning": text[:500],
    }
