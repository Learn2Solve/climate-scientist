"""Research idea generation via 2-stage LLM pipeline.

Stage 1 - Brainstorm (temp 0.8): creative generation of candidate ideas.
Stage 2 - Evaluate (temp 0.0): structured scoring of each candidate.

Costs 2 inference calls per invocation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.idea_generator")


BRAINSTORM_SYSTEM = (
    "You are a creative climate scientist. Given a research topic, existing knowledge, "
    "and recent literature, propose novel research ideas that fill gaps, resolve "
    "contradictions, or combine underexplored approaches. Focus on ideas that are "
    "testable and distinct from existing hypotheses.\n\n"
    "For each idea, provide:\n"
    "- A concise title\n"
    "- A 2-3 sentence description of the idea\n"
    "- Why it is novel (what gap it fills)\n\n"
    "Number your ideas clearly (1., 2., etc.)."
)

EVALUATE_SYSTEM = (
    "You are a research evaluator. Given a research topic and a set of candidate ideas, "
    "score each idea on three dimensions:\n"
    "- novelty (0-1): How new and unexplored is this idea?\n"
    "- feasibility (0-1): How practical is it to investigate with available data/methods?\n"
    "- relevance (0-1): How relevant is it to the research topic?\n\n"
    "You MUST respond with ONLY a JSON array (no markdown, no extra text). Each element:\n"
    '{"title": "...", "description": "...", "novelty_score": 0.0, '
    '"feasibility_score": 0.0, "relevance_score": 0.0, "rationale": "..."}\n\n'
    "Scoring guide:\n"
    "- 0.0-0.2: Very low\n"
    "- 0.3-0.5: Below average\n"
    "- 0.6-0.7: Moderate\n"
    "- 0.8-1.0: High"
)


class IdeaGenerator:
    """Generate and evaluate research ideas via a 2-stage LLM pipeline."""

    def __init__(
        self,
        inference: Any,
        db: Any,
        config: Any,
        literature: Any = None,
    ) -> None:
        self._inference = inference
        self._db = db
        self._config = config
        self._literature = literature

    def generate(
        self,
        topic: str,
        max_ideas: int = 5,
        constraint: str = "",
        use_literature: bool = True,
    ) -> dict[str, Any]:
        """Generate ranked research ideas for a topic.

        Returns dict with ok, ideas (ranked list), and metadata.
        """
        gen_id = str(ULID())
        model = getattr(self._config, "idea_generator_model", None) or self._inference.get_default_model()

        # 1. Gather context
        context = self._gather_context(topic, use_literature)

        # 2. Brainstorm (high temperature)
        log.info("Idea generation: brainstorm stage (model=%s)", model)
        raw_ideas = self._brainstorm(topic, context, constraint, max_ideas, model)

        # 3. Evaluate (low temperature)
        log.info("Idea generation: evaluate stage")
        ideas = self._evaluate(topic, raw_ideas, context, model)

        # 4. Rank by composite score (avg of 3 dimensions)
        for idea in ideas:
            idea["composite_score"] = round(
                (idea.get("novelty_score", 0) + idea.get("feasibility_score", 0) + idea.get("relevance_score", 0)) / 3,
                3,
            )
        ideas.sort(key=lambda x: x["composite_score"], reverse=True)

        # Trim to max_ideas
        ideas = ideas[:max_ideas]

        # 5. Store top ideas in knowledge base
        for idea in ideas:
            try:
                self._db.insert_knowledge(
                    topic=topic,
                    content=f"[Generated Idea] {idea['title']}: {idea['description']}",
                    source=f"idea_generator:{gen_id}",
                    confidence=idea["composite_score"],
                    entry_type="observation",
                )
            except Exception:
                log.debug("Failed to store idea in knowledge base", exc_info=True)

        log.info(
            "Idea generation complete: %d ideas, top score=%.2f",
            len(ideas),
            ideas[0]["composite_score"] if ideas else 0,
        )

        return {
            "ok": True,
            "generation_id": gen_id,
            "topic": topic,
            "ideas": ideas,
            "model": model,
            "context_length": len(context),
        }

    def _gather_context(self, topic: str, use_literature: bool) -> str:
        """Gather context from DB knowledge, hypotheses, and literature."""
        parts: list[str] = []

        # Active hypotheses
        try:
            hypotheses = self._db.get_hypotheses(status="active", limit=10)
            if hypotheses:
                parts.append("ACTIVE HYPOTHESES:")
                for h in hypotheses:
                    parts.append(f"- {h.get('statement', h.get('content', ''))[:200]}")
        except Exception:
            pass

        # Recent findings/conclusions
        try:
            entries = self._db.search_knowledge_fts(topic, limit=10)
            if entries:
                parts.append("\nRECENT KNOWLEDGE:")
                for e in entries:
                    entry_type = e.get("entry_type", "?")
                    content = e.get("content", "")[:200]
                    parts.append(f"- [{entry_type}] {content}")
        except Exception:
            pass

        # Literature (local FTS)
        try:
            papers = self._db.search_papers_fts(topic, limit=10)
            if papers:
                parts.append("\nRELATED PAPERS (local DB):")
                for p in papers:
                    title = p.get("title", "untitled")[:150]
                    year = p.get("year", "?")
                    parts.append(f"- [{year}] {title}")
        except Exception:
            pass

        # External literature search
        if use_literature and self._literature:
            try:
                result = self._literature.search_all(topic, max_per_source=3, sources=["arxiv", "semantic_scholar"])
                for source_key in ("arxiv", "semantic_scholar"):
                    source_papers = result.get(source_key, [])
                    if source_papers:
                        parts.append(f"\nRECENT LITERATURE ({source_key}):")
                        for p in source_papers[:3]:
                            title = p.get("title", "untitled")[:150]
                            year = p.get("year", "?")
                            parts.append(f"- [{year}] {title}")
            except Exception:
                log.debug("External literature search failed", exc_info=True)

        return "\n".join(parts) if parts else ""

    def _brainstorm(
        self, topic: str, context: str, constraint: str, max_ideas: int, model: str
    ) -> str:
        """Stage 1: creative brainstorming at high temperature."""
        user_prompt = f"RESEARCH TOPIC: {topic}\n\nGenerate {max_ideas} novel research ideas."
        if context:
            user_prompt += f"\n\nEXISTING KNOWLEDGE AND LITERATURE:\n{context}"
        if constraint:
            user_prompt += f"\n\nCONSTRAINT: {constraint}"

        resp = self._inference.chat(
            messages=[
                {"role": "system", "content": BRAINSTORM_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.8,
            max_tokens=3000,
        )
        return resp.content

    def _evaluate(
        self, topic: str, raw_ideas: str, context: str, model: str
    ) -> list[dict[str, Any]]:
        """Stage 2: structured evaluation at low temperature."""
        user_prompt = (
            f"RESEARCH TOPIC: {topic}\n\n"
            f"CANDIDATE IDEAS:\n{raw_ideas}"
        )
        if context:
            user_prompt += f"\n\nCONTEXT (existing knowledge):\n{context[:2000]}"

        resp = self._inference.chat(
            messages=[
                {"role": "system", "content": EVALUATE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=2000,
        )

        return _parse_evaluation(resp.content)


def _parse_evaluation(text: str) -> list[dict[str, Any]]:
    """Parse the evaluation JSON array from LLM output."""
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            # Validate and normalize each entry
            result = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                result.append({
                    "title": str(item.get("title", "Untitled")),
                    "description": str(item.get("description", "")),
                    "novelty_score": float(item.get("novelty_score", 0.5)),
                    "feasibility_score": float(item.get("feasibility_score", 0.5)),
                    "relevance_score": float(item.get("relevance_score", 0.5)),
                    "rationale": str(item.get("rationale", "")),
                })
            return result
    except (json.JSONDecodeError, ValueError):
        log.warning("Failed to parse evaluation JSON, returning empty list")

    return []
