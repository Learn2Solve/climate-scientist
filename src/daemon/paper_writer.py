"""Paper writer - generates and iteratively improves research papers.

Gathers structured knowledge (hypotheses, experiments, findings, conclusions)
from the database and uses LLM inference to draft, review, and revise
scientific papers in markdown format.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ulid import ULID

from .inference import InferenceClient

log = logging.getLogger("daemon.paper_writer")

PAPER_TEMPLATE = '''# {title}

## Abstract
{abstract}

## 1. Introduction
{introduction}

## 2. Related Work
{related_work}

## 3. Methodology
{methodology}

## 4. Experiments
{experiments}

## 5. Results
{results}

## 6. Discussion
{discussion}

## 7. Conclusion
{conclusion}

## References
{references}
'''

DRAFT_SYSTEM_PROMPT = """You are a scientific paper writer specializing in climate science.
Given research data (hypotheses, experiments, findings, conclusions, citations),
write a well-structured research paper section.

Requirements:
- Use precise scientific language
- Cite papers using [Author, Year] format
- Include quantitative results where available
- Discuss limitations honestly
- Compare with prior work"""

REVIEW_SYSTEM_PROMPT = """You are a rigorous peer reviewer for a climate science journal.
Review the paper and provide:
1. Overall score (1-10)
2. Category scores: {"novelty": 1-10, "soundness": 1-10, "clarity": 1-10, "significance": 1-10}
3. Strengths (specific, 3-5 points)
4. Weaknesses (specific, 3-5 points)
5. Suggestions for improvement
6. Decision: "accept", "revise", or "reject"

You MUST respond with ONLY a JSON object."""

REVISION_SYSTEM_PROMPT = """You are a scientific paper writer revising a paper based on peer review.
You will receive the original paper content and reviewer feedback.
Revise the paper to address the weaknesses and incorporate suggestions.
Maintain the same section structure. Output ONLY the revised paper content."""

# Section-specific guidance for the LLM
SECTION_GUIDANCE: dict[str, str] = {
    "abstract": (
        "Write a concise abstract (150-250 words) summarizing the research: "
        "motivation, methods, key results, and conclusions."
    ),
    "introduction": (
        "Write the introduction section. Include: background on the topic, "
        "motivation for this research, specific research questions or hypotheses, "
        "and a brief outline of the paper structure."
    ),
    "related_work": (
        "Write the related work section. Synthesize the cited papers, "
        "highlighting how they relate to the current research and identifying gaps."
    ),
    "methodology": (
        "Write the methodology section. Describe the experimental approach, "
        "data sources, analysis methods, and any models used."
    ),
    "experiments": (
        "Write the experiments section. Describe the experimental setup, "
        "parameters, datasets used, and evaluation metrics."
    ),
    "results": (
        "Write the results section. Present quantitative findings with specific "
        "numbers, tables where appropriate, and clear descriptions of outcomes."
    ),
    "discussion": (
        "Write the discussion section. Interpret the results, compare with prior work, "
        "discuss limitations, and suggest future directions."
    ),
    "conclusion": (
        "Write the conclusion section. Summarize key contributions, "
        "restate main findings, and discuss broader implications."
    ),
    "references": (
        "Format the references list. Use [Author, Year] citation format. "
        "Include DOI where available."
    ),
}


class PaperWriter:
    """Generates and iteratively improves research papers from the knowledge base."""

    def __init__(self, inference: InferenceClient, db: Any, config: Any) -> None:
        self._inference = inference
        self._db = db
        self._config = config
        self._topic: str = getattr(config, "topic", "climate science")
        self._paper_format: str = getattr(config, "paper_format", "markdown")
        self._reviewer_model: str = getattr(config, "paper_reviewer_model", "claude-opus-4-6")

    def draft_paper(
        self,
        title: str | None = None,
        linked_plan_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a full paper draft from the knowledge base.

        Args:
            title: Paper title; auto-generated if not provided.
            linked_plan_id: Optional research plan this paper is linked to.

        Returns:
            Dict with draft_id, title, version, and a content preview.
        """
        research_data = self._gather_research_data()

        # Generate title if not provided
        if not title:
            title = self._generate_title(research_data)

        log.info("Drafting paper: %s", title)

        # Generate each section
        sections: dict[str, str] = {}
        section_order = [
            "abstract", "introduction", "related_work", "methodology",
            "experiments", "results", "discussion", "conclusion", "references",
        ]

        for section_name in section_order:
            log.info("Generating section: %s", section_name)
            sections[section_name] = self._generate_section(
                section_name, research_data, existing_sections=sections,
            )

        # Assemble full paper
        content = PAPER_TEMPLATE.format(title=title, **sections)

        # Store in DB
        draft_id = str(ULID())
        self._db.insert_paper_draft({
            "id": draft_id,
            "title": title,
            "abstract": sections["abstract"],
            "content": content,
            "format": self._paper_format,
            "version": 1,
            "status": "draft",
            "linked_plan_id": linked_plan_id,
        })

        log.info("Paper draft stored: %s (id=%s)", title, draft_id)

        return {
            "draft_id": draft_id,
            "title": title,
            "version": 1,
            "preview": content[:500],
        }

    def review_paper(self, draft_id: str) -> dict[str, Any]:
        """Run a simulated peer review on a paper draft.

        Args:
            draft_id: ID of the paper draft to review.

        Returns:
            Dict with review_id, overall_score, decision, strengths,
            weaknesses, and suggestions.
        """
        draft = self._db.get_paper_draft(draft_id)
        if not draft:
            raise ValueError(f"Paper draft not found: {draft_id}")

        log.info("Reviewing paper: %s", draft["title"])

        messages = [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Review this climate science paper:\n\n"
                    f"Title: {draft['title']}\n\n"
                    f"{draft['content']}"
                ),
            },
        ]

        response = self._inference.chat(
            messages=messages,
            model=self._reviewer_model,
            temperature=0.4,
            max_tokens=2000,
        )

        review_data = self._parse_review(response.content)

        review_id = str(ULID())
        self._db.insert_paper_review({
            "id": review_id,
            "draft_id": draft_id,
            "reviewer_model": self._reviewer_model,
            "overall_score": review_data["overall_score"],
            "scores": review_data["scores"],
            "strengths": review_data["strengths"],
            "weaknesses": review_data["weaknesses"],
            "suggestions": review_data["suggestions"],
            "decision": review_data["decision"],
        })

        log.info(
            "Review complete: score=%.1f, decision=%s",
            review_data["overall_score"], review_data["decision"],
        )

        return {
            "review_id": review_id,
            "overall_score": review_data["overall_score"],
            "decision": review_data["decision"],
            "strengths": review_data["strengths"],
            "weaknesses": review_data["weaknesses"],
            "suggestions": review_data["suggestions"],
        }

    def revise_paper(
        self,
        draft_id: str,
        review_id: str | None = None,
    ) -> dict[str, Any]:
        """Revise a paper based on review feedback.

        Args:
            draft_id: ID of the paper draft to revise.
            review_id: Specific review to address; uses latest if not provided.

        Returns:
            Dict with draft_id, new version number, and changes summary.
        """
        draft = self._db.get_paper_draft(draft_id)
        if not draft:
            raise ValueError(f"Paper draft not found: {draft_id}")

        # Get review feedback
        if review_id:
            reviews = self._db.get_reviews_for_draft(draft_id)
            review = next((r for r in reviews if r["id"] == review_id), None)
            if not review:
                raise ValueError(f"Review not found: {review_id}")
        else:
            reviews = self._db.get_reviews_for_draft(draft_id)
            if not reviews:
                raise ValueError(f"No reviews found for draft: {draft_id}")
            review = reviews[0]  # Most recent (ordered DESC)

        log.info(
            "Revising paper '%s' based on review (score=%.1f, decision=%s)",
            draft["title"], review["overall_score"], review["decision"],
        )

        feedback = (
            f"Reviewer decision: {review['decision']}\n"
            f"Overall score: {review['overall_score']}/10\n\n"
            f"Strengths:\n{review['strengths']}\n\n"
            f"Weaknesses:\n{review['weaknesses']}\n\n"
            f"Suggestions:\n{review['suggestions']}"
        )

        messages = [
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original paper:\n\n{draft['content']}\n\n"
                    f"---\n\nReviewer feedback:\n\n{feedback}\n\n"
                    f"Please revise the paper to address the feedback."
                ),
            },
        ]

        response = self._inference.chat(
            messages=messages,
            model=self._reviewer_model,
            temperature=0.3,
            max_tokens=4000,
        )

        revised_content = response.content
        new_version = draft["version"] + 1

        # Generate a brief changes summary
        changes_summary = self._summarize_changes(draft["content"], revised_content)

        # Update the draft in DB
        self._db.update_paper_draft(
            draft_id,
            content=revised_content,
            version=new_version,
            status="revised",
        )

        log.info("Paper revised to version %d", new_version)

        return {
            "draft_id": draft_id,
            "version": new_version,
            "changes_summary": changes_summary,
        }

    # -- Data gathering ----------------------------------------------------

    def _gather_research_data(self) -> dict[str, Any]:
        """Pull structured research data from the knowledge base.

        Returns a dict organized by category for paper generation.
        """
        data: dict[str, Any] = {
            "topic": self._topic,
            "hypotheses": [],
            "experiments": [],
            "findings": [],
            "conclusions": [],
            "observations": [],
            "papers": [],
            "executions": [],
        }

        # Knowledge entries by type
        for entry_type in ("hypothesis", "experiment", "finding", "conclusion", "observation"):
            entries = self._db.get_knowledge_by_type(entry_type, limit=100)
            key = entry_type if entry_type != "hypothesis" else "hypotheses"
            if entry_type == "hypothesis":
                key = "hypotheses"
            elif entry_type == "experiment":
                key = "experiments"
            elif entry_type == "finding":
                key = "findings"
            elif entry_type == "conclusion":
                key = "conclusions"
            else:
                key = "observations"
            data[key] = entries

        # Cited papers
        all_knowledge = self._db.get_recent_knowledge(limit=200)
        cited_paper_ids: set[str] = set()
        for k in all_knowledge:
            cites = self._db.get_citations_for_knowledge(k["id"])
            for c in cites:
                cited_paper_ids.add(c["paper_id"])

        papers = []
        for pid in cited_paper_ids:
            paper = self._db.get_paper_by_id(pid)
            if paper:
                papers.append(paper)
        data["papers"] = papers

        # Sandbox execution results
        data["executions"] = self._db.get_recent_sandbox_executions(limit=20)

        log.info(
            "Gathered research data: %d hypotheses, %d experiments, "
            "%d findings, %d conclusions, %d papers, %d executions",
            len(data["hypotheses"]), len(data["experiments"]),
            len(data["findings"]), len(data["conclusions"]),
            len(data["papers"]), len(data["executions"]),
        )

        return data

    def _generate_section(
        self,
        section_name: str,
        research_data: dict[str, Any],
        existing_sections: dict[str, str] | None = None,
    ) -> str:
        """Generate a single paper section using inference.

        Args:
            section_name: Name of the section (e.g. "abstract", "introduction").
            research_data: Full research data dict from _gather_research_data.
            existing_sections: Already-generated sections for coherence.

        Returns:
            The generated section text.
        """
        guidance = SECTION_GUIDANCE.get(section_name, f"Write the {section_name} section.")

        # Build context with relevant data for this section
        context_parts: list[str] = [
            f"Research topic: {research_data['topic']}",
        ]

        # Section-specific data selection
        if section_name == "abstract":
            context_parts.append(self._format_all_data_brief(research_data))
        elif section_name == "introduction":
            context_parts.append(self._format_hypotheses(research_data))
            context_parts.append(self._format_observations(research_data))
        elif section_name == "related_work":
            context_parts.append(self._format_papers(research_data))
        elif section_name == "methodology":
            context_parts.append(self._format_experiments(research_data))
        elif section_name == "experiments":
            context_parts.append(self._format_experiments(research_data))
            context_parts.append(self._format_executions(research_data))
        elif section_name == "results":
            context_parts.append(self._format_findings(research_data))
            context_parts.append(self._format_executions(research_data))
        elif section_name == "discussion":
            context_parts.append(self._format_findings(research_data))
            context_parts.append(self._format_conclusions(research_data))
            context_parts.append(self._format_papers(research_data))
        elif section_name == "conclusion":
            context_parts.append(self._format_conclusions(research_data))
        elif section_name == "references":
            context_parts.append(self._format_papers(research_data))

        # Include already-written sections for coherence
        if existing_sections:
            coherence = "\n\n".join(
                f"[Already written - {name}]:\n{text[:300]}..."
                for name, text in existing_sections.items()
            )
            context_parts.append(f"\nPrevious sections for coherence:\n{coherence}")

        context = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Section to write: {section_name}\n\n"
                    f"Instructions: {guidance}\n\n"
                    f"Research data:\n{context}\n\n"
                    f"Write ONLY the section content (no heading)."
                ),
            },
        ]

        response = self._inference.chat(
            messages=messages,
            model=self._reviewer_model,
            temperature=0.4,
            max_tokens=2000,
        )

        return response.content.strip()

    # -- Title generation --------------------------------------------------

    def _generate_title(self, research_data: dict[str, Any]) -> str:
        """Auto-generate a paper title from the research data."""
        brief = self._format_all_data_brief(research_data)

        messages = [
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Generate a concise, informative title for a research paper "
                    f"based on this data:\n\n{brief}\n\n"
                    f"Output ONLY the title, no quotes or extra text."
                ),
            },
        ]

        response = self._inference.chat(
            messages=messages,
            model=self._reviewer_model,
            temperature=0.5,
            max_tokens=100,
        )

        return response.content.strip().strip('"').strip("'")

    # -- Review parsing ----------------------------------------------------

    def _parse_review(self, text: str) -> dict[str, Any]:
        """Parse review JSON from model response.

        Falls back to sensible defaults if parsing fails.
        """
        stripped = text.strip()

        # Handle markdown code fences
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            inner = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )
            stripped = inner.strip()

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return {
                    "overall_score": float(parsed.get("overall_score", 5)),
                    "scores": parsed.get("scores", {
                        "novelty": 5, "soundness": 5,
                        "clarity": 5, "significance": 5,
                    }),
                    "strengths": str(parsed.get("strengths", "")),
                    "weaknesses": str(parsed.get("weaknesses", "")),
                    "suggestions": str(parsed.get("suggestions", "")),
                    "decision": str(parsed.get("decision", "revise")),
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            log.warning("Failed to parse review JSON, using fallback")

        # Fallback: return the raw text as the review body
        return {
            "overall_score": 5.0,
            "scores": {"novelty": 5, "soundness": 5, "clarity": 5, "significance": 5},
            "strengths": "Review parsing failed - raw response available.",
            "weaknesses": text[:500],
            "suggestions": "Re-run review with structured output.",
            "decision": "revise",
        }

    # -- Changes summary ---------------------------------------------------

    def _summarize_changes(self, original: str, revised: str) -> str:
        """Generate a brief summary of changes between versions."""
        messages = [
            {"role": "system", "content": "Summarize the changes between two versions of a paper in 2-3 sentences."},
            {
                "role": "user",
                "content": (
                    f"Original (first 1000 chars):\n{original[:1000]}\n\n"
                    f"Revised (first 1000 chars):\n{revised[:1000]}\n\n"
                    f"Summarize the key changes."
                ),
            },
        ]

        response = self._inference.chat(
            messages=messages,
            model=self._reviewer_model,
            temperature=0.3,
            max_tokens=200,
        )

        return response.content.strip()

    # -- Data formatting helpers -------------------------------------------

    def _format_all_data_brief(self, data: dict[str, Any]) -> str:
        """Format a brief overview of all research data."""
        parts: list[str] = [f"Topic: {data['topic']}"]

        if data["hypotheses"]:
            parts.append(f"Hypotheses ({len(data['hypotheses'])}):")
            for h in data["hypotheses"][:5]:
                status = h.get("status", "active")
                parts.append(f"  [{status}] {h['content'][:150]}")

        if data["findings"]:
            parts.append(f"Key findings ({len(data['findings'])}):")
            for f in data["findings"][:5]:
                parts.append(f"  - {f['content'][:150]}")

        if data["conclusions"]:
            parts.append(f"Conclusions ({len(data['conclusions'])}):")
            for c in data["conclusions"][:5]:
                meta = json.loads(c.get("metadata_json", "{}"))
                verdict = meta.get("verdict", "?")
                parts.append(f"  [{verdict}] {c['content'][:150]}")

        return "\n".join(parts)

    def _format_hypotheses(self, data: dict[str, Any]) -> str:
        if not data["hypotheses"]:
            return "No hypotheses recorded."
        lines = ["Hypotheses:"]
        for h in data["hypotheses"]:
            status = h.get("status", "active")
            lines.append(f"  [{status}] {h['content'][:200]}")
        return "\n".join(lines)

    def _format_observations(self, data: dict[str, Any]) -> str:
        if not data["observations"]:
            return "No observations recorded."
        lines = ["Observations:"]
        for o in data["observations"][:10]:
            lines.append(f"  - {o['content'][:200]}")
        return "\n".join(lines)

    def _format_experiments(self, data: dict[str, Any]) -> str:
        if not data["experiments"]:
            return "No experiments recorded."
        lines = ["Experiments:"]
        for e in data["experiments"]:
            lines.append(f"  - {e['content'][:200]}")
            meta = json.loads(e.get("metadata_json", "{}"))
            if meta:
                lines.append(f"    Metadata: {json.dumps(meta)[:200]}")
        return "\n".join(lines)

    def _format_findings(self, data: dict[str, Any]) -> str:
        if not data["findings"]:
            return "No findings recorded."
        lines = ["Findings:"]
        for f in data["findings"]:
            lines.append(f"  - {f['content'][:200]}")
            meta = json.loads(f.get("metadata_json", "{}"))
            if meta:
                lines.append(f"    Metrics: {json.dumps(meta)[:200]}")
        return "\n".join(lines)

    def _format_conclusions(self, data: dict[str, Any]) -> str:
        if not data["conclusions"]:
            return "No conclusions recorded."
        lines = ["Conclusions:"]
        for c in data["conclusions"]:
            meta = json.loads(c.get("metadata_json", "{}"))
            verdict = meta.get("verdict", "?")
            lines.append(f"  [{verdict}] {c['content'][:200]}")
        return "\n".join(lines)

    def _format_papers(self, data: dict[str, Any]) -> str:
        if not data["papers"]:
            return "No cited papers."
        lines = ["Cited papers:"]
        for p in data["papers"]:
            authors = p.get("authors", "")[:60]
            year = p.get("year", "?")
            title = p.get("title", "Unknown")
            doi = p.get("doi", "")
            line = f"  - {authors} ({year}). {title}"
            if doi:
                line += f" DOI: {doi}"
            lines.append(line)
        return "\n".join(lines)

    def _format_executions(self, data: dict[str, Any]) -> str:
        if not data["executions"]:
            return "No sandbox executions."
        lines = ["Sandbox executions:"]
        for ex in data["executions"][:10]:
            desc = (ex.get("description") or "")[:100]
            metrics = ex.get("metrics") or {}
            metrics_str = ", ".join(
                f"{k}={v}" for k, v in list(metrics.items())[:5]
            ) if metrics else "none"
            lines.append(
                f"  - {desc} (rc={ex['returncode']}, {ex['duration_s']:.1f}s, "
                f"metrics: {metrics_str})"
            )
        return "\n".join(lines)
