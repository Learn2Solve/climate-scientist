"""Persistent knowledge accumulation for the climate research agent.

Supports structured knowledge: hypothesis -> experiment -> finding -> conclusion chains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ulid import ULID

if TYPE_CHECKING:
    from .database import DaemonDatabase


@dataclass
class KnowledgeEntry:
    id: str
    topic: str
    content: str
    source: str
    confidence: float
    created_at: str
    entry_type: str = "observation"     # hypothesis|experiment|finding|conclusion|observation
    parent_id: str | None = None        # links to parent entry
    status: str = "active"              # active|superseded|refuted|confirmed
    metadata: dict[str, Any] = field(default_factory=dict)

    # Handle metadata_json from DB rows
    metadata_json: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.metadata_json is not None and not self.metadata:
            try:
                self.metadata = json.loads(self.metadata_json)
            except (json.JSONDecodeError, TypeError):
                self.metadata = {}


def _entry_from_row(row: dict[str, Any]) -> KnowledgeEntry:
    """Create a KnowledgeEntry from a database row dict."""
    return KnowledgeEntry(
        id=row["id"],
        topic=row["topic"],
        content=row["content"],
        source=row.get("source", ""),
        confidence=row.get("confidence", 0.5),
        created_at=row.get("created_at", ""),
        entry_type=row.get("entry_type", "observation"),
        parent_id=row.get("parent_id"),
        status=row.get("status", "active"),
        metadata_json=row.get("metadata_json", "{}"),
    )


def save_knowledge(
    db: DaemonDatabase,
    topic: str,
    content: str,
    source: str,
    confidence: float = 0.5,
    entry_type: str = "observation",
    parent_id: str | None = None,
    status: str = "active",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a knowledge entry and return its id."""
    entry_id = str(ULID())
    db.insert_knowledge(
        entry_id, topic, content, source, confidence,
        entry_type=entry_type,
        parent_id=parent_id,
        status=status,
        metadata_json=json.dumps(metadata or {}),
    )
    return entry_id


def search_knowledge(
    db: DaemonDatabase,
    query: str,
    limit: int = 10,
    entry_type: str | None = None,
) -> list[KnowledgeEntry]:
    """Search knowledge using FTS5, falling back to LIKE if FTS5 is empty."""
    # Try FTS5 first
    rows = db.search_knowledge_fts(query, limit)
    if not rows:
        # Fall back to LIKE search
        rows = db.search_knowledge(query, limit)
    if entry_type:
        rows = [r for r in rows if r.get("entry_type") == entry_type]
    return [_entry_from_row(r) for r in rows]


def get_recent_knowledge(
    db: DaemonDatabase,
    limit: int = 20,
) -> list[KnowledgeEntry]:
    """Get most recent knowledge entries ordered by created_at DESC."""
    rows = db.get_recent_knowledge(limit)
    return [_entry_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Typed save functions for structured research chains
# ---------------------------------------------------------------------------


def save_hypothesis(
    db: DaemonDatabase,
    content: str,
    source: str,
    confidence: float = 0.5,
) -> str:
    """Save a hypothesis entry and return its id."""
    return save_knowledge(
        db,
        topic="hypothesis",
        content=content,
        source=source,
        confidence=confidence,
        entry_type="hypothesis",
        status="active",
    )


def save_experiment(
    db: DaemonDatabase,
    content: str,
    source: str,
    hypothesis_id: str,
    confidence: float = 0.5,
) -> str:
    """Save an experiment linked to a hypothesis and return its id."""
    return save_knowledge(
        db,
        topic="experiment",
        content=content,
        source=source,
        confidence=confidence,
        entry_type="experiment",
        parent_id=hypothesis_id,
    )


def save_finding(
    db: DaemonDatabase,
    content: str,
    source: str,
    experiment_id: str,
    confidence: float = 0.5,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Save a finding linked to an experiment and return its id."""
    return save_knowledge(
        db,
        topic="finding",
        content=content,
        source=source,
        confidence=confidence,
        entry_type="finding",
        parent_id=experiment_id,
        metadata=metrics,
    )


def save_conclusion(
    db: DaemonDatabase,
    content: str,
    source: str,
    hypothesis_id: str,
    verdict: str,
    confidence: float = 0.5,
) -> str:
    """Save a conclusion for a hypothesis and update the hypothesis status.

    verdict: confirmed | refuted | inconclusive
    """
    # Map verdict to hypothesis status
    status_map = {
        "confirmed": "confirmed",
        "refuted": "refuted",
        "inconclusive": "active",  # stays active if inconclusive
    }
    new_status = status_map.get(verdict, "active")
    db.update_knowledge_status(hypothesis_id, new_status)

    return save_knowledge(
        db,
        topic="conclusion",
        content=content,
        source=source,
        confidence=confidence,
        entry_type="conclusion",
        parent_id=hypothesis_id,
        metadata={"verdict": verdict},
    )


def get_hypothesis_tree(
    db: DaemonDatabase,
    hypothesis_id: str,
) -> dict[str, Any]:
    """Get the full tree for a hypothesis: hypothesis + experiments + findings + conclusion."""
    hypothesis = db.get_knowledge_by_id(hypothesis_id)
    if not hypothesis:
        return {"error": f"hypothesis not found: {hypothesis_id}"}

    chain = db.get_knowledge_chain(hypothesis_id)

    # Organize by type
    experiments = [e for e in chain if e.get("entry_type") == "experiment"]
    findings = [e for e in chain if e.get("entry_type") == "finding"]
    conclusions = [e for e in chain if e.get("entry_type") == "conclusion"]

    return {
        "hypothesis": hypothesis,
        "experiments": experiments,
        "findings": findings,
        "conclusions": conclusions,
        "status": hypothesis.get("status", "active"),
    }


# ---------------------------------------------------------------------------
# Formatting for system prompt
# ---------------------------------------------------------------------------


def format_knowledge_summary(
    entries: list[KnowledgeEntry],
    max_chars: int = 4000,
) -> str:
    """Format knowledge entries for inclusion in a system prompt.

    Each entry is rendered as: [{confidence:.0%}] ({source}) {topic}: {content}
    The total output is truncated to max_chars.
    """
    if not entries:
        return "No knowledge accumulated yet."

    lines: list[str] = []
    total = 0
    for entry in entries:
        line = (
            f"[{entry.confidence:.0%}] ({entry.source}) "
            f"{entry.topic}: {entry.content}"
        )
        if total + len(line) + 1 > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                lines.append(line[:remaining])
            break
        lines.append(line)
        total += len(line) + 1  # +1 for the newline

    return "\n".join(lines)


def format_world_model(
    db: DaemonDatabase,
    max_chars: int = 6000,
) -> str:
    """Build a structured world model summary for the system prompt.

    Groups entries by type:
    1. Active hypotheses with their experiment/finding/conclusion chains
    2. Recent confirmed conclusions
    3. Recent observations
    """
    sections: list[str] = []
    total = 0

    def _add(text: str) -> bool:
        nonlocal total
        if total + len(text) + 1 > max_chars:
            return False
        sections.append(text)
        total += len(text) + 1
        return True

    # 1. Active hypotheses with chains
    active_hypotheses = db.get_active_hypotheses()
    if active_hypotheses:
        _add("== Active Hypotheses ==")
        for h in active_hypotheses[:10]:
            tree = get_hypothesis_tree(db, h["id"])
            n_exp = len(tree.get("experiments", []))
            n_find = len(tree.get("findings", []))
            conclusions = tree.get("conclusions", [])
            status_str = h.get("status", "active")

            # Check for citations on this hypothesis
            citations = db.get_citations_for_knowledge(h["id"])
            cite_suffix = ""
            if citations:
                cite_titles = [
                    c.get("title", "?")[:40] for c in citations[:3]
                ]
                cite_suffix = f" (cites: {', '.join(cite_titles)})"

            line = f"[H] {h['content'][:200]}{cite_suffix}"
            if not _add(line):
                break
            chain_line = f"    -> {n_exp} experiment(s) -> {n_find} finding(s)"
            if conclusions:
                verdict = json.loads(
                    conclusions[0].get("metadata_json", "{}")
                ).get("verdict", "?")
                chain_line += f" -> CONCLUSION: {verdict}"
            elif status_str == "active":
                chain_line += " -> pending conclusion"
            if not _add(chain_line):
                break

    # 2. Confirmed/refuted conclusions
    concluded = db.get_knowledge_by_type("conclusion", limit=10)
    if concluded:
        _add("")
        _add("== Recent Conclusions ==")
        for c in concluded:
            meta = json.loads(c.get("metadata_json", "{}"))
            verdict = meta.get("verdict", "?")
            tag = "CONFIRMED" if verdict == "confirmed" else (
                "REFUTED" if verdict == "refuted" else "INCONCLUSIVE"
            )
            line = f"[{tag}] {c['content'][:200]}"
            if not _add(line):
                break

    # 3. Recent observations/findings
    recent = db.get_recent_knowledge(limit=15)
    observations = [
        r for r in recent
        if r.get("entry_type", "observation") == "observation"
    ]
    if observations:
        _add("")
        _add("== Recent Observations ==")
        for o in observations[:8]:
            line = f"[{o.get('confidence', 0.5):.0%}] {o['topic']}: {o['content'][:150]}"
            if not _add(line):
                break

    if not sections:
        return "No knowledge accumulated yet."

    return "\n".join(sections)
