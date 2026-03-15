"""Report generation - produces research reports from knowledge base.

Extends src/agent/paper.py patterns (marker-based replacement, table rendering)
to generate full research reports from accumulated knowledge.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("daemon.report_generator")

# Markers for report sections
SECTION_MARKERS = {
    "summary": ("<!-- REPORT_SUMMARY_START -->", "<!-- REPORT_SUMMARY_END -->"),
    "hypotheses": ("<!-- REPORT_HYPOTHESES_START -->", "<!-- REPORT_HYPOTHESES_END -->"),
    "experiments": ("<!-- REPORT_EXPERIMENTS_START -->", "<!-- REPORT_EXPERIMENTS_END -->"),
    "findings": ("<!-- REPORT_FINDINGS_START -->", "<!-- REPORT_FINDINGS_END -->"),
    "conclusions": ("<!-- REPORT_CONCLUSIONS_START -->", "<!-- REPORT_CONCLUSIONS_END -->"),
    "references": ("<!-- REPORT_REFERENCES_START -->", "<!-- REPORT_REFERENCES_END -->"),
}


class ReportGenerator:
    """Generate research reports from the knowledge base."""

    def __init__(self, db: Any, config: Any) -> None:
        self._db = db
        self._config = config
        self._repo_root = Path(getattr(config, "repo_root", Path.cwd()))

    def generate_research_report(
        self,
        plan_id: str | None = None,
        title: str = "Research Report",
        sections: list[str] | None = None,
    ) -> str:
        """Generate a full research report from knowledge base.

        If plan_id is given, scopes the report to that plan's linked entries.
        If sections is given, only generates those sections.
        """
        all_sections = sections or [
            "summary", "hypotheses", "experiments", "findings",
            "conclusions", "references",
        ]

        parts: list[str] = []
        parts.append(f"# {title}")
        parts.append("")
        parts.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
        parts.append("")

        if "summary" in all_sections:
            parts.append("## Summary")
            parts.append("")
            parts.append(self._render_summary())
            parts.append("")

        if "hypotheses" in all_sections:
            parts.append("## Hypotheses")
            parts.append("")
            parts.append(self._render_hypotheses())
            parts.append("")

        if "experiments" in all_sections:
            parts.append("## Experiments")
            parts.append("")
            parts.append(self._render_experiments())
            parts.append("")

        if "findings" in all_sections:
            parts.append("## Findings")
            parts.append("")
            parts.append(self._render_findings())
            parts.append("")

        if "conclusions" in all_sections:
            parts.append("## Conclusions")
            parts.append("")
            parts.append(self._render_conclusions())
            parts.append("")

        if "references" in all_sections:
            parts.append("## References")
            parts.append("")
            parts.append(self._render_references())
            parts.append("")

        return "\n".join(parts)

    def generate_experiment_summary(
        self, execution_ids: list[str] | None = None
    ) -> str:
        """Render metrics tables from sandbox executions."""
        if execution_ids:
            executions = [
                self._db.get_sandbox_execution(eid)
                for eid in execution_ids
            ]
            executions = [e for e in executions if e]
        else:
            executions = self._db.get_recent_sandbox_executions(limit=20)

        if not executions:
            return "_No sandbox executions found._"

        lines: list[str] = []
        lines.append("| # | Description | Backend | Return | Duration | Metrics |")
        lines.append("| --- | --- | --- | --- | --- | --- |")

        for i, ex in enumerate(executions, 1):
            desc = (ex.get("description") or "")[:40]
            metrics = ex.get("metrics") or {}
            metrics_str = ", ".join(
                f"{k}={v}" for k, v in list(metrics.items())[:5]
            ) if metrics else "-"
            lines.append(
                f"| {i} | {desc} | {ex['backend']} | {ex['returncode']} | "
                f"{ex['duration_s']:.1f}s | {metrics_str} |"
            )

        return "\n".join(lines)

    def render_to_file(self, content: str, path: str | Path) -> Path:
        """Write report content to a file in docs/ directory."""
        out_path = Path(path)
        if not out_path.is_absolute():
            out_path = self._repo_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        log.info("Report written to %s", out_path)
        return out_path

    def update_section(
        self, section: str, content: str, paper_path: str | Path
    ) -> str:
        """Update a specific section using marker-based replacement."""
        from agent.paper import _replace_between

        path = Path(paper_path)
        if not path.is_absolute():
            path = self._repo_root / path

        if not path.exists():
            # Create with the section
            markers = SECTION_MARKERS.get(section)
            if markers:
                start, end = markers
                text = f"# Research Report\n\n## {section.title()}\n\n{start}\n{content}\n{end}\n"
            else:
                text = f"# Research Report\n\n## {section.title()}\n\n{content}\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return text

        existing = path.read_text(encoding="utf-8")
        markers = SECTION_MARKERS.get(section)
        if markers:
            start, end = markers
            updated = _replace_between(existing, start, end, content)
        else:
            # Append if no markers defined
            updated = existing.rstrip() + f"\n\n## {section.title()}\n\n{content}\n"

        path.write_text(updated, encoding="utf-8")
        return updated

    # -- Section renderers -----------------------------------------------

    def _render_summary(self) -> str:
        """Render summary from knowledge base stats."""
        hypotheses = self._db.get_knowledge_by_type("hypothesis", limit=100)
        experiments = self._db.get_knowledge_by_type("experiment", limit=100)
        findings = self._db.get_knowledge_by_type("finding", limit=100)
        conclusions = self._db.get_knowledge_by_type("conclusion", limit=100)
        observations = self._db.get_knowledge_by_type("observation", limit=100)

        active_h = [h for h in hypotheses if h.get("status") == "active"]
        confirmed = [h for h in hypotheses if h.get("status") == "confirmed"]
        refuted = [h for h in hypotheses if h.get("status") == "refuted"]

        lines = [
            f"This report summarizes research on **{self._config.topic}**.",
            "",
            f"- **{len(hypotheses)}** hypotheses ({len(active_h)} active, "
            f"{len(confirmed)} confirmed, {len(refuted)} refuted)",
            f"- **{len(experiments)}** experiments conducted",
            f"- **{len(findings)}** findings recorded",
            f"- **{len(conclusions)}** formal conclusions reached",
            f"- **{len(observations)}** observations logged",
        ]

        # Add sandbox execution stats if available
        try:
            executions = self._db.get_recent_sandbox_executions(limit=100)
            if executions:
                successful = sum(1 for e in executions if e["returncode"] == 0)
                lines.append(
                    f"- **{len(executions)}** sandbox executions "
                    f"({successful} successful)"
                )
        except Exception:
            pass

        return "\n".join(lines)

    def _render_hypotheses(self) -> str:
        """Render hypothesis trees."""
        from .knowledge import get_hypothesis_tree

        hypotheses = self._db.get_knowledge_by_type("hypothesis", limit=50)
        if not hypotheses:
            return "_No hypotheses recorded._"

        lines: list[str] = []
        for h in hypotheses:
            status = h.get("status", "active").upper()
            lines.append(f"### [{status}] {h['content'][:200]}")
            lines.append("")

            tree = get_hypothesis_tree(self._db, h["id"])
            exps = tree.get("experiments", [])
            finds = tree.get("findings", [])
            concs = tree.get("conclusions", [])

            if exps:
                lines.append(f"**Experiments** ({len(exps)}):")
                for e in exps:
                    lines.append(f"- {e['content'][:150]}")
            if finds:
                lines.append(f"\n**Findings** ({len(finds)}):")
                for f in finds:
                    lines.append(f"- {f['content'][:150]}")
            if concs:
                lines.append(f"\n**Conclusion**:")
                for c in concs:
                    meta = json.loads(c.get("metadata_json", "{}"))
                    verdict = meta.get("verdict", "?")
                    lines.append(f"- [{verdict}] {c['content'][:200]}")

            # Citations
            cites = self._db.get_citations_for_knowledge(h["id"])
            if cites:
                lines.append(f"\n**Citations**:")
                for c in cites:
                    lines.append(
                        f"- {c.get('title', '?')[:80]} "
                        f"({c.get('context', '')})"
                    )

            lines.append("")

        return "\n".join(lines)

    def _render_experiments(self) -> str:
        """Render experiment details including sandbox executions."""
        experiments = self._db.get_knowledge_by_type("experiment", limit=50)
        if not experiments:
            return "_No experiments recorded._"

        lines: list[str] = []
        for e in experiments:
            lines.append(f"- **{e['content'][:200]}**")
            lines.append(f"  Source: {e.get('source', 'unknown')}")

        # Sandbox execution summary
        try:
            executions = self._db.get_recent_sandbox_executions(limit=20)
            if executions:
                lines.append("")
                lines.append("### Sandbox Executions")
                lines.append("")
                lines.append(self.generate_experiment_summary())
        except Exception:
            pass

        return "\n".join(lines)

    def _render_findings(self) -> str:
        """Render findings with metrics."""
        findings = self._db.get_knowledge_by_type("finding", limit=50)
        if not findings:
            return "_No findings recorded._"

        lines: list[str] = []
        for f in findings:
            lines.append(f"- {f['content'][:200]}")
            meta = json.loads(f.get("metadata_json", "{}"))
            if meta and meta != {}:
                metrics_str = ", ".join(
                    f"{k}={v}" for k, v in list(meta.items())[:5]
                )
                lines.append(f"  Metrics: {metrics_str}")

        return "\n".join(lines)

    def _render_conclusions(self) -> str:
        """Render formal conclusions."""
        conclusions = self._db.get_knowledge_by_type("conclusion", limit=50)
        if not conclusions:
            return "_No conclusions reached._"

        lines: list[str] = []
        for c in conclusions:
            meta = json.loads(c.get("metadata_json", "{}"))
            verdict = meta.get("verdict", "?")
            icon = {"confirmed": "[+]", "refuted": "[-]", "inconclusive": "[?]"}.get(
                verdict, "[?]"
            )
            lines.append(f"{icon} {c['content'][:300]}")

        return "\n".join(lines)

    def _render_references(self) -> str:
        """Render cited papers."""
        # Get all papers that are cited by any knowledge entry
        all_knowledge = self._db.get_recent_knowledge(limit=200)
        cited_paper_ids: set[str] = set()
        for k in all_knowledge:
            cites = self._db.get_citations_for_knowledge(k["id"])
            for c in cites:
                cited_paper_ids.add(c["paper_id"])

        if not cited_paper_ids:
            return "_No references cited._"

        lines: list[str] = []
        for pid in sorted(cited_paper_ids):
            paper = self._db.get_paper_by_id(pid)
            if not paper:
                continue
            authors = paper.get("authors", "")[:60]
            year = paper.get("year", "?")
            title = paper.get("title", "Unknown")
            doi = paper.get("doi", "")
            ref = f"- {authors} ({year}). *{title}*."
            if doi:
                ref += f" DOI: {doi}"
            lines.append(ref)

        return "\n".join(lines)
