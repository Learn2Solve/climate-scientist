from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scratchpad:
    goal: str
    notes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    tool_summaries: list[str] = field(default_factory=list)

    def add_note(self, text: str) -> None:
        if text:
            self.notes.append(text)

    def add_failure(self, text: str) -> None:
        if text:
            self.failures.append(text)

    def add_tool_summary(self, text: str) -> None:
        if text:
            self.tool_summaries.append(text)

    def summary_for_prompt(self, max_chars: int = 4000) -> str:
        parts: list[str] = []
        if self.notes:
            parts.append("NOTES:\n" + "\n".join(f"- {n}" for n in self.notes[-10:]))
        if self.failures:
            parts.append("FAILURES:\n" + "\n".join(f"- {f}" for f in self.failures[-10:]))
        if self.tool_summaries:
            parts.append("TOOLS:\n" + "\n".join(f"- {t}" for t in self.tool_summaries[-15:]))
        text = "\n\n".join(parts).strip()
        if len(text) > max_chars:
            return text[-max_chars:]
        return text

