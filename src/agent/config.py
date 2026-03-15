from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


BudgetName = Literal["fast", "standard", "deep"]
ApprovalMode = Literal["auto", "prompt", "dry-run"]
GitPolicy = Literal["branch-commit", "none", "patch"]


@dataclass(frozen=True)
class Budget:
    name: BudgetName
    max_iters: int
    max_steps: int
    sample_limit: int
    hypothesis_limit: int


def budget_profile(name: BudgetName) -> Budget:
    if name == "fast":
        return Budget(name="fast", max_iters=6, max_steps=60, sample_limit=50, hypothesis_limit=2)
    if name == "standard":
        return Budget(name="standard", max_iters=10, max_steps=120, sample_limit=200, hypothesis_limit=4)
    if name == "deep":
        return Budget(name="deep", max_iters=20, max_steps=300, sample_limit=500, hypothesis_limit=8)
    raise ValueError(f"Unknown budget: {name}")


@dataclass(frozen=True)
class AgentConfig:
    repo_root: Path
    run_dir: Path
    run_id: str
    goal: str
    data_dir: Path

    model: str
    forecast_model: str
    budget: Budget

    approval_mode: ApprovalMode
    git_policy: GitPolicy
    allow_web: bool
    allow_code_edit: bool

    paper_path: Path

    timeout_s: int

    @property
    def docs_root(self) -> Path:
        return (self.repo_root / "docs").resolve()

    @property
    def runs_root(self) -> Path:
        return self.run_dir.parent.resolve()

