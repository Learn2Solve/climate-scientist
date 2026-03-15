"""Daemon configuration - YAML loader for <repo>/.agent/config.yml.

Port of automaton/src/config.ts adapted for local research context.
All agent state lives in <repo_root>/.agent/ alongside the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(p: str, base: Path | None = None) -> Path:
    if p.startswith("~"):
        return Path(os.path.expanduser(p))
    path = Path(p)
    if not path.is_absolute() and base:
        return base / path
    return path


def find_agent_dir(start: Path | None = None) -> Path:
    """Find .agent/ dir relative to repo root (walk up to find it)."""
    p = (start or Path.cwd()).resolve()
    while p != p.parent:
        candidate = p / ".agent"
        if candidate.is_dir():
            return candidate
        # Also check for config.yml directly
        if (candidate / "config.yml").exists():
            return candidate
        p = p.parent
    # Default: .agent/ in cwd
    return (start or Path.cwd()).resolve() / ".agent"


def get_config_path(repo_root: Path | None = None) -> Path:
    """Get path to config.yml."""
    if repo_root:
        return repo_root / ".agent" / "config.yml"
    return find_agent_dir() / "config.yml"


@dataclass
class DaemonConfig:
    # Identity
    name: str = "climate_researcher"
    topic: str = "rapid intensification"
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    # Persistence (relative to repo_root/.agent/)
    db_path: str = ".agent/state.db"
    identity_path: str = ".agent/identity.md"
    # Inference
    inference_model: str = "gpt-4o"
    low_compute_model: str = "gpt-4.1-mini"
    max_tokens_per_turn: int = 4096
    reasoning_effort: str | None = None  # "low", "medium", "high", "xhigh" for reasoning models
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    # Heartbeat
    heartbeat_config_path: str = ".agent/heartbeat.yml"
    heartbeat_interval_s: int = 60
    # Research
    data_dir: Path = field(default_factory=lambda: Path("data"))
    runs_root: Path = field(default_factory=lambda: Path("runs"))
    # Budget
    max_api_cost_cents: int = 5000  # $50
    max_turns_per_wake: int = 20
    allow_web: bool = True
    allow_code_edit: bool = False
    # Skills
    skills_dir: str = ".agent/skills"
    # CLIProxy (routes all models through local Claude Code / Codex proxy)
    cli_proxy_url: str | None = None
    cli_proxy_api_key: str | None = None
    # Social
    social_relay_port: int | None = None
    # Sandbox
    sandbox_docker_image: str = "python:3.13-slim"
    sandbox_docker_memory: str = "2g"
    sandbox_docker_cpus: str = "2"
    sandbox_docker_network: str = "none"
    sandbox_timeout_default_s: int = 120
    sandbox_timeout_max_s: int = 600
    # Data acquisition
    max_dataset_size_bytes: int = 100_000_000  # 100MB
    # Novelty checking
    novelty_advocate_model: str = "claude-opus-4-6"
    novelty_critic_model: str = "gpt-5.3-codex"
    novelty_auto_check: bool = False
    # Idea generation
    idea_generator_model: str | None = None  # defaults to inference_model
    # Vision
    vision_model: str = "gpt-4o"
    # Paper writing
    paper_reviewer_model: str = "claude-opus-4-6"
    paper_format: str = "markdown"
    # Experiment loops
    experiment_loop_max_trials: int = 50
    experiment_loop_convergence_threshold: float = 0.01
    # Statistical rigor
    significance_threshold: float = 0.05
    require_statistical_test: bool = True
    min_cross_validation_folds: int = 5
    # Debate
    debate_max_rounds: int = 3
    debate_proposer_model: str = "claude-opus-4-6"
    debate_opponent_model: str = "gpt-5.3-codex"
    # Verification
    auto_verify_results: bool = False
    reproducibility_tolerance: float = 0.05
    # Meta-learning
    meta_learning_enabled: bool = True
    # Causal reasoning
    causal_default_method: str = "backdoor"
    causal_significance_threshold: float = 0.05
    # Benchmarks
    benchmark_auto_check: bool = True
    # Tool learning
    tool_learning_enabled: bool = True
    max_learned_tools: int = 50
    # Safety
    safety_audit_enabled: bool = True
    phacking_test_threshold: int = 5
    harking_time_threshold_s: int = 60
    # Multi-agent specialization
    max_specialist_children: int = 4
    specialist_budget_cents: int = 500
    # Physics simulation
    physics_sim_timeout_s: int = 300
    physics_sim_resolution: int = 100
    # Literature / full-text paper reading
    literature_papers_dir: str = "data/papers"
    literature_max_pdf_bytes: int = 50_000_000  # 50MB
    literature_max_fulltext_chars: int = 500_000
    # Data APIs
    era5_api_key: str | None = None
    era5_api_url: str = "https://cds.climate.copernicus.eu/api"
    # Logging
    log_level: str = "info"
    version: str = "0.1.0"

    @property
    def agent_dir(self) -> Path:
        return self.repo_root / ".agent"

    @property
    def resolved_db_path(self) -> Path:
        return _resolve_path(self.db_path, self.repo_root)

    @property
    def resolved_identity_path(self) -> Path:
        return _resolve_path(self.identity_path, self.repo_root)

    @property
    def resolved_heartbeat_config_path(self) -> Path:
        return _resolve_path(self.heartbeat_config_path, self.repo_root)

    @property
    def resolved_skills_dir(self) -> Path:
        return _resolve_path(self.skills_dir, self.repo_root)


def load_config(config_path: Path | None = None) -> DaemonConfig | None:
    """Load config from YAML. Returns None if file does not exist.

    Search order: explicit path > <repo_root>/.agent/config.yml > cwd walk-up.
    """
    if config_path:
        path = config_path
    else:
        # Try repo root (parent of src/ where this file lives)
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidate = repo_root / ".agent" / "config.yml"
        if candidate.exists():
            path = candidate
        else:
            path = get_config_path()
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return _config_from_dict(raw)


def save_config(config: DaemonConfig, config_path: Path | None = None) -> None:
    """Persist config to YAML."""
    path = config_path or (config.repo_root / ".agent" / "config.yml")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data: dict[str, Any] = {}
    for f in fields(config):
        val = getattr(config, f.name)
        if isinstance(val, Path):
            val = str(val)
        data[f.name] = val
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


def create_default_config(
    topic: str = "rapid intensification",
    repo_root: Path | None = None,
    **overrides: Any,
) -> DaemonConfig:
    """Create a fresh config with sensible defaults."""
    kw: dict[str, Any] = {"topic": topic}
    if repo_root:
        kw["repo_root"] = repo_root
    # Pull API keys from environment if not provided
    kw["openai_api_key"] = overrides.pop(
        "openai_api_key", os.environ.get("OPENAI_API_KEY")
    )
    kw["anthropic_api_key"] = overrides.pop(
        "anthropic_api_key", os.environ.get("ANTHROPIC_API_KEY")
    )
    kw["cli_proxy_url"] = overrides.pop(
        "cli_proxy_url", os.environ.get("CLI_PROXY_URL")
    )
    kw["cli_proxy_api_key"] = overrides.pop(
        "cli_proxy_api_key", os.environ.get("CLI_PROXY_API_KEY")
    )
    # When CLIProxy is configured, default to Claude Code model names
    if kw.get("cli_proxy_url") and "inference_model" not in overrides:
        kw.setdefault("inference_model", "claude-sonnet-4")
        kw.setdefault("low_compute_model", "claude-haiku-4")
    # When Anthropic API key set but no OpenAI key, default to API model names
    elif kw.get("anthropic_api_key") and not kw.get("openai_api_key") and "inference_model" not in overrides:
        kw.setdefault("inference_model", "claude-sonnet-4-20250514")
        kw.setdefault("low_compute_model", "claude-haiku-4-20250414")
    kw.update(overrides)
    return DaemonConfig(**kw)


def init_agent_dir(repo_root: Path | None = None) -> Path:
    """Create <repo_root>/.agent/ with required subdirectories."""
    root = repo_root or Path.cwd()
    agent_dir = root / ".agent"
    agent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    (agent_dir / "skills").mkdir(exist_ok=True)
    (agent_dir / "children").mkdir(exist_ok=True)
    return agent_dir


def _config_from_dict(raw: dict[str, Any]) -> DaemonConfig:
    """Build DaemonConfig from a raw dict, coercing types as needed."""
    valid_fields = {f.name for f in fields(DaemonConfig)}
    kw: dict[str, Any] = {}
    for key, val in raw.items():
        if key not in valid_fields:
            continue
        if key in ("repo_root", "data_dir", "runs_root"):
            kw[key] = Path(val) if val else Path.cwd()
        else:
            kw[key] = val
    return DaemonConfig(**kw)
