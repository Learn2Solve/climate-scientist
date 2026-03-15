from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import Agent
from .config import AgentConfig
from .hypotheses import inject_baseline_guidance, subset_payloads
from .memory import RunMemory
from .paper import paper_template_text, runs_index_entry, update_paper_text
from .tool_registry import ToolContext, handlers, validate_cmd
from .tools import run_shell, tail


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExperimentResult:
    id: str
    label: str
    dir: Path
    models: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_env(var: str) -> bool:
    return bool(os.environ.get(var))


def _run_and_log(cfg: AgentConfig, memory: RunMemory, *, step_id: str, cmd: str, cwd: Path, timeout_s: int, artifacts_dir: Path) -> dict[str, Any]:
    ok_cmd, reason = validate_cmd(cmd, cfg)
    if not ok_cmd:
        record = {"type": "shell", "step_id": step_id, "cmd": cmd, "cwd": str(cwd), "ok": False, "error": reason}
        memory.log_tool_call(record)
        return record

    res = run_shell(cmd, cwd=cwd, timeout_s=timeout_s)
    out_path = artifacts_dir / f"{step_id}.stdout.txt"
    err_path = artifacts_dir / f"{step_id}.stderr.txt"
    out_path.write_text(res.stdout or "", encoding="utf-8")
    err_path.write_text(res.stderr or "", encoding="utf-8")
    record = {
        "type": "shell",
        "step_id": step_id,
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": res.returncode,
        "ok": res.returncode == 0,
        "stdout_tail": tail(res.stdout, 2000),
        "stderr_tail": tail(res.stderr, 2000),
        "stdout_file": str(out_path),
        "stderr_file": str(err_path),
        "started_at": res.started_at,
        "ended_at": res.ended_at,
        "duration_s": res.duration_s,
    }
    memory.log_tool_call(record)
    return record


def _slug(text: str, *, max_len: int = 48) -> str:
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out).strip("_")
    s = s[:max_len] if s else "item"
    return s


def _attach_ri_metrics(
    cfg: AgentConfig,
    memory: RunMemory,
    *,
    exp_id: str,
    payloads: Path,
    truth: Path,
    models: dict[str, Any],
    artifacts_dir: Path,
    lead_hours: int = 24,
    threshold_kt: float = 30.0,
) -> dict[str, Any]:
    py = "uv run --no-project python"
    for label, m in models.items():
        if not isinstance(m, dict):
            continue
        pred_path_str = m.get("predictions")
        if not pred_path_str:
            continue
        pred_path = Path(str(pred_path_str))
        if not pred_path.is_absolute():
            pred_path = (cfg.repo_root / pred_path).resolve()
        if not pred_path.exists():
            continue

        slug = _slug(str(label))
        out_json = (artifacts_dir / f"ri_{slug}.json").resolve()
        step_id = f"{exp_id}_ri_{slug}"
        _run_and_log(
            cfg,
            memory,
            step_id=step_id,
            cmd=(
                f"{py} src/ri_metrics.py "
                f"--payloads {shlex.quote(str(payloads))} "
                f"--truth {shlex.quote(str(truth))} "
                f"--predictions {shlex.quote(str(pred_path))} "
                f"--lead-hours {int(lead_hours)} "
                f"--threshold-kt {float(threshold_kt)} "
                f"--out-json {shlex.quote(str(out_json))}"
            ),
            cwd=cfg.repo_root,
            timeout_s=cfg.timeout_s,
            artifacts_dir=artifacts_dir,
        )
        if out_json.exists():
            try:
                m["ri"] = _read_json(out_json)
            except Exception:
                pass
    return models


def _best_model(experiments: list[ExperimentResult]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None

    for exp in experiments:
        for model, m in exp.models.items():
            if not isinstance(m, dict):
                continue
            vjr = float(m.get("valid_json_rate") or 0.0)
            if vjr < 0.8:
                continue
            track = m.get("track") or {}
            wind = m.get("wind") or {}
            try:
                t24 = float((track.get("24") or {}).get("mean"))
                t48 = float((track.get("48") or {}).get("mean"))
                t72 = float((track.get("72") or {}).get("mean"))
            except Exception:
                continue
            w24 = float((wind.get("24") or {}).get("mean") or 0.0)
            key = (t24, t48, t72, w24)
            if best_key is None or key < best_key:
                best_key = key
                best = {"experiment": exp.id, "model": model, "track_mean": {"24": t24, "48": t48, "72": t72}, "wind_mean_24": w24, "valid_json_rate": vjr}

    return best or {}


def bootstrap_context(cfg: AgentConfig) -> dict[str, Any]:
    repo = cfg.repo_root
    files = [
        repo / "SPEC.md",
        repo / "docs" / "DISCUSSION.md",
        repo / "docs" / "RELATED_WORK.md",
        repo / "docs" / "DEMO.md",
        repo / "results" / "metrics.md",
    ]
    read = {}
    for p in files:
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            read[str(p.relative_to(repo))] = text[:4000] + ("...(truncated)" if len(text) > 4000 else "")

    data_preview = {}
    payloads = cfg.data_dir / "payloads.jsonl"
    truth = cfg.data_dir / "truth.jsonl"
    if payloads.exists():
        data_preview["payloads_head"] = payloads.read_text(encoding="utf-8").splitlines()[:2]
    if truth.exists():
        data_preview["truth_head"] = truth.read_text(encoding="utf-8").splitlines()[:2]

    return {"files_preview": read, "data_preview": data_preview}


def run_experiment_baselines(cfg: AgentConfig, memory: RunMemory) -> ExperimentResult:
    exp_id = "exp_001_baselines"
    exp_dir = (cfg.run_dir / "experiments" / exp_id).resolve()
    artifacts = exp_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    payloads = cfg.data_dir / "payloads.jsonl"
    truth = cfg.data_dir / "truth.jsonl"
    persistence = exp_dir / "persistence.jsonl"
    kinematic = exp_dir / "kinematic.jsonl"
    trend = exp_dir / "trend.jsonl"
    ri_gate = exp_dir / "ri_gate.jsonl"
    ri_logit = exp_dir / "ri_logit.jsonl"
    ri_logit_meta = exp_dir / "ri_logit_meta.json"
    metrics = exp_dir / "metrics.json"

    py = "uv run --no-project python"

    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_persistence",
        cmd=f"{py} src/baselines.py --payloads {shlex.quote(str(payloads))} --method persistence --out {shlex.quote(str(persistence))}",
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )
    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_kinematic",
        cmd=f"{py} src/baselines.py --payloads {shlex.quote(str(payloads))} --method kinematic --out {shlex.quote(str(kinematic))}",
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )
    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_trend",
        cmd=f"{py} src/baselines.py --payloads {shlex.quote(str(payloads))} --method trend --out {shlex.quote(str(trend))}",
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )
    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_ri_gate",
        cmd=f"{py} src/baselines.py --payloads {shlex.quote(str(payloads))} --method ri_gate --out {shlex.quote(str(ri_gate))}",
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )
    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_ri_logit",
        cmd=(
            f"{py} src/ri_logit_baseline.py "
            f"--payloads {shlex.quote(str(payloads))} "
            f"--truth {shlex.quote(str(truth))} "
            f"--out {shlex.quote(str(ri_logit))} "
            f"--out-meta {shlex.quote(str(ri_logit_meta))}"
        ),
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )
    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_score",
        cmd=(
            f"{py} src/report_metrics.py --truth {shlex.quote(str(truth))} "
            f"--model persistence:{shlex.quote(str(persistence))} "
            f"--model kinematic:{shlex.quote(str(kinematic))} "
            f"--model trend:{shlex.quote(str(trend))} "
            f"--model ri_gate:{shlex.quote(str(ri_gate))} "
            f"--model ri_logit:{shlex.quote(str(ri_logit))} "
            f"--out-json {shlex.quote(str(metrics))}"
        ),
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )

    if not metrics.exists():
        memory.log_tool_call(
            {
                "type": "note",
                "message": f"Baseline scoring did not produce metrics.json; see {artifacts}",
                "experiment": exp_id,
            }
        )
        return ExperimentResult(id=exp_id, label="Baselines", dir=exp_dir, models={})

    models = _read_json(metrics)
    models = _attach_ri_metrics(cfg, memory, exp_id=exp_id, payloads=payloads, truth=truth, models=models, artifacts_dir=artifacts)
    memory.write_json(metrics, models)
    return ExperimentResult(id=exp_id, label="Baselines", dir=exp_dir, models=models)


def run_experiment_llm_baseline(cfg: AgentConfig, memory: RunMemory) -> ExperimentResult | None:
    if not _has_env("DEEPSEEK_API_KEY"):
        memory.log_tool_call({"type": "note", "message": "Skipping LLM baseline: DEEPSEEK_API_KEY not set."})
        return None

    exp_id = "exp_002_llm_baseline"
    exp_dir = (cfg.run_dir / "experiments" / exp_id).resolve()
    artifacts = exp_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    payloads_full = cfg.data_dir / "payloads.jsonl"
    truth_full = cfg.data_dir / "truth.jsonl"

    payloads = exp_dir / "payloads.jsonl"
    truth = exp_dir / "truth.jsonl"
    subset_meta = subset_payloads(payloads_full, truth_full, payloads, truth, limit=cfg.budget.sample_limit, seed=42)
    tool_ctx = ToolContext(cfg=cfg, memory=memory, touched_paths=set())
    write = handlers()["write_text"]
    write(tool_ctx, {"path": str((exp_dir / "subset_meta.json").relative_to(cfg.repo_root)), "content": json.dumps(subset_meta.to_dict(), indent=2) + "\n"})

    preds = exp_dir / "predictions.jsonl"
    metrics = exp_dir / "metrics.json"

    py = "uv run --no-project python"

    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_forecast",
        cmd=(
            f"{py} src/run_forecaster_jsonl.py "
            f"--payloads {shlex.quote(str(payloads))} "
            f"--out {shlex.quote(str(preds))} "
            f"--model {shlex.quote(cfg.forecast_model)} --json --max-tokens 2048 "
            f"--timeout {min(60, int(cfg.timeout_s))}"
        ),
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )
    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_score",
        cmd=(
            f"{py} src/report_metrics.py --truth {shlex.quote(str(truth))} "
            f"--model {shlex.quote(cfg.forecast_model)}:{shlex.quote(str(preds))} "
            f"--out-json {shlex.quote(str(metrics))}"
        ),
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )

    models = _read_json(metrics)
    models = _attach_ri_metrics(cfg, memory, exp_id=exp_id, payloads=payloads, truth=truth, models=models, artifacts_dir=artifacts)
    memory.write_json(metrics, models)
    return ExperimentResult(id=exp_id, label="LLM baseline", dir=exp_dir, models=models)


def run_experiment_guided(cfg: AgentConfig, memory: RunMemory) -> ExperimentResult | None:
    if not _has_env("DEEPSEEK_API_KEY"):
        return None

    exp_id = "exp_003_guided_persistence"
    exp_dir = (cfg.run_dir / "experiments" / exp_id).resolve()
    artifacts = exp_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # Reuse the subset from exp_002 if it exists; otherwise create one.
    subset_dir = (cfg.run_dir / "experiments" / "exp_002_llm_baseline").resolve()
    payloads = subset_dir / "payloads.jsonl"
    truth = subset_dir / "truth.jsonl"
    if not payloads.exists() or not truth.exists():
        payloads_full = cfg.data_dir / "payloads.jsonl"
        truth_full = cfg.data_dir / "truth.jsonl"
        payloads = exp_dir / "payloads.jsonl"
        truth = exp_dir / "truth.jsonl"
        subset_meta = subset_payloads(payloads_full, truth_full, payloads, truth, limit=cfg.budget.sample_limit, seed=42)
        tool_ctx = ToolContext(cfg=cfg, memory=memory, touched_paths=set())
        write = handlers()["write_text"]
        write(tool_ctx, {"path": str((exp_dir / "subset_meta.json").relative_to(cfg.repo_root)), "content": json.dumps(subset_meta.to_dict(), indent=2) + "\n"})

    py = "uv run --no-project python"

    baseline_preds = exp_dir / "persistence.jsonl"
    guided_payloads = exp_dir / "payloads_guided.jsonl"
    preds = exp_dir / "predictions.jsonl"
    metrics = exp_dir / "metrics.json"

    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_baseline",
        cmd=f"{py} src/baselines.py --payloads {shlex.quote(str(payloads))} --method persistence --out {shlex.quote(str(baseline_preds))}",
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )

    inject_meta = inject_baseline_guidance(payloads, baseline_preds, guided_payloads, label="persistence")
    tool_ctx = ToolContext(cfg=cfg, memory=memory, touched_paths=set())
    write = handlers()["write_text"]
    write(tool_ctx, {"path": str((exp_dir / "inject_meta.json").relative_to(cfg.repo_root)), "content": json.dumps(inject_meta, indent=2) + "\n"})
    memory.log_tool_call({"type": "note", "message": f"Injected guidance into {guided_payloads}."})

    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_forecast",
        cmd=(
            f"{py} src/run_forecaster_jsonl.py "
            f"--payloads {shlex.quote(str(guided_payloads))} "
            f"--out {shlex.quote(str(preds))} "
            f"--model {shlex.quote(cfg.forecast_model)} --json --max-tokens 2048 "
            f"--timeout {min(60, int(cfg.timeout_s))}"
        ),
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )
    _run_and_log(
        cfg,
        memory,
        step_id=f"{exp_id}_score",
        cmd=(
            f"{py} src/report_metrics.py --truth {shlex.quote(str(truth))} "
            f"--model guided_{shlex.quote(cfg.forecast_model)}:{shlex.quote(str(preds))} "
            f"--out-json {shlex.quote(str(metrics))}"
        ),
        cwd=cfg.repo_root,
        timeout_s=cfg.timeout_s,
        artifacts_dir=artifacts,
    )

    models = _read_json(metrics)
    models = _attach_ri_metrics(cfg, memory, exp_id=exp_id, payloads=payloads, truth=truth, models=models, artifacts_dir=artifacts)
    memory.write_json(metrics, models)
    return ExperimentResult(id=exp_id, label="Guided (persistence)", dir=exp_dir, models=models)


def run_auto(cfg: AgentConfig, memory: RunMemory) -> dict[str, Any]:
    bootstrap = bootstrap_context(cfg)
    tool_ctx = ToolContext(cfg=cfg, memory=memory, touched_paths=set())
    write = handlers()["write_text"]
    append = handlers()["append_text"]
    make_patch = handlers()["make_patch"]

    plan = {
        "created_at": utc_now(),
        "run_id": cfg.run_id,
        "goal": cfg.goal,
        "workflow": {"name": "auto", "description": "Auto climate scientist run (v1)."},
        "budget": cfg.budget.name,
        "policies": {
            "approval_mode": cfg.approval_mode,
            "git_policy": cfg.git_policy,
            "allow_web": cfg.allow_web,
            "allow_code_edit": cfg.allow_code_edit,
        },
        "data": str(cfg.data_dir),
        "bootstrap": {"files": list(bootstrap.get("files_preview", {}).keys())},
        "hypotheses": [
            {
                "id": "kinematic_motion",
                "title": "Use motion to advect the storm center (kinematic baseline)",
                "rationale": "A simple great-circle advection baseline can outperform persistence when motion is reliable.",
                "expected_effect": "Lower track MAE vs persistence on some subsets.",
            },
            {
                "id": "guided_persistence",
                "title": "Inject persistence baseline as guidance",
                "rationale": "LLM may benefit from a simple physical baseline as a reference trajectory/intensity.",
                "expected_effect": "Improve track MAE vs unguided LLM baseline, at least at 24h.",
            }
        ],
    }
    memory.write_json(memory.paths.plan_json, plan)

    if cfg.approval_mode == "dry-run":
        memory.write_text(memory.paths.report_md, "# DRY RUN\n\nPlan written; execution skipped.\n")
        return {"ok": True, "skipped": True}

    experiments: list[ExperimentResult] = []
    experiments.append(run_experiment_baselines(cfg, memory))
    llm_base = run_experiment_llm_baseline(cfg, memory)
    if llm_base:
        experiments.append(llm_base)
    guided = run_experiment_guided(cfg, memory)
    if guided:
        experiments.append(guided)

    best = _best_model(experiments)
    metrics_summary = {
        "meta": {"created_at": utc_now(), "run_id": cfg.run_id, "goal": cfg.goal, "budget": cfg.budget.name, "data": str(cfg.data_dir)},
        "experiments": [{"id": e.id, "label": e.label, "dir": str(e.dir), "models": e.models} for e in experiments],
        "best": best,
    }
    memory.write_json(memory.paths.metrics_json, metrics_summary)

    # Paper updates (deterministic baseline), with audit logging.
    rel_paper = str(cfg.paper_path.relative_to(cfg.repo_root))
    existing_paper = paper_template_text()
    if cfg.paper_path.exists():
        existing_paper = cfg.paper_path.read_text(encoding="utf-8", errors="replace")
    updated_paper = update_paper_text(existing_paper, metrics_summary, run_id=cfg.run_id, run_dir=cfg.run_dir, paper_path=cfg.paper_path)
    write(tool_ctx, {"path": rel_paper, "content": updated_paper})

    runs_md = (cfg.repo_root / "docs" / "RUNS.md").resolve()
    if not runs_md.exists():
        write(tool_ctx, {"path": str(runs_md.relative_to(cfg.repo_root)), "content": "# Runs Index\n\n"})
    append(
        tool_ctx,
        {
            "path": str(runs_md.relative_to(cfg.repo_root)),
            "content": runs_index_entry(run_id=cfg.run_id, goal=cfg.goal, run_dir=cfg.run_dir, repo_root=cfg.repo_root, best=best),
        },
    )

    # Optional LLM pass: refine narrative and propose hypotheses.
    llm_used = False
    if _has_env("OPENAI_API_KEY") and cfg.approval_mode != "dry-run":
        try:
            agent = Agent(cfg, memory)
            _ = agent.run(bootstrap=bootstrap, touched_paths=tool_ctx.touched_paths)
            llm_used = True
        except Exception as exc:
            memory.log_tool_call({"type": "note", "message": f"LLM agent failed: {exc}"})

    # Git policy (branch+commit or patch).
    git_out: dict[str, Any] = {}
    touched_docs = sorted(
        str(p.relative_to(cfg.repo_root))
        for p in tool_ctx.touched_paths
        if p.exists() and p.is_file() and p.resolve().is_relative_to(cfg.docs_root)
    )
    patch_paths = touched_docs or ["docs"]
    if cfg.git_policy == "patch":
        patch_rel = str((cfg.run_dir / "artifacts" / "changes.patch").relative_to(cfg.repo_root))
        out = make_patch(tool_ctx, {"out_path": patch_rel, "paths": patch_paths})
        git_out = {"policy": "patch", **out}
    elif cfg.git_policy == "branch-commit":
        branch = f"agent/{cfg.run_id}"
        msg = f"Agent: {cfg.goal} ({cfg.run_id})"
        try:
            r1 = _run_and_log(cfg, memory, step_id="git_checkout_branch", cmd=f"git checkout -b {shlex.quote(branch)}", cwd=cfg.repo_root, timeout_s=60, artifacts_dir=(cfg.run_dir / "artifacts"))
            add_cmd = "git add " + " ".join(shlex.quote(p) for p in patch_paths)
            r2 = _run_and_log(cfg, memory, step_id="git_add_docs", cmd=add_cmd, cwd=cfg.repo_root, timeout_s=60, artifacts_dir=(cfg.run_dir / "artifacts"))
            r3 = _run_and_log(cfg, memory, step_id="git_commit", cmd=f"git commit -m {shlex.quote(msg)}", cwd=cfg.repo_root, timeout_s=60, artifacts_dir=(cfg.run_dir / "artifacts"))
            sha_res = _run_and_log(cfg, memory, step_id="git_rev_parse", cmd="git rev-parse HEAD", cwd=cfg.repo_root, timeout_s=60, artifacts_dir=(cfg.run_dir / "artifacts"))
            if not (r1.get("ok") and r2.get("ok") and r3.get("ok") and sha_res.get("ok")):
                raise RuntimeError("git commit sequence failed")
            sha = (sha_res.get("stdout_tail") or "").strip().splitlines()[-1]
            git_out = {"policy": "branch-commit", "branch": branch, "sha": sha}
        except Exception as exc:
            # Fallback to patch if commit fails (e.g. missing git user config).
            patch_rel = str((cfg.run_dir / "artifacts" / "changes.patch").relative_to(cfg.repo_root))
            out = make_patch(tool_ctx, {"out_path": patch_rel, "paths": patch_paths})
            git_out = {"policy": "patch_fallback", "error": str(exc), **out}

    # Report
    lines = []
    lines.append(f"# Climate Scientist Auto Run: `{cfg.run_id}`\n")
    lines.append(f"- created_at: {metrics_summary['meta']['created_at']}")
    lines.append(f"- budget: `{cfg.budget.name}`")
    lines.append(f"- data: `{cfg.data_dir}`")
    lines.append(f"- openai_used: `{llm_used}`")
    lines.append(f"- deepseek_used: `{_has_env('DEEPSEEK_API_KEY')}`")
    lines.append("")
    lines.append("## Goal\n")
    lines.append(cfg.goal + "\n")
    lines.append("## Best\n")
    lines.append("```json")
    lines.append(json.dumps(best, indent=2))
    lines.append("```\n")
    lines.append("## Outputs\n")
    lines.append(f"- metrics: `{memory.paths.metrics_json}`")
    lines.append(f"- paper: `{cfg.paper_path}`")
    lines.append(f"- runs index: `{cfg.repo_root / 'docs' / 'RUNS.md'}`")
    if git_out:
        lines.append(f"- git: `{json.dumps(git_out)}`")
    lines.append("")
    lines.append("## Reproduce\n")
    cmd = (
        f"uv run --no-project python src/agent_cli.py "
        f"--goal {shlex.quote(cfg.goal)} "
        f"--workflow auto "
        f"--data {shlex.quote(str(cfg.data_dir.relative_to(cfg.repo_root) if cfg.data_dir.is_relative_to(cfg.repo_root) else cfg.data_dir))} "
        f"--budget {shlex.quote(cfg.budget.name)} "
        f"--git-policy {shlex.quote(cfg.git_policy)} "
        f"--out {shlex.quote(str(cfg.run_dir.parent.relative_to(cfg.repo_root) if cfg.run_dir.parent.is_relative_to(cfg.repo_root) else cfg.run_dir.parent))}"
    )
    lines.append("```bash")
    lines.append(cmd)
    lines.append("```")
    lines.append("")
    if not _has_env("OPENAI_API_KEY"):
        lines.append("- Note: set `OPENAI_API_KEY` to enable the LLM tool-calling agent (paper rewriting + hypothesis exploration).")
    if not _has_env("DEEPSEEK_API_KEY"):
        lines.append("- Note: set `DEEPSEEK_API_KEY` to enable DeepSeek forecast experiments (LLM baseline + guided hypothesis).")
    lines.append("")
    memory.write_text(memory.paths.report_md, "\n".join(lines))

    return {"ok": True, "metrics": metrics_summary, "git": git_out}
