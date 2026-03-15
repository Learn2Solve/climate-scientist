#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.auto import run_auto
from agent.config import AgentConfig, budget_profile
from agent.memory import RunMemory
from agent.planner import build_plan, find_workflow_file
from agent.runner import run_plan
from agent.safety import SafetyConfig


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def render_metrics_md(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "_No metrics.json produced._\n"
    # JSONL metrics format (report_metrics.py): {model: {samples, valid_json_rate, track{lead:{mean}}, wind{lead:{mean}}}}
    if all(isinstance(v, dict) and ("track" in v or "wind" in v) for v in metrics.values()):
        lines = []
        lines.append("| Model | Samples | Valid JSON | Lead | Track MAE (km, mean) | Wind MAE (kt, mean) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for model, m in metrics.items():
            if not isinstance(m, dict):
                continue
            samples = m.get("samples", "")
            vjr = m.get("valid_json_rate", "")
            track = m.get("track", {}) or {}
            wind = m.get("wind", {}) or {}
            for lead in sorted(track.keys(), key=lambda x: int(x)):
                te = track.get(lead, {}) or {}
                we = wind.get(lead, {}) or {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(model),
                            str(samples),
                            f"{float(vjr):.3f}" if isinstance(vjr, (int, float)) else str(vjr),
                            str(lead),
                            f"{float(te.get('mean', 0.0)):.2f}",
                            f"{float(we.get('mean', 0.0)):.2f}" if we else "",
                        ]
                    )
                    + " |"
                )
        lines.append("")
        return "\n".join(lines)

    # HURDAT2 persistence format (hurdat2_baseline.py): {split: {samples, track_mae_km{mean,median}, wind_mae_kt{mean,median}}}
    if any(k in metrics for k in ["train", "val", "test"]):
        lines = []
        lines.append("| Split | Samples | Track MAE (km, mean) | Track MAE (km, median) | Wind MAE (kt, mean) | Wind MAE (kt, median) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for split in ["train", "val", "test"]:
            m = metrics.get(split, {})
            if not isinstance(m, dict) or not m:
                continue
            track = m.get("track_mae_km", {}) or {}
            wind = m.get("wind_mae_kt", {}) or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        str(m.get("samples", "")),
                        f"{float(track.get('mean', 0.0)):.2f}",
                        f"{float(track.get('median', 0.0)):.2f}",
                        f"{float(wind.get('mean', 0.0)):.2f}",
                        f"{float(wind.get('median', 0.0)):.2f}",
                    ]
                )
                + " |"
            )
        lines.append("")
        return "\n".join(lines)

    return "```json\n" + json.dumps(metrics, indent=2) + "\n```\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Workflow-driven Climate Scientist Agent (MVP).")
    parser.add_argument("--goal", type=str, required=True, help="Natural language research goal.")
    parser.add_argument("--workflow", type=str, required=True, help="Workflow name (e.g. reproduce).")
    parser.add_argument("--data", type=str, default="sim_outputs", help="Dataset pointer (dir path).")
    parser.add_argument("--out", type=Path, default=Path("runs"), help="Runs output root.")
    parser.add_argument("--run-id", type=str, default="", help="Optional run id override.")
    parser.add_argument("--budget", choices=["fast", "standard", "deep"], default="fast")
    parser.add_argument("--model", type=str, default="gpt-5.2", help="Agent LLM model (workflow=auto).")
    parser.add_argument("--forecast-model", type=str, default="deepseek-chat", help="Forecast model (DeepSeek; workflow=auto).")
    parser.add_argument("--approval-mode", choices=["auto", "prompt", "dry-run"], default="auto")
    parser.add_argument("--git-policy", choices=["branch-commit", "none", "patch"], default="branch-commit")
    parser.add_argument("--allow-web", action="store_true", help="Allow web search/fetch tools for the LLM agent.")
    parser.add_argument("--allow-code-edit", action="store_true", help="Allow editing src/ (default: docs-only).")
    parser.add_argument("--paper-path", type=Path, default=Path("docs/PAPER.md"))
    parser.add_argument("--max-iters", type=int, default=0, help="Override agent max iterations (workflow=auto).")
    parser.add_argument("--max-steps", type=int, default=0, help="Override max steps (workflow=auto and YAML runner).")
    parser.add_argument("--timeout", type=int, default=600, help="Default per-step timeout (seconds).")
    parser.add_argument("--dry-run", action="store_true", help="Write plan + manifests, do not execute.")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    workflows_dir = repo_root / "src" / "workflows"

    out_root = args.out if args.out.is_absolute() else (repo_root / args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or f"{utc_now_compact()}_{args.workflow}_{secrets.token_hex(3)}"
    run_dir = (out_root / run_id).resolve()

    memory = RunMemory(run_dir)
    safety = SafetyConfig(
        max_steps=args.max_steps if args.max_steps else 50,
        default_timeout_s=args.timeout,
    )

    if args.dry_run:
        args.approval_mode = "dry-run"

    if args.workflow == "auto":
        budget = budget_profile(args.budget)
        if args.max_iters:
            budget = replace(budget, max_iters=int(args.max_iters))
        if args.max_steps:
            budget = replace(budget, max_steps=int(args.max_steps))

        data_dir = Path(args.data)
        if not data_dir.is_absolute():
            data_dir = (repo_root / data_dir).resolve()

        paper_path = args.paper_path if args.paper_path.is_absolute() else (repo_root / args.paper_path).resolve()

        cfg = AgentConfig(
            repo_root=repo_root,
            run_dir=run_dir,
            run_id=run_id,
            goal=args.goal,
            data_dir=data_dir,
            model=args.model,
            forecast_model=args.forecast_model,
            budget=budget,
            approval_mode=args.approval_mode,
            git_policy=args.git_policy,
            allow_web=bool(args.allow_web),
            allow_code_edit=bool(args.allow_code_edit),
            paper_path=paper_path,
            timeout_s=int(args.timeout),
        )

        # Audit chain core
        memory.record_env_lock(repo_root)
        manifest_files: list[Path] = [
            repo_root / "SPEC.md",
            repo_root / "docs" / "DISCUSSION.md",
            repo_root / "docs" / "RELATED_WORK.md",
            repo_root / "results" / "metrics.md",
            data_dir / "payloads.jsonl",
            data_dir / "truth.jsonl",
        ]
        memory.record_data_manifest(repo_root, manifest_files, extra={"workflow": "auto", "budget": cfg.budget.name})

        out = run_auto(cfg, memory)
        print(f"run_dir: {run_dir}")
        print(f"ok: {out.get('ok', False)}")
        return

    ctx: dict[str, Any] = {
        "goal": args.goal,
        "workflow": args.workflow,
        "data": args.data,
        "repo_root": str(repo_root),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "python": "uv run --no-project python",
    }

    workflow_path = find_workflow_file(args.workflow, workflows_dir)
    if not workflow_path.exists():
        available = sorted(p.stem for p in workflows_dir.glob("*.yaml"))
        raise SystemExit(f"workflow not found: {workflow_path} (available: {available})")
    plan_obj = build_plan(workflow_path, ctx)

    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "goal": args.goal,
        "context": {
            "workflow": args.workflow,
            "data": args.data,
            "repo_root": str(repo_root),
            "run_dir": str(run_dir),
        },
        **plan_obj.to_dict(),
    }
    memory.write_json(memory.paths.plan_json, plan)
    memory.record_env_lock(repo_root)

    manifest_files: list[Path] = [workflow_path, repo_root / "SPEC.md"]
    for p_str in plan_obj.data_files:
        p = Path(p_str)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        manifest_files.append(p)
    memory.record_data_manifest(repo_root, manifest_files, extra={"data": args.data, "workflow": args.workflow})

    if args.dry_run:
        print(f"DRY RUN: wrote plan to {memory.paths.plan_json}")
        print(f"run_dir: {run_dir}")
        return

    run_summary = run_plan(plan, repo_root=repo_root, memory=memory, safety=safety)

    metrics = {}
    if memory.paths.metrics_json.exists():
        try:
            metrics = json.loads(memory.paths.metrics_json.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    steps = run_summary.get("steps", [])
    failures = [s for s in steps if isinstance(s, dict) and not s.get("ok", True)]

    try:
        python_tokens = shlex.split(ctx["python"])
    except ValueError:
        python_tokens = [ctx["python"]]
    cmd_parts = [
        *python_tokens,
        "src/agent_cli.py",
        "--goal",
        args.goal,
        "--workflow",
        args.workflow,
        "--data",
        args.data,
        "--out",
        str(args.out),
    ]
    cmd = " ".join(shlex.quote(p) for p in cmd_parts)

    report_lines = []
    report_lines.append(f"# Climate Scientist Agent Run: `{run_id}`\n")
    report_lines.append(f"- created_at: {plan['created_at']}")
    report_lines.append(f"- workflow: `{args.workflow}`")
    report_lines.append(f"- data: `{args.data}`")
    report_lines.append(f"- ok: `{run_summary.get('ok', False)}`")
    report_lines.append("")
    report_lines.append("## Goal\n")
    report_lines.append(args.goal + "\n")
    report_lines.append("## Reproduce\n")
    report_lines.append("```bash")
    report_lines.append(cmd)
    report_lines.append("```\n")
    report_lines.append("## Metrics\n")
    report_lines.append(render_metrics_md(metrics))
    report_lines.append("## Steps\n")
    for s in steps:
        if not isinstance(s, dict):
            continue
        line = f"- {s.get('id')} ({s.get('type')}): ok={s.get('ok')}"
        if s.get("error"):
            line += f" error={s.get('error')}"
        report_lines.append(line)
    report_lines.append("")
    if failures:
        report_lines.append("## Failure Analysis\n")
        for s in failures:
            report_lines.append(f"- {s.get('id')}: {s.get('error')}")
        report_lines.append("")

    memory.write_text(memory.paths.report_md, "\n".join(report_lines))

    print(f"run_dir: {run_dir}")
    print(f"ok: {run_summary.get('ok', False)}")


if __name__ == "__main__":
    main()
