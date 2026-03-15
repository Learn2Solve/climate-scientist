#!/usr/bin/env python3
"""Swarm CLI — run autonomous research cycles from the command line.

Usage:
    # Single research cycle
    uv run python src/swarm_cli.py --goal "Improve RI detection for tropical cyclones" \
        --predictions sim_outputs/predictions.jsonl \
        --truth sim_outputs/truth.jsonl

    # Anomaly analysis only (no hypothesis generation)
    uv run python src/swarm_cli.py --anomalies-only \
        --predictions sim_outputs/predictions.jsonl \
        --truth sim_outputs/truth.jsonl

    # Multi-cycle autonomous run
    uv run python src/swarm_cli.py --goal "Discover novel RI predictors" \
        --cycles 5 --max-hypotheses 3

    # Quick cycle (skip literature scan)
    uv run python src/swarm_cli.py --goal "Why do LLMs fail at RI?" --fast
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

VERSION = "0.1.0"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("swarm_cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Climate Research Agent Swarm — autonomous discovery cycles",
    )
    parser.add_argument("--goal", type=str, help="Research goal / question")
    parser.add_argument("--predictions", type=str, help="Path to predictions JSONL")
    parser.add_argument("--truth", type=str, help="Path to truth JSONL")
    parser.add_argument("--cycles", type=int, default=1, help="Number of research cycles (default: 1)")
    parser.add_argument("--max-hypotheses", type=int, default=3, help="Max hypotheses per cycle")
    parser.add_argument("--fast", action="store_true", help="Skip literature scan")
    parser.add_argument("--anomalies-only", action="store_true", help="Only run anomaly analysis")
    parser.add_argument("--paper", action="store_true", help="Generate a full scientific paper")
    parser.add_argument("--paper-title", type=str, help="Paper title (auto-generated if omitted)")
    parser.add_argument("--max-revisions", type=int, default=2, help="Max paper revision rounds")
    parser.add_argument("--config", type=str, help="Path to config.yml")
    parser.add_argument("--out", type=str, default="runs", help="Output directory")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(f"Climate Research Swarm v{VERSION}")
        sys.exit(0)

    # Add src/ to path
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from daemon.config import load_config, create_default_config
    from daemon.database import DaemonDatabase

    # Load config
    cfg = load_config(Path(args.config) if args.config else None)
    if not cfg:
        repo_root = Path(__file__).resolve().parent.parent
        cfg = create_default_config(
            topic=args.goal or "climate science research",
            repo_root=repo_root,
        )

    # Initialize DB
    db = DaemonDatabase(cfg.resolved_db_path)

    # Create inference client
    from daemon.inference import InferenceClient

    inference = InferenceClient(
        default_model=cfg.inference_model,
        max_tokens=cfg.max_tokens_per_turn,
        openai_api_key=cfg.openai_api_key,
        anthropic_api_key=cfg.anthropic_api_key,
        low_compute_model=cfg.low_compute_model,
        cli_proxy_url=cfg.cli_proxy_url,
        cli_proxy_api_key=cfg.cli_proxy_api_key,
        reasoning_effort=cfg.reasoning_effort,
    )

    # Create tool executor
    from daemon.tools import ToolExecutor

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_root / f"swarm_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)

    tool_executor = ToolExecutor(cfg, db, run_dir, run_dir.name, inference=inference)

    # Anomaly-only mode
    if args.anomalies_only:
        if not args.predictions or not args.truth:
            parser.error("--anomalies-only requires --predictions and --truth")

        from daemon.anomaly_analyzer import AnomalyAnalyzer
        import subprocess

        analyzer = AnomalyAnalyzer(config=cfg, db=db, inference=inference)

        # Load per-sample data via direct subprocess call
        log.info("Analyzing anomalies in %s vs %s", args.predictions, args.truth)
        proc = subprocess.run(
            [
                "python3", "src/evaluate_jsonl.py",
                "--predictions", args.predictions,
                "--truth", args.truth,
                "--per-sample-json",
            ],
            capture_output=True, text=True, timeout=60,
            cwd=str(cfg.repo_root),
        )

        per_sample = []
        metrics = {}
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                parsed = json.loads(proc.stdout)
                metrics = parsed.get("summary", parsed)
                per_sample = parsed.get("per_sample", [])
            except (json.JSONDecodeError, TypeError):
                log.error("Failed to parse evaluate_jsonl output")
        else:
            log.error("evaluate_jsonl failed: %s", proc.stderr[:300])

        anomalies = analyzer.analyze({"metrics": metrics, "per_sample": per_sample})

        # Output
        out_path = run_dir / "anomalies.json"
        out_path.write_text(json.dumps(anomalies, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Found %d anomalies → %s", len(anomalies), out_path)

        for a in anomalies:
            severity_bar = "█" * int(a.get("severity", 0) * 10)
            print(f"\n  [{a['type']}] {severity_bar} ({a.get('severity', 0):.2f})")
            print(f"  {a['description']}")
            print(f"  → {a.get('suggested_investigation', '')[:120]}")

        db.close()
        return

    # Full swarm cycle mode
    if not args.goal:
        parser.error("--goal is required for swarm cycles")

    from daemon.orchestrator import SwarmOrchestrator

    experiment_context = {}
    if args.predictions:
        experiment_context["predictions_path"] = args.predictions
    if args.truth:
        experiment_context["truth_path"] = args.truth

    orchestrator = SwarmOrchestrator(
        config=cfg,
        db=db,
        inference=inference,
        tool_executor=tool_executor,
    )

    all_results = []
    for cycle_num in range(1, args.cycles + 1):
        log.info("=" * 60)
        log.info("RESEARCH CYCLE %d/%d", cycle_num, args.cycles)
        log.info("=" * 60)

        result = orchestrator.run_cycle(
            research_goal=args.goal,
            max_hypotheses=args.max_hypotheses,
            skip_literature=args.fast,
            experiment_context=experiment_context or None,
            generate_paper=args.paper,
            paper_title=args.paper_title,
            max_revisions=args.max_revisions,
        )
        all_results.append(result)

        # Save cycle result
        cycle_path = run_dir / f"cycle_{cycle_num:03d}.json"
        cycle_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        if result.get("ok"):
            log.info(
                "Cycle %d complete: %d anomalies, %d hypotheses, %d validated",
                cycle_num,
                result.get("anomalies_found", 0),
                result.get("hypotheses_generated", 0),
                result.get("findings_validated", 0),
            )
        else:
            log.error("Cycle %d failed: %s", cycle_num, result.get("error", "unknown"))

    # Final summary
    summary_path = run_dir / "swarm_summary.json"
    summary = {
        "goal": args.goal,
        "cycles_completed": len(all_results),
        "total_anomalies": sum(r.get("anomalies_found", 0) for r in all_results),
        "total_hypotheses": sum(r.get("hypotheses_generated", 0) for r in all_results),
        "total_findings": sum(r.get("findings_validated", 0) for r in all_results),
        "cycles": all_results,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("Swarm complete. Summary → %s", summary_path)

    db.close()


if __name__ == "__main__":
    main()
