from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS_START = "<!-- AUTO_RESULTS_TABLE_START -->"
RESULTS_END = "<!-- AUTO_RESULTS_TABLE_END -->"
RI_START = "<!-- AUTO_RI_TABLE_START -->"
RI_END = "<!-- AUTO_RI_TABLE_END -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def paper_template_text() -> str:
    return (
        "\n".join(
            [
                "# Climate Scientist Agent for Hurricane Forecasting",
                "",
                "## Abstract",
                "",
                "_TODO: Summarize the claim and results._",
                "",
                "## Method",
                "",
                "We study a workflow-driven **climate scientist agent** that can run experiments,",
                "log an auditable evidence chain, and update a paper draft based on computed metrics.",
                "",
                "## Experiments",
                "",
                "- Task: hurricane track (km) + intensity (kt) at 24/48/72h.",
                "- Metrics: track MAE (great-circle), wind MAE, valid JSON rate.",
                "",
                "## Results",
                "",
                RESULTS_START,
                "_No results yet._",
                RESULTS_END,
                "",
                "## Rapid Intensification (RI)",
                "",
                RI_START,
                "_No RI metrics yet._",
                RI_END,
                "",
                "## Discussion / Limitations",
                "",
                "- Fairness: different models may consume different input modalities.",
                "- Robustness: LLM output validity requires strict schemas and verifiers.",
                "",
                "## Reproducibility",
                "",
                "Each run writes an auditable bundle under `runs/<run_id>/` including:",
                "`plan.json`, `tool_calls.jsonl`, `env.lock`, `data_manifest.json`, `metrics.json`, `report.md`.",
                "",
            ]
        )
        + "\n"
    )


def ensure_paper_template(path: Path) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paper_template_text(), encoding="utf-8")
    return True


def ensure_runs_index(path: Path) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Runs Index\n\n", encoding="utf-8")
    return True


def _replace_between(text: str, start: str, end: str, replacement: str) -> str:
    if start in text and end in text:
        pre = text.split(start, 1)[0]
        post = text.split(end, 1)[1]
        return pre + start + "\n" + replacement.rstrip() + "\n" + end + post
    return text.rstrip() + "\n\n" + start + "\n" + replacement.rstrip() + "\n" + end + "\n"


@dataclass(frozen=True)
class MetricRow:
    experiment: str
    model: str
    samples: int
    valid_json_rate: float
    track_mean: dict[int, float]
    wind_mean: dict[int, float]


def rows_from_metrics_summary(summary: dict[str, Any]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for exp in summary.get("experiments", []):
        if not isinstance(exp, dict):
            continue
        exp_id = str(exp.get("id", "exp"))
        models = exp.get("models", {})
        if not isinstance(models, dict):
            continue
        for model, m in models.items():
            if not isinstance(m, dict):
                continue
            samples = int(m.get("samples") or 0)
            vjr = float(m.get("valid_json_rate") or 0.0)
            track = m.get("track") or {}
            wind = m.get("wind") or {}
            track_mean = {int(h): float(v.get("mean")) for h, v in track.items() if isinstance(v, dict) and v.get("mean") is not None}
            wind_mean = {int(h): float(v.get("mean")) for h, v in wind.items() if isinstance(v, dict) and v.get("mean") is not None}
            rows.append(
                MetricRow(
                    experiment=exp_id,
                    model=str(model),
                    samples=samples,
                    valid_json_rate=vjr,
                    track_mean=track_mean,
                    wind_mean=wind_mean,
                )
            )
    return rows


def render_results_table(rows: list[MetricRow]) -> str:
    if not rows:
        return "_No metrics available._"
    leads = [24, 48, 72]
    header = ["Experiment", "Model", "Samples", "Valid JSON"] + [f"Track@{h} (km)" for h in leads] + [f"Wind@{h} (kt)" for h in leads]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        track_vals = [f"{r.track_mean.get(h, float('nan')):.2f}" if h in r.track_mean else "" for h in leads]
        wind_vals = [f"{r.wind_mean.get(h, float('nan')):.2f}" if h in r.wind_mean else "" for h in leads]
        lines.append(
            "| "
            + " | ".join(
                [
                    r.experiment,
                    r.model,
                    str(r.samples),
                    f"{r.valid_json_rate:.3f}",
                    *track_vals,
                    *wind_vals,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class RiRow:
    experiment: str
    model: str
    lead_hours: int
    threshold_kt: float
    samples: int
    truth_ri_rate: float
    valid_wind_pred_rate: float
    precision: float | None
    recall: float | None
    f1: float | None
    wind_mae_ri: float | None
    wind_mae_non_ri: float | None


def ri_rows_from_metrics_summary(summary: dict[str, Any]) -> list[RiRow]:
    rows: list[RiRow] = []
    for exp in summary.get("experiments", []):
        if not isinstance(exp, dict):
            continue
        exp_id = str(exp.get("id", "exp"))
        models = exp.get("models", {})
        if not isinstance(models, dict):
            continue
        for model, m in models.items():
            if not isinstance(m, dict):
                continue
            ri = m.get("ri")
            if not isinstance(ri, dict):
                continue
            meta = ri.get("meta", {}) or {}
            truth = ri.get("truth", {}) or {}
            pred = ri.get("pred", {}) or {}
            wind = ri.get("wind_mae", {}) or {}

            try:
                lead = int(meta.get("lead_hours") or 24)
            except Exception:
                lead = 24
            try:
                thr = float(meta.get("threshold_kt") or 30.0)
            except Exception:
                thr = 30.0

            samples = int(meta.get("samples") or m.get("samples") or 0)
            truth_rate = float(truth.get("ri_rate") or 0.0)
            valid_rate = float(meta.get("valid_wind_pred_rate") or 0.0)

            precision = pred.get("precision")
            recall = pred.get("recall")
            f1 = pred.get("f1")
            precision = float(precision) if precision is not None else None
            recall = float(recall) if recall is not None else None
            f1 = float(f1) if f1 is not None else None

            wind_ri = wind.get("ri")
            wind_non = wind.get("non_ri")
            wind_ri = float(wind_ri) if wind_ri is not None else None
            wind_non = float(wind_non) if wind_non is not None else None

            rows.append(
                RiRow(
                    experiment=exp_id,
                    model=str(model),
                    lead_hours=lead,
                    threshold_kt=thr,
                    samples=samples,
                    truth_ri_rate=truth_rate,
                    valid_wind_pred_rate=valid_rate,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    wind_mae_ri=wind_ri,
                    wind_mae_non_ri=wind_non,
                )
            )
    return rows


def _fmt_opt(x: float | None, *, digits: int = 3) -> str:
    if x is None:
        return ""
    fmt = f"{{:.{digits}f}}"
    return fmt.format(x)


def render_ri_table(rows: list[RiRow]) -> str:
    if not rows:
        return "_No RI metrics available._"
    header = [
        "Experiment",
        "Model",
        "Samples",
        "Truth RI",
        "Valid Wind",
        "Precision",
        "Recall",
        "F1",
        "Wind MAE (RI)",
        "Wind MAE (non-RI)",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.experiment,
                    r.model,
                    str(r.samples),
                    _fmt_opt(r.truth_ri_rate, digits=3),
                    _fmt_opt(r.valid_wind_pred_rate, digits=3),
                    _fmt_opt(r.precision, digits=3),
                    _fmt_opt(r.recall, digits=3),
                    _fmt_opt(r.f1, digits=3),
                    _fmt_opt(r.wind_mae_ri, digits=2),
                    _fmt_opt(r.wind_mae_non_ri, digits=2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def ensure_ri_section(text: str, replacement: str) -> str:
    if RI_START in text and RI_END in text:
        return _replace_between(text, RI_START, RI_END, replacement)

    block = "\n".join(
        [
            "",
            "## Rapid Intensification (RI)",
            "",
            RI_START,
            replacement.rstrip(),
            RI_END,
            "",
        ]
    )
    if RESULTS_END in text:
        return text.replace(RESULTS_END, RESULTS_END + block, 1)
    return text.rstrip() + block + "\n"


def _rel_md_path(from_dir: Path, to_path: Path) -> str:
    rel = os.path.relpath(str(to_path.resolve()), str(from_dir.resolve()))
    return Path(rel).as_posix()


def update_paper_text(existing: str, metrics_summary: dict[str, Any], *, run_id: str, run_dir: Path, paper_path: Path) -> str:
    rows = rows_from_metrics_summary(metrics_summary)
    table = render_results_table(rows)
    metrics_rel = _rel_md_path(paper_path.parent, run_dir / "metrics.json")
    results_replacement = "\n".join(
        [
            f"_Latest auto-run: `{run_id}` (see `{metrics_rel}`)._",
            "",
            table,
            "",
        ]
    )
    text = _replace_between(existing, RESULTS_START, RESULTS_END, results_replacement)

    ri_rows = ri_rows_from_metrics_summary(metrics_summary)
    if ri_rows:
        thr = ri_rows[0].threshold_kt
        lead = ri_rows[0].lead_hours
        ri_table = render_ri_table(ri_rows)
        ri_replacement = "\n".join(
            [
                f"_RI definition: ΔV ≥ {thr:.0f} kt over {lead}h; classification derived from predicted wind._",
                "",
                ri_table,
                "",
            ]
        )
    else:
        ri_replacement = "_No RI metrics available._\n"

    return ensure_ri_section(text, ri_replacement)


def update_paper_results(paper_path: Path, metrics_summary: dict[str, Any], *, run_id: str, run_dir: Path) -> None:
    ensure_paper_template(paper_path)
    text = paper_path.read_text(encoding="utf-8", errors="replace")
    updated = update_paper_text(text, metrics_summary, run_id=run_id, run_dir=run_dir, paper_path=paper_path)
    paper_path.write_text(updated, encoding="utf-8")


def append_runs_index(
    runs_md_path: Path, *, run_id: str, goal: str, run_dir: Path, repo_root: Path | None = None, best: dict[str, Any] | None = None
) -> None:
    ensure_runs_index(runs_md_path)
    with runs_md_path.open("a", encoding="utf-8") as f:
        f.write(runs_index_entry(run_id=run_id, goal=goal, run_dir=run_dir, repo_root=repo_root, best=best))


def runs_index_entry(*, run_id: str, goal: str, run_dir: Path, repo_root: Path | None = None, best: dict[str, Any] | None = None) -> str:
    ts = utc_now()
    best_str = ""
    if best:
        try:
            best_str = " best=" + json.dumps(best, ensure_ascii=False)
        except Exception:
            best_str = ""
    run_display = str(run_dir)
    if repo_root:
        try:
            run_display = str(run_dir.resolve().relative_to(repo_root.resolve()))
        except Exception:
            run_display = str(run_dir)
    return f"- {ts} `{run_id}` — {goal} (`{run_display}`){best_str}\n"
