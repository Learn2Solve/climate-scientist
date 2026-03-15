#!/usr/bin/env python3
"""
Headless RI research summarizer.

Reads the latest shortlist/harvest JSON in docs/papers/ri and writes a concise
research memo with:
  - Forecasting/modeling papers
  - Mechanistic/precursor papers
  - Concrete hypotheses + next experiments for this repo
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def find_latest(pattern: str, root: Path) -> Path | None:
    files = sorted(root.glob(pattern))
    return files[-1] if files else None


def load_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("results", []) if isinstance(data, dict) else []


def score_keywords(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for k in keywords if k in t)


def classify(entry: dict[str, Any]) -> tuple[int, int]:
    text = " ".join(
        [
            str(entry.get("title") or ""),
            str(entry.get("abstract") or ""),
            str(entry.get("venue") or ""),
        ]
    ).lower()
    forecast_kw = [
        "forecast",
        "prediction",
        "predict",
        "probabilistic",
        "probability",
        "machine learning",
        "deep learning",
        "model",
        "intensity forecast",
        "rapid intensification",
    ]
    mech_kw = [
        "ocean heat content",
        "salinity",
        "eyewall",
        "inner-core",
        "structure",
        "microphysics",
        "convection",
        "satellite",
        "heat flux",
        "upper-ocean",
    ]
    return score_keywords(text, forecast_kw), score_keywords(text, mech_kw)


def tidy(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RI literature into a research memo.")
    parser.add_argument("--ri-dir", type=Path, default=Path("docs/papers/ri"))
    parser.add_argument("--input", type=Path, default=None, help="Explicit shortlist/harvest JSON.")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    ri_dir = args.ri_dir
    ts = utc_compact()

    input_path = args.input
    if input_path is None:
        input_path = find_latest("shortlist_*.json", ri_dir) or find_latest("harvest_*.json", ri_dir)
    if input_path is None or not input_path.exists():
        raise SystemExit("No shortlist/harvest JSON found under docs/papers/ri/")

    entries = load_entries(input_path)

    ranked = []
    for e in entries:
        forecast_score, mech_score = classify(e)
        cited = e.get("cited_by_count") or 0
        year = e.get("publication_year") or 0
        ranked.append(
            {
                "entry": e,
                "forecast_score": int(forecast_score),
                "mech_score": int(mech_score),
                "cited": int(cited),
                "year": int(year),
            }
        )

    forecast_ranked = sorted(
        ranked,
        key=lambda x: (x["forecast_score"], x["year"], x["cited"]),
        reverse=True,
    )
    mech_ranked = sorted(
        ranked,
        key=lambda x: (x["mech_score"], x["year"], x["cited"]),
        reverse=True,
    )

    top_n = int(args.top_n)
    forecast_top = [r["entry"] for r in forecast_ranked if r["forecast_score"] > 0][:top_n]
    mech_top = [r["entry"] for r in mech_ranked if r["mech_score"] > 0][:top_n]

    lines = []
    lines.append(f"# RI Research Memo ({ts})")
    lines.append("")
    lines.append(f"- source: `{input_path}`")
    lines.append("")

    lines.append("## Forecasting / Modeling (priority)")
    if not forecast_top:
        lines.append("_No clear forecasting papers found in this shortlist._")
    else:
        for e in forecast_top:
            title = tidy(str(e.get("title") or ""))
            lines.append(f"### {title}")
            if e.get("publication_date"):
                lines.append(f"- date: `{e['publication_date']}`")
            if e.get("venue"):
                lines.append(f"- venue: {e['venue']}")
            if e.get("doi"):
                lines.append(f"- doi: {e['doi']}")
            if e.get("landing_page_url"):
                lines.append(f"- landing_page: {e['landing_page_url']}")
            if e.get("pdf_url"):
                lines.append(f"- pdf_url: {e['pdf_url']}")
            if e.get("pdf_path"):
                lines.append(f"- downloaded_pdf: `{e['pdf_path']}`")
            if e.get("abstract"):
                lines.append("")
                lines.append("**Abstract (OpenAlex)**")
                lines.append(tidy(str(e.get("abstract"))))
            lines.append("")

    lines.append("## Mechanistic / Precursor signals (secondary)")
    if not mech_top:
        lines.append("_No clear mechanism papers found in this shortlist._")
    else:
        for e in mech_top:
            title = tidy(str(e.get("title") or ""))
            lines.append(f"### {title}")
            if e.get("publication_date"):
                lines.append(f"- date: `{e['publication_date']}`")
            if e.get("venue"):
                lines.append(f"- venue: {e['venue']}")
            if e.get("doi"):
                lines.append(f"- doi: {e['doi']}")
            if e.get("landing_page_url"):
                lines.append(f"- landing_page: {e['landing_page_url']}")
            lines.append("")

    # Hypotheses + experiments grounded in current repo features
    lines.append("## Hypotheses for this repo (actionable)")
    lines.append("1) **RI probability + calibrated thresholds**: Use `ri_logit_baseline.py` probabilities and apply a calibration/thresholding step (match RI rate or maximize F1) to improve RI F1 without harming wind MAE.")
    lines.append("2) **Intensity-change features matter**: Use 6h/24h wind change from guidance history as explicit predictors; test ablation (with/without wind-change features).")
    lines.append("3) **Environment gating**: Compare RI performance when gating on environment-only vs environment+history features to quantify marginal gains.")
    lines.append("")

    lines.append("## Next experiments (1–2 hours)")
    lines.append("- Add an ablation run: `ri_logit` with/without `dwind_6h` + `dwind_24h` (same split).")
    lines.append("- Add simple calibration: map predicted RI probability to match truth RI rate on a held-out split.")
    lines.append("- Report RI precision/recall/F1 + RI wind MAE in `docs/PAPER.md`.")
    lines.append("")

    out_path = ri_dir / f"agent_review_{ts}.md"
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    # Append a short pointer to DISCUSSION.md
    discussion = Path("docs/DISCUSSION.md")
    if discussion.exists():
        note = []
        note.append("\n## RI literature update\n")
        note.append(f"- Added memo: `{out_path}`\n")
        note.append("- Actionable hypotheses: calibrated RI probability, wind-change features, environment gating.\n")
        discussion.write_text(discussion.read_text(encoding="utf-8") + "".join(note), encoding="utf-8")


if __name__ == "__main__":
    main()

