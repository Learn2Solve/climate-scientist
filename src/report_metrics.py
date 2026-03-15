#!/usr/bin/env python3
"""
Aggregate MAE metrics across multiple prediction files and emit CSV/JSON/Markdown.
Usage:
  uv run python src/report_metrics.py \
    --truth sim_outputs/truth.jsonl \
    --model "DeepSeek-chat:sim_outputs/predictions_200.jsonl" \
    --model "TTM-aligned:sim_outputs_ttm_aligned/predictions.jsonl" \
    --model "Claude-Opus-4.5:sim_outputs_agent/predictions_claude_200.jsonl" \
    --model "Codex-gpt-5.2-codex:sim_outputs_agent/predictions_codex_200.jsonl" \
    --out-csv results/metrics.csv \
    --out-json results/metrics.json \
    --out-md results/metrics.md
"""

from __future__ import annotations

import argparse
import json
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def parse_pred(obj: dict) -> dict[int, dict]:
    if "forecast" in obj and isinstance(obj["forecast"], list):
        out = {}
        for item in obj["forecast"]:
            if not isinstance(item, dict):
                continue
            lead = item.get("lead_hours")
            if lead is None:
                continue
            out[int(lead)] = item
        return out
    out = {}
    for key in list(obj.keys()):
        if key.startswith("lat_") and key.endswith("h"):
            h = int(key.replace("lat_", "").replace("h", ""))
            out.setdefault(h, {})["lat"] = obj.get(key)
        if key.startswith("lon_") and key.endswith("h"):
            h = int(key.replace("lon_", "").replace("h", ""))
            out.setdefault(h, {})["lon"] = obj.get(key)
        if key.startswith("wind_") and key.endswith("h"):
            h = int(key.replace("wind_", "").replace("h", ""))
            out.setdefault(h, {})["wind"] = obj.get(key)
    return out


def parse_truth(obj: dict) -> dict[int, dict]:
    out = {}
    for key in list(obj.keys()):
        if key.startswith("lat_") and key.endswith("h"):
            h = int(key.replace("lat_", "").replace("h", ""))
            out.setdefault(h, {})["lat"] = obj.get(key)
        if key.startswith("lon_") and key.endswith("h"):
            h = int(key.replace("lon_", "").replace("h", ""))
            out.setdefault(h, {})["lon"] = obj.get(key)
        if key.startswith("wind_") and key.endswith("h"):
            h = int(key.replace("wind_", "").replace("h", ""))
            out.setdefault(h, {})["wind"] = obj.get(key)
    return out


def eval_file(pred_path: Path, truth_path: Path) -> dict:
    preds = pred_path.read_text(encoding="utf-8").strip().splitlines()
    trues = truth_path.read_text(encoding="utf-8").strip().splitlines()
    n = min(len(preds), len(trues))

    track_err: dict[int, list[float]] = {}
    wind_err: dict[int, list[float]] = {}
    valid = 0

    for i in range(n):
        pred_raw = json.loads(preds[i])
        pred_obj = pred_raw.get("parsed") if isinstance(pred_raw, dict) else None
        if pred_obj is None:
            pred_obj = pred_raw
        truth_obj = json.loads(trues[i])

        pred_map = parse_pred(pred_obj)
        truth_map = parse_truth(truth_obj)

        if pred_map:
            valid += 1
        for h, truth in truth_map.items():
            if h not in pred_map:
                continue
            try:
                lat_p = float(pred_map[h]["lat"])
                lon_p = float(pred_map[h]["lon"])
                wind_p = float(pred_map[h]["wind"])
                lat_t = float(truth["lat"])
                lon_t = float(truth["lon"])
                wind_t = float(truth["wind"])
            except Exception:
                continue
            track_err.setdefault(h, []).append(haversine_km(lat_p, lon_p, lat_t, lon_t))
            wind_err.setdefault(h, []).append(abs(wind_p - wind_t))

    metrics = {
        "samples": n,
        "valid_json_rate": valid / n if n else 0.0,
        "track": {},
        "wind": {},
    }

    for h in sorted(track_err.keys()):
        te = track_err[h]
        we = wind_err.get(h, [])
        if not te:
            continue
        te_sorted = sorted(te)
        we_sorted = sorted(we) if we else []
        metrics["track"][h] = {
            "mean": sum(te) / len(te),
            "median": te_sorted[len(te_sorted) // 2],
        }
        if we:
            metrics["wind"][h] = {
                "mean": sum(we) / len(we),
                "median": we_sorted[len(we_sorted) // 2],
            }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MAE metrics for multiple models.")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="label:path to predictions.jsonl (repeatable)",
    )
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    rows = []
    summary = {}
    for item in args.model:
        if ":" not in item:
            raise SystemExit(f"--model expects label:path, got '{item}'")
        label, path_str = item.split(":", 1)
        pred_path = Path(path_str)
        metrics = eval_file(pred_path, args.truth)
        summary[label] = {"predictions": str(pred_path), **metrics}
        for h in sorted(metrics["track"].keys()):
            track = metrics["track"].get(h, {})
            wind = metrics["wind"].get(h, {})
            rows.append(
                {
                    "model": label,
                    "lead_hours": h,
                    "track_mae_km_mean": track.get("mean"),
                    "track_mae_km_median": track.get("median"),
                    "wind_mae_mean": wind.get("mean"),
                    "wind_mae_median": wind.get("median"),
                    "samples": metrics["samples"],
                    "valid_json_rate": metrics["valid_json_rate"],
                }
            )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "model",
            "lead_hours",
            "track_mae_km_mean",
            "track_mae_km_median",
            "wind_mae_mean",
            "wind_mae_median",
            "samples",
            "valid_json_rate",
        ]
        lines = [",".join(header)]
        for row in rows:
            lines.append(
                ",".join(
                    [
                        str(row.get(col, ""))
                        for col in header
                    ]
                )
            )
        args.out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "Model",
            "Lead (h)",
            "Track MAE (km, mean)",
            "Track MAE (km, median)",
            "Wind MAE (kt, mean)",
            "Wind MAE (kt, median)",
            "Samples",
            "Valid JSON",
        ]
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("model", "")),
                        str(row.get("lead_hours", "")),
                        f"{row.get('track_mae_km_mean', 0):.2f}",
                        f"{row.get('track_mae_km_median', 0):.2f}",
                        f"{row.get('wind_mae_mean', 0):.2f}",
                        f"{row.get('wind_mae_median', 0):.2f}",
                        str(row.get("samples", "")),
                        f"{row.get('valid_json_rate', 0):.3f}",
                    ]
                )
                + " |"
            )
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # also print to stdout for quick check
    for row in rows:
        print(
            f"{row['model']} lead {row['lead_hours']}h: "
            f"track={row['track_mae_km_mean']:.2f} km, "
            f"wind={row['wind_mae_mean']:.2f} kt"
        )


if __name__ == "__main__":
    main()
