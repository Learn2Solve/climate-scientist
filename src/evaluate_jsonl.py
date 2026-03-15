#!/usr/bin/env python3
"""
Evaluate JSONL predictions against JSONL truth (same line order).
Supports two prediction schemas:
  1) {"forecast": [{"lead_hours": 24, "lat": .., "lon": .., "wind": ..}, ...]}
  2) {"lat_24h": .., "lon_24h": .., "wind_24h": .., ...}
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
    # flattened keys
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL predictions vs truth.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--per-sample-json", action="store_true",
                        help="Output per-sample errors as JSON (for anomaly analysis)")
    args = parser.parse_args()

    preds = args.predictions.read_text(encoding="utf-8").strip().splitlines()
    trues = args.truth.read_text(encoding="utf-8").strip().splitlines()
    n = min(len(preds), len(trues))

    track_err = {}
    wind_err = {}
    valid = 0
    per_sample = []  # for --per-sample-json

    for i in range(n):
        pred_raw = json.loads(preds[i])
        # allow wrapper with "parsed"
        pred_obj = pred_raw.get("parsed") if isinstance(pred_raw, dict) else None
        if pred_obj is None:
            pred_obj = pred_raw
        truth_obj = json.loads(trues[i])

        pred_map = parse_pred(pred_obj)
        truth_map = parse_truth(truth_obj)

        sample_record: dict = {"index": i}

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
            t_err = haversine_km(lat_p, lon_p, lat_t, lon_t)
            w_err = abs(wind_p - wind_t)
            track_err.setdefault(h, []).append(t_err)
            wind_err.setdefault(h, []).append(w_err)

            sample_record[f"track_error_{h}h"] = round(t_err, 2)
            sample_record[f"wind_error_{h}h"] = round(w_err, 2)
            sample_record[f"wind_predicted_{h}h"] = round(wind_p, 2)
            sample_record[f"wind_truth_{h}h"] = round(wind_t, 2)
            sample_record[f"lat_truth_{h}h"] = round(lat_t, 2)
            sample_record[f"lon_truth_{h}h"] = round(lon_t, 2)

        # Copy useful fields from truth for anomaly analysis
        for key in ("lat_0h", "lon_0h", "wind_0h", "lat", "lon",
                     "initial_wind", "wind0", "dwind_24h", "delta_wind_24h"):
            if key in truth_obj:
                sample_record[key] = truth_obj[key]

        per_sample.append(sample_record)

    if args.per_sample_json:
        # Build summary + per_sample output
        summary: dict = {"samples": n, "valid_json_rate": round(valid / n, 3) if n else 0}
        for h in sorted(track_err.keys()):
            te = track_err[h]
            we = wind_err.get(h, [])
            if te:
                summary[f"track_mae_{h}h_mean"] = round(sum(te) / len(te), 2)
            if we:
                summary[f"wind_mae_{h}h_mean"] = round(sum(we) / len(we), 2)

        print(json.dumps({"summary": summary, "per_sample": per_sample}))
        return

    print(f"samples: {n}")
    print(f"valid_json_rate: {valid / n:.3f}")
    for h in sorted(track_err.keys()):
        te = track_err[h]
        we = wind_err.get(h, [])
        if te:
            te_sorted = sorted(te)
            we_sorted = sorted(we) if we else []
            te_mean = sum(te) / len(te)
            te_med = te_sorted[len(te_sorted) // 2]
            print(f"lead {h}h track_mae_km: mean={te_mean:.2f} median={te_med:.2f}")
            if we:
                we_mean = sum(we) / len(we)
                we_med = we_sorted[len(we_sorted) // 2]
                print(f"lead {h}h wind_mae: mean={we_mean:.2f} median={we_med:.2f}")


if __name__ == "__main__":
    main()
