#!/usr/bin/env python3
"""
Rapid Intensification (RI) metrics for JSONL forecasts.

Inputs are line-aligned:
  - payloads.jsonl: must contain storm.wind at t0
  - truth.jsonl: must contain wind_{lead}h (e.g., wind_24h)
  - predictions.jsonl: each line can be either:
      * {"forecast": [{"lead_hours": 24, "wind": ...}, ...], ...}
      * a dict with flattened keys like wind_24h
      * a wrapper dict with {"parsed": <forecast_json>} (as produced by run_forecaster_jsonl.py)

Definition (default): RI if wind_lead - wind_0h >= threshold_kt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_pred(obj: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if "forecast" in obj and isinstance(obj["forecast"], list):
        out: dict[int, dict[str, Any]] = {}
        for item in obj["forecast"]:
            if not isinstance(item, dict):
                continue
            lead = item.get("lead_hours")
            if lead is None:
                continue
            out[int(lead)] = item
        return out

    out: dict[int, dict[str, Any]] = {}
    for key in list(obj.keys()):
        if key.startswith("wind_") and key.endswith("h"):
            h = int(key.replace("wind_", "").replace("h", ""))
            out.setdefault(h, {})["wind"] = obj.get(key)
    return out


def parse_truth(obj: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for key in list(obj.keys()):
        if key.startswith("wind_") and key.endswith("h"):
            h = int(key.replace("wind_", "").replace("h", ""))
            out.setdefault(h, {})["wind"] = obj.get(key)
    return out


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RI classification metrics from payloads/truth/predictions JSONL.")
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--lead-hours", type=int, default=24)
    parser.add_argument("--threshold-kt", type=float, default=30.0)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    payload_lines = args.payloads.read_text(encoding="utf-8").strip().splitlines()
    truth_lines = args.truth.read_text(encoding="utf-8").strip().splitlines()
    pred_lines = args.predictions.read_text(encoding="utf-8").strip().splitlines()
    n = min(len(payload_lines), len(truth_lines), len(pred_lines))

    lead = int(args.lead_hours)
    thr = float(args.threshold_kt)

    used = 0
    valid_wind = 0

    tp = fp = fn = tn = 0
    n_ri_true = 0

    wind_err_all: list[float] = []
    wind_err_ri: list[float] = []
    wind_err_non_ri: list[float] = []

    for i in range(n):
        payload = json.loads(payload_lines[i])
        truth_obj = json.loads(truth_lines[i])
        pred_raw = json.loads(pred_lines[i])

        storm = payload.get("storm", {}) if isinstance(payload, dict) else {}
        w0 = _safe_float(storm.get("wind"))
        if w0 is None:
            continue

        truth_map = parse_truth(truth_obj if isinstance(truth_obj, dict) else {})
        wt = _safe_float((truth_map.get(lead) or {}).get("wind"))
        if wt is None:
            continue

        pred_obj: Any = None
        if isinstance(pred_raw, dict):
            pred_obj = pred_raw.get("parsed")
        if pred_obj is None:
            pred_obj = pred_raw

        pred_map = parse_pred(pred_obj if isinstance(pred_obj, dict) else {})
        wp = _safe_float((pred_map.get(lead) or {}).get("wind"))
        if wp is not None:
            valid_wind += 1
            wind_err = abs(wp - wt)
            wind_err_all.append(wind_err)
        else:
            # Missing prediction counts as a "no-RI" call for classification purposes.
            wp = None

        used += 1

        ri_true = (wt - w0) >= thr
        if ri_true:
            n_ri_true += 1

        ri_pred = False
        if wp is not None:
            ri_pred = (wp - w0) >= thr

        if ri_true and ri_pred:
            tp += 1
        elif (not ri_true) and ri_pred:
            fp += 1
        elif ri_true and (not ri_pred):
            fn += 1
        else:
            tn += 1

        if wp is not None:
            if ri_true:
                wind_err_ri.append(abs(wp - wt))
            else:
                wind_err_non_ri.append(abs(wp - wt))

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision is not None and recall is not None and (precision + recall) > 0) else None
    accuracy = (tp + tn) / used if used else None

    def mean(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    out = {
        "meta": {
            "lead_hours": lead,
            "threshold_kt": thr,
            "samples": used,
            "valid_wind_pred_rate": (valid_wind / used) if used else 0.0,
        },
        "truth": {
            "ri_samples": n_ri_true,
            "ri_rate": (n_ri_true / used) if used else 0.0,
        },
        "pred": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        },
        "wind_mae": {
            "all": mean(wind_err_all),
            "ri": mean(wind_err_ri),
            "non_ri": mean(wind_err_non_ri),
            "n_valid": len(wind_err_all),
            "n_valid_ri": len(wind_err_ri),
            "n_valid_non_ri": len(wind_err_non_ri),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

