#!/usr/bin/env python3
"""
Hybrid RI Ensemble: Combine LLM base forecast with logistic RI classifier.

Strategy:
  1. Use LLM model for base wind forecast (good at non-RI cases)
  2. Use logistic model's RI probability to flag potential RI events
  3. When P(RI) > threshold, boost the predicted wind by an RI correction

The correction is: if logistic says RI likely, set predicted wind change to at
least the RI threshold (30kt), weighted by the RI probability.

Inputs:
  - LLM predictions JSONL (wind forecasts)
  - Logistic predictions JSONL (wind forecasts with RI probability embedded)
  - Payloads JSONL (for wind_0)
  - Truth JSONL (for evaluation)

Output:
  - Hybrid predictions JSONL
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def safe_float(x, default=None):
    try:
        v = float(x)
        if v != v:  # NaN check
            return default
        return v
    except (TypeError, ValueError):
        return default


def extract_wind_pred(obj, lead=24):
    """Extract predicted wind at given lead from a prediction object."""
    # Handle wrapped format
    if isinstance(obj, dict) and "parsed" in obj:
        obj = obj["parsed"]
    if not isinstance(obj, dict):
        return None

    # Format 1: forecast list
    if "forecast" in obj and isinstance(obj["forecast"], list):
        for item in obj["forecast"]:
            if isinstance(item, dict) and item.get("lead_hours") == lead:
                return safe_float(item.get("wind"))

    # Format 2: flat keys
    key = f"wind_{lead}h"
    if key in obj:
        return safe_float(obj[key])

    return None


def main():
    parser = argparse.ArgumentParser(description="Hybrid LLM + Logistic RI Ensemble")
    parser.add_argument("--llm-predictions", type=Path, required=True,
                        help="LLM model predictions JSONL")
    parser.add_argument("--logit-predictions", type=Path, required=True,
                        help="Logistic baseline predictions JSONL")
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--lead-hours", type=int, default=24)
    parser.add_argument("--ri-threshold-kt", type=float, default=30.0)
    parser.add_argument("--ri-prob-threshold", type=float, default=0.15,
                        help="P(RI) threshold to trigger correction")
    parser.add_argument("--blend-weight", type=float, default=0.7,
                        help="Weight for RI correction (0=LLM only, 1=full correction)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    llm_lines = args.llm_predictions.read_text().strip().splitlines()
    logit_lines = args.logit_predictions.read_text().strip().splitlines()
    payload_lines = args.payloads.read_text().strip().splitlines()
    truth_lines = args.truth.read_text().strip().splitlines()

    n = min(len(llm_lines), len(logit_lines), len(payload_lines), len(truth_lines))
    lead = args.lead_hours
    ri_thr = args.ri_threshold_kt
    prob_thr = args.ri_prob_threshold
    blend_w = args.blend_weight

    results = []
    stats = {"n": 0, "n_corrected": 0, "corrections": []}

    for i in range(n):
        payload = json.loads(payload_lines[i])
        truth = json.loads(truth_lines[i])
        llm_pred = json.loads(llm_lines[i])
        logit_pred = json.loads(logit_lines[i])

        storm = payload.get("storm", {}) if isinstance(payload, dict) else {}
        w0 = safe_float(storm.get("wind"))
        if w0 is None:
            results.append(llm_pred)
            continue

        # Get LLM wind prediction
        llm_wind = extract_wind_pred(llm_pred, lead)
        # Get logistic wind prediction and RI probability
        logit_wind = extract_wind_pred(logit_pred, lead)

        # Extract RI probability from logistic predictions
        # The logistic model embeds ri_prob in its output
        logit_obj = logit_pred.get("parsed", logit_pred) if isinstance(logit_pred, dict) else {}
        ri_prob = safe_float(logit_obj.get("ri_prob"))

        # Get truth for logging
        truth_key = f"wind_{lead}h"
        true_wind = safe_float(truth.get(truth_key)) if isinstance(truth, dict) else None

        stats["n"] += 1

        if llm_wind is None:
            # If LLM didn't produce a forecast, fall back to logistic
            hybrid_wind = logit_wind
            results.append(logit_pred)
            continue

        hybrid_wind = llm_wind  # default: use LLM prediction

        # Apply RI correction if logistic model flags RI
        if ri_prob is not None and ri_prob > prob_thr:
            # Expected RI wind = w0 + ri_threshold
            ri_target = w0 + ri_thr
            # Blend: move LLM prediction toward RI target proportional to P(RI) and blend weight
            correction = blend_w * ri_prob * (ri_target - llm_wind)
            hybrid_wind = llm_wind + correction
            stats["n_corrected"] += 1
            stats["corrections"].append({
                "idx": i, "w0": w0, "llm_wind": llm_wind,
                "ri_prob": ri_prob, "correction": correction,
                "hybrid_wind": hybrid_wind,
                "true_wind": true_wind,
            })

        # Build output in standard forecast format
        out_obj = {
            "forecast": [
                {"lead_hours": lead, "wind": round(hybrid_wind, 2)}
            ]
        }
        results.append(out_obj)

    # Write predictions
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    print(f"Hybrid ensemble: {stats['n']} samples, {stats['n_corrected']} RI-corrected")
    print(f"  P(RI) threshold: {prob_thr}, blend weight: {blend_w}")
    if stats["corrections"]:
        avg_correction = sum(c["correction"] for c in stats["corrections"]) / len(stats["corrections"])
        print(f"  Avg correction magnitude: {avg_correction:.1f} kt")

    # Write stats
    stats_path = args.out.with_suffix(".stats.json")
    # Remove large corrections list for clean output
    stats_summary = {k: v for k, v in stats.items() if k != "corrections"}
    stats_summary["avg_correction"] = (
        sum(c["correction"] for c in stats["corrections"]) / len(stats["corrections"])
        if stats["corrections"] else 0.0
    )
    stats_summary["corrections_sample"] = stats["corrections"][:10]
    stats_path.write_text(json.dumps(stats_summary, indent=2))
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
