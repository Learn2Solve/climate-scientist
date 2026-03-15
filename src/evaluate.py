#!/usr/bin/env python3
"""
Lightweight evaluator for DeepSeek hurricane forecasts.

- Validates the response JSON schema (24/48/72h entries, numeric ranges).
- Optionally computes errors vs. provided ground truth for 24h (lat/lon/wind).

Usage:
  python src/evaluate.py --response path/to/output.json [--truth truth.json]

The response file should contain the JSON emitted by src/forecaster.py.
The truth file (optional) should contain keys: lat_24h, lon_24h, wind_24h.
"""

from __future__ import annotations

import argparse
import json
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, List


def great_circle_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance."""
    R = 6371.0
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_response(resp: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if "forecast" not in resp or not isinstance(resp["forecast"], list):
        errors.append("Missing or invalid 'forecast' list.")
        return errors

    leads_seen = set()
    for idx, entry in enumerate(resp["forecast"]):
        if not isinstance(entry, dict):
            errors.append(f"forecast[{idx}] is not an object.")
            continue
        lead = entry.get("lead_hours")
        lat = entry.get("lat")
        lon = entry.get("lon")
        wind = entry.get("wind")
        if lead not in {24, 48, 72}:
            errors.append(f"forecast[{idx}] invalid lead_hours={lead}")
        else:
            leads_seen.add(lead)
        if not isinstance(lat, (int, float)) or not -90 <= lat <= 90:
            errors.append(f"forecast[{idx}] invalid lat={lat}")
        if not isinstance(lon, (int, float)) or not -180 <= lon <= 180:
            errors.append(f"forecast[{idx}] invalid lon={lon}")
        if not isinstance(wind, (int, float)) or wind <= 0:
            errors.append(f"forecast[{idx}] invalid wind={wind}")

    missing = {24, 48, 72} - leads_seen
    if missing:
        errors.append(f"Missing lead_hours entries: {sorted(missing)}")
    return errors


def evaluate_truth(resp: Dict[str, Any], truth: Dict[str, Any]) -> Dict[str, float]:
    """Compute simple errors for 24h lead if truth provided."""
    out: Dict[str, float] = {}
    forecast_map = {f["lead_hours"]: f for f in resp.get("forecast", []) if isinstance(f, dict)}
    if 24 not in forecast_map:
        return out
    fc = forecast_map[24]
    try:
        dist_km = great_circle_distance_km(float(fc["lat"]), float(fc["lon"]),
                                           float(truth["lat_24h"]), float(truth["lon_24h"]))
        wind_err = abs(float(fc["wind"]) - float(truth["wind_24h"]))
    except Exception:
        return out
    out["track_mae_km_24h"] = dist_km
    out["wind_mae_kt_24h"] = wind_err
    return out


def main():
    parser = argparse.ArgumentParser(description="Validate and optionally score a DeepSeek forecast JSON.")
    parser.add_argument("--response", required=True, type=Path, help="Path to model output JSON.")
    parser.add_argument("--truth", type=Path, help="Optional path to truth JSON with lat_24h, lon_24h, wind_24h.")
    args = parser.parse_args()

    resp = load_json(args.response)
    errors = validate_response(resp)
    if errors:
        print("VALIDATION: FAIL")
        for e in errors:
            print(f"- {e}")
    else:
        print("VALIDATION: PASS")

    if args.truth and args.truth.exists():
        truth = load_json(args.truth)
        scores = evaluate_truth(resp, truth)
        if scores:
            print("SCORES:")
            for k, v in scores.items():
                print(f"- {k}: {v:.2f}")
        else:
            print("SCORES: could not compute (missing keys?)")


if __name__ == "__main__":
    main()
