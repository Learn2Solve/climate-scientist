#!/usr/bin/env python3
"""
Simple baselines for simulated payloads: persistence and kinematic.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def move_great_circle(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    r = 6371.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    d = dist_km / r
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), ((math.degrees(lon2) + 540) % 360) - 180


def parse_motion(text: str) -> tuple[float, float] | None:
    # format: "305 deg at 9.0 kt"
    try:
        parts = text.replace("deg", "").replace("kt", "").split()
        bearing = float(parts[0])
        speed = float(parts[-1])
        return bearing, speed
    except Exception:
        return None


def parse_dt_hours_from_guidance(guidance: list[str]) -> float | None:
    for g in guidance:
        m = re.search(r"every\s+(\d+)\s+min", g)
        if not m:
            continue
        minutes = int(m.group(1))
        if minutes <= 0:
            continue
        return minutes / 60.0
    return None


def extract_winds_from_guidance(guidance: list[str]) -> list[float]:
    winds: list[float] = []
    for g in guidance:
        for m in re.finditer(r"wind\s+(-?\d+(?:\.\d+)?)", g):
            try:
                winds.append(float(m.group(1)))
            except Exception:
                continue
    return winds


def trend_slope_kt_per_h(wind_now: float, winds_hist: list[float], *, dt_hours: float, window_hours: float) -> float:
    if dt_hours <= 0:
        return 0.0
    if len(winds_hist) < 2:
        return 0.0

    steps_back = int(round(window_hours / dt_hours))
    steps_back = max(1, min(steps_back, len(winds_hist) - 1))
    w_prev = winds_hist[-1 - steps_back]
    denom = steps_back * dt_hours
    if denom <= 0:
        return 0.0
    return (wind_now - w_prev) / denom


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    if q <= 0:
        return min(xs)
    if q >= 1:
        return max(xs)
    xs_sorted = sorted(xs)
    idx = int(round(q * (len(xs_sorted) - 1)))
    return xs_sorted[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline forecasts from payload JSONL.")
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("baseline_predictions.jsonl"))
    parser.add_argument("--method", choices=["persistence", "kinematic", "trend", "ri_gate"], default="persistence")
    parser.add_argument("--leads", type=str, default="24,48,72")
    parser.add_argument("--trend-window-hours", type=float, default=6.0, help="Intensity trend window for method=trend.")
    parser.add_argument("--trend-dt-hours", type=float, default=6.0, help="Fallback dt (hours) if guidance lacks cadence.")
    parser.add_argument("--trend-alpha", type=float, default=0.5, help="Damping factor for trend extrapolation.")
    parser.add_argument("--trend-max-abs-slope", type=float, default=3.0, help="Clamp |dV/dt| (kt/h) for method=trend.")
    parser.add_argument("--trend-max-wind", type=float, default=200.0, help="Clamp forecast wind (kt) for method=trend.")
    parser.add_argument("--ri-feature", type=str, default="pstore", help="Environment feature used to gate RI calls (method=ri_gate).")
    parser.add_argument("--ri-direction", choices=["gt", "lt"], default="gt", help="RI call direction for feature threshold.")
    parser.add_argument("--ri-threshold", type=float, default=float("nan"), help="Feature threshold; if NaN, use --ri-quantile.")
    parser.add_argument("--ri-quantile", type=float, default=0.75, help="Quantile to pick threshold when --ri-threshold is NaN.")
    parser.add_argument("--ri-delta24", type=float, default=39.2, help="Additive wind delta (kt) at 24h when RI is predicted.")
    parser.add_argument("--ri-delta48", type=float, default=13.7, help="Additive wind delta (kt) at 48h when RI is predicted.")
    parser.add_argument("--ri-delta72", type=float, default=0.7, help="Additive wind delta (kt) at 72h when RI is predicted.")
    parser.add_argument("--ri-max-wind", type=float, default=200.0, help="Clamp forecast wind (kt) for method=ri_gate.")
    args = parser.parse_args()

    leads = [int(x.strip()) for x in args.leads.split(",") if x.strip()]

    lines = args.payloads.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines if line.strip()]

    ri_threshold = float(args.ri_threshold)
    if args.method == "ri_gate" and ri_threshold != ri_threshold:
        vals = []
        for payload in payloads:
            env = payload.get("environment", {}) if isinstance(payload, dict) else {}
            if not isinstance(env, dict):
                continue
            v = env.get(args.ri_feature)
            try:
                vals.append(float(v))
            except Exception:
                continue
        ri_threshold = quantile(vals, float(args.ri_quantile))

    ri_deltas = {24: float(args.ri_delta24), 48: float(args.ri_delta48), 72: float(args.ri_delta72)}

    with args.out.open("w", encoding="utf-8") as f_out:
        for payload in payloads:
            storm = payload.get("storm", {})
            lat = float(storm.get("lat"))
            lon = float(storm.get("lon"))
            wind = float(storm.get("wind"))
            motion = storm.get("motion", "")
            guidance = payload.get("guidance", []) or []
            if not isinstance(guidance, list):
                guidance = []

            motion_parsed = None
            if args.method in {"kinematic", "trend"}:
                motion_parsed = parse_motion(motion)

            slope = 0.0
            if args.method == "trend":
                winds_hist = extract_winds_from_guidance(guidance)
                dt_hours = parse_dt_hours_from_guidance(guidance) or float(args.trend_dt_hours)
                slope = trend_slope_kt_per_h(wind, winds_hist, dt_hours=dt_hours, window_hours=float(args.trend_window_hours))
                slope *= float(args.trend_alpha)
                max_abs = abs(float(args.trend_max_abs_slope))
                if max_abs > 0:
                    slope = max(-max_abs, min(max_abs, slope))

            ri_pred = False
            if args.method == "ri_gate":
                env = payload.get("environment", {}) if isinstance(payload, dict) else {}
                if isinstance(env, dict):
                    v = env.get(args.ri_feature)
                    try:
                        feat = float(v)
                        if args.ri_direction == "gt":
                            ri_pred = feat > ri_threshold
                        else:
                            ri_pred = feat < ri_threshold
                    except Exception:
                        ri_pred = False

            forecast = []
            for h in leads:
                if motion_parsed:
                    bearing, speed_kt = motion_parsed
                    dist_km = speed_kt * 1.852 * h
                    lat_h, lon_h = move_great_circle(lat, lon, bearing, dist_km)
                else:
                    lat_h, lon_h = lat, lon
                wind_h = wind
                if args.method == "trend":
                    wind_h = wind + slope * h
                    wind_h = max(0.0, min(float(args.trend_max_wind), wind_h))
                if args.method == "ri_gate":
                    delta = ri_deltas.get(h, ri_deltas.get(24, 0.0)) if ri_pred else 0.0
                    wind_h = max(0.0, min(float(args.ri_max_wind), wind + float(delta)))
                forecast.append({"lead_hours": h, "lat": lat_h, "lon": lon_h, "wind": wind_h})

            f_out.write(json.dumps({"forecast": forecast, "reasoning": args.method}) + "\n")

    print(f"Wrote baseline forecasts to {args.out}")


if __name__ == "__main__":
    main()
