#!/usr/bin/env python3
"""Convert ERA5-derived track JSONL into LLM payloads + truth JSONL."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def compute_motion(prev, curr, dt_hours: float) -> str:
    try:
        lat1, lon1 = float(prev["lat"]), float(prev["lon"])
        lat2, lon2 = float(curr["lat"]), float(curr["lon"])
        if dt_hours <= 0:
            return "unknown"
        dlon = math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - (
            math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)
        )
        bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
        speed_km_h = haversine_km(lat1, lon1, lat2, lon2) / dt_hours
        speed_kt = speed_km_h * 0.539957
        return f"{bearing:.0f} deg at {speed_kt:.1f} kt"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build payloads/truth from ERA5 track JSONL.")
    parser.add_argument("--track-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/era5_dorian_payloads"))
    parser.add_argument("--history-hours", type=int, default=48)
    parser.add_argument("--lead-hours", type=str, default="24,48,72")
    parser.add_argument("--stride-hours", type=int, default=6)
    parser.add_argument("--storm-id", type=str, default="Dorian_2019")
    parser.add_argument("--basin", type=str, default="ATL")
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Hide absolute dates (use relative t-48h..t0) to reduce leakage for LLM baselines.",
    )
    args = parser.parse_args()

    leads = [int(x.strip()) for x in args.lead_hours.split(",") if x.strip()]
    if not leads:
        raise SystemExit("lead-hours must include at least one integer")

    rows = [json.loads(line) for line in args.track_jsonl.read_text().strip().splitlines()]
    if len(rows) < 10:
        raise SystemExit("track too short")

    # infer step hours from first two timestamps
    t0 = datetime.fromisoformat(rows[0]["time"].replace("Z", ""))
    t1 = datetime.fromisoformat(rows[1]["time"].replace("Z", ""))
    step_hours = abs((t1 - t0).total_seconds()) / 3600.0

    history_steps = int(round(args.history_hours / step_hours))
    lead_steps = {h: int(round(h / step_hours)) for h in leads}
    max_lead = max(lead_steps.values())
    stride_steps = max(1, int(round(args.stride_hours / step_hours)))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "payloads.jsonl"
    truth_path = out_dir / "truth.jsonl"

    n_written = 0
    with payload_path.open("w", encoding="utf-8") as f_payload, truth_path.open("w", encoding="utf-8") as f_truth:
        for i in range(history_steps - 1, len(rows) - max_lead):
            if (i - (history_steps - 1)) % stride_steps != 0:
                continue
            hist = rows[i - history_steps + 1 : i + 1]
            curr = rows[i]
            prev = rows[i - 1] if i > 0 else curr

            motion = compute_motion(prev, curr, step_hours)
            history_lines = []
            if args.anonymize:
                for j, h in enumerate(hist):
                    offset_h = (j - (len(hist) - 1)) * step_hours
                    tag = f"t{offset_h:+.0f}h"
                    history_lines.append(
                        f"{tag}: lat {h['lat']:.2f}, lon {h['lon']:.2f}, wind {h['wind']:.1f}"
                    )
                guidance = [f"Past {args.history_hours}h (step {step_hours}h):\n" + "\n".join(history_lines)]
                storm_time = "t0"
            else:
                for h in hist:
                    history_lines.append(
                        f"{h['time']}: lat {h['lat']:.2f}, lon {h['lon']:.2f}, wind {h['wind']:.1f}"
                    )
                guidance = [f"Past {args.history_hours}h (step {step_hours}h):\n" + "\n".join(history_lines)]
                storm_time = curr["time"]

            payload = {
                "storm": {
                    "id": args.storm_id,
                    "basin": args.basin,
                    # Keep the real timestamp for alignment, but optionally hide it from the LLM prompt.
                    "valid_time": curr["time"],
                    "time": storm_time,
                    "lat": float(curr["lat"]),
                    "lon": float(curr["lon"]),
                    "wind": float(curr["wind"]),
                    "pressure": float(curr.get("msl_min", float("nan"))),
                    "motion": motion,
                },
                "environment": {},
                "large_scale": {},
                "analogs": [],
                "guidance": guidance,
            }

            truth = {}
            for h, steps in lead_steps.items():
                tgt = rows[i + steps]
                truth[f"lat_{h}h"] = float(tgt["lat"])
                truth[f"lon_{h}h"] = float(tgt["lon"])
                truth[f"wind_{h}h"] = float(tgt["wind"])

            f_payload.write(json.dumps(payload) + "\n")
            f_truth.write(json.dumps(truth) + "\n")
            n_written += 1

    print(f"Wrote {n_written} samples to {out_dir}")


if __name__ == "__main__":
    main()
