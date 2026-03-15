#!/usr/bin/env python3
"""
Build LLM payloads + truth JSONL from simulated hurricane long time series CSVs.

Example:
  python src/sim_adapter.py \
    --csv /path/to/hurricane_long_time_series.csv \
    --out-dir sim_outputs \
    --history-hours 48 \
    --lead-hours 24,48,72 \
    --sample-minutes 60 \
    --stride-hours 6
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'pandas'. Install with `pip install -r requirements.txt`."
    ) from exc


def parse_leads(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def lon_360_to_180(lon: float) -> float:
    return ((lon + 180) % 360) - 180


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def compute_motion(prev_row, curr_row, dt_hours: float) -> str:
    try:
        lat1, lon1 = float(prev_row["latstore"]), float(prev_row["longstore"])
        lat2, lon2 = float(curr_row["latstore"]), float(curr_row["longstore"])
        if dt_hours <= 0:
            return "unknown"
        # simple bearing
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


def summarize_history(hist: pd.DataFrame, total_hours: float) -> str:
    start = hist.iloc[0]
    end = hist.iloc[-1]
    try:
        lat1, lon1 = float(start["latstore"]), float(start["longstore"])
        lat2, lon2 = float(end["latstore"]), float(end["longstore"])
        wind1, wind2 = float(start["vstore"]), float(end["vstore"])
        dist_km = haversine_km(lat1, lon1, lat2, lon2)
        bearing = compute_motion(start, end, total_hours)
        return (
            f"Start: lat {lat1:.2f}, lon {lon1:.2f}, wind {wind1:.1f}\n"
            f"End:   lat {lat2:.2f}, lon {lon2:.2f}, wind {wind2:.1f}\n"
            f"Delta: dlat {lat2-lat1:.2f}, dlon {lon2-lon1:.2f}, dwind {wind2-wind1:.1f}\n"
            f"48h displacement: {dist_km:.1f} km, mean motion: {bearing}"
        )
    except Exception:
        return "Summary unavailable"


def detect_base_minutes(times: pd.Series) -> int:
    deltas = times.diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return 0
    return int(round(deltas.median()))


def segment_events(df: pd.DataFrame) -> list[tuple[int, int]]:
    active = (df["vstore"] > 0).to_numpy()
    segments: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(df) - 1))
    return segments


def build_history_lines(hist: pd.DataFrame, step: int = 1) -> str:
    lines = []
    if step < 1:
        step = 1
    for _, row in hist.iloc[::step].iterrows():
        t_str = row["datetime"].strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"{t_str}Z: lat {row['latstore']:.2f}, lon {row['longstore']:.2f}, "
            f"wind {row['vstore']:.1f}"
        )
    return "\n".join(lines)


def pick_env(df: pd.DataFrame, cols: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    last = df.iloc[-1]
    for c in cols:
        if c in df.columns:
            try:
                out[c] = float(last[c])
            except Exception:
                pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM payloads from simulated CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to hurricane_long_time_series.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("sim_outputs"))
    parser.add_argument("--history-hours", type=int, default=48)
    parser.add_argument("--lead-hours", type=str, default="24,48,72")
    parser.add_argument("--sample-minutes", type=int, default=60)
    parser.add_argument("--stride-hours", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storm-id", type=str, default="")
    parser.add_argument("--guidance-step", type=int, default=1,
                        help="Subsample history lines in guidance (1 = keep all).")
    parser.add_argument("--guidance-mode", type=str, default="full",
                        choices=["full", "summary"],
                        help="Guidance content: full history lines or compact summary.")
    args = parser.parse_args()

    leads = parse_leads(args.lead_hours)
    if not leads:
        raise SystemExit("lead-hours must include at least one integer (e.g., 24,48,72).")

    df = pd.read_csv(args.csv, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    base_minutes = detect_base_minutes(df["datetime"])
    if base_minutes > 0 and args.sample_minutes % base_minutes == 0:
        factor = max(1, int(round(args.sample_minutes / base_minutes)))
        df = df.iloc[::factor].reset_index(drop=True)
    else:
        df = (
            df.set_index("datetime")
            .resample(f"{args.sample_minutes}min")
            .mean(numeric_only=True)
            .dropna()
            .reset_index()
        )

    if "longstore" in df.columns and df["longstore"].max() > 180:
        df["longstore"] = df["longstore"].apply(lon_360_to_180)

    segments = segment_events(df)

    history_steps = int(round((args.history_hours * 60) / args.sample_minutes))
    lead_steps = {h: int(round((h * 60) / args.sample_minutes)) for h in leads}
    max_lead_steps = max(lead_steps.values())
    stride_steps = max(1, int(round((args.stride_hours * 60) / args.sample_minutes)))

    env_cols = [
        "shearstore",
        "u850store",
        "v850store",
        "T600store",
        "rhstore",
        "pstore",
        "vpstore",
    ]

    rng = pd.Series(range(sum(end - start + 1 for start, end in segments))).sample(
        frac=1, random_state=args.seed
    )
    _ = rng  # deterministic shuffle seed; payloads are still produced sequentially

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "payloads.jsonl"
    truth_path = out_dir / "truth.jsonl"
    meta_path = out_dir / "meta.json"

    storm_id = args.storm_id or args.csv.parent.name
    n_written = 0

    with payload_path.open("w", encoding="utf-8") as f_payload, truth_path.open("w", encoding="utf-8") as f_truth:
        for seg_start, seg_end in segments:
            seg = df.iloc[seg_start:seg_end + 1].reset_index(drop=True)
            if len(seg) < history_steps + max_lead_steps + 1:
                continue
            for i in range(history_steps - 1, len(seg) - max_lead_steps):
                if (i - (history_steps - 1)) % stride_steps != 0:
                    continue
                hist = seg.iloc[i - history_steps + 1:i + 1]
                curr = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) >= 2 else curr

                motion = compute_motion(prev, curr, args.sample_minutes / 60.0)

                if args.guidance_mode == "summary":
                    guidance_text = summarize_history(hist, args.history_hours)
                else:
                    guidance_text = (
                        f"Past {args.history_hours}h (every {args.sample_minutes} min), "
                        f"guidance_step={args.guidance_step}:\n"
                        + build_history_lines(hist, step=args.guidance_step)
                    )
                guidance = [guidance_text]
                payload = {
                    "storm": {
                        "id": storm_id,
                        "basin": "Simulated",
                        "time": str(curr["datetime"]),
                        "lat": float(curr["latstore"]),
                        "lon": float(curr["longstore"]),
                        "wind": float(curr["vstore"]),
                        "pressure": float(curr.get("pstore", "nan")),
                        "motion": motion,
                    },
                    "environment": pick_env(hist, env_cols),
                    "large_scale": {},
                    "analogs": [],
                    "guidance": guidance,
                }

                truth = {}
                for h, steps in lead_steps.items():
                    tgt = seg.iloc[i + steps]
                    truth[f"lat_{h}h"] = float(tgt["latstore"])
                    truth[f"lon_{h}h"] = float(tgt["longstore"])
                    truth[f"wind_{h}h"] = float(tgt["vstore"])

                f_payload.write(json.dumps(payload) + "\n")
                f_truth.write(json.dumps(truth) + "\n")
                n_written += 1
                if n_written >= args.max_samples:
                    break
            if n_written >= args.max_samples:
                break

    meta = {
        "csv": str(args.csv),
        "storm_id": storm_id,
        "history_hours": args.history_hours,
        "lead_hours": leads,
        "sample_minutes": args.sample_minutes,
        "stride_hours": args.stride_hours,
        "samples_written": n_written,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {n_written} samples to {out_dir}")


if __name__ == "__main__":
    main()
