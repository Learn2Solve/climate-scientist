#!/usr/bin/env python3
"""
Build a DeepSeek forecaster payload JSON from one HURDAT2 sample.

This bridges the 48h->24h dataset in hurdat2_llm_toy/all_samples.parquet
to the structured input expected by src/forecaster.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def extract_past_observations(prompt: str) -> str:
    """
    Extract the 'Past observations (every 6 hours)' block from the SFT prompt.
    Falls back to an empty string if the expected markers are not found.
    """
    start_marker = "Past observations (every 6 hours):"
    end_marker = "\n\nTask: Based on the above information"
    if start_marker not in prompt:
        return ""
    try:
        after = prompt.split(start_marker, 1)[1]
        block = after.split(end_marker, 1)[0]
    except Exception:
        return ""
    return block.lstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a HURDAT2 sample row into a DeepSeek payload JSON."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("hurdat2_llm_toy/all_samples.parquet"),
        help="Path to all_samples.parquet.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index (0-based iloc) to use as the sample.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("my_storm_payload.json"),
        help="Output JSON path for the DeepSeek payload.",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"Parquet file not found: {args.parquet}")

    df = pd.read_parquet(args.parquet)
    if not (0 <= args.index < len(df)):
        raise SystemExit(f"index {args.index} out of range [0, {len(df) - 1}]")

    row = df.iloc[args.index]

    past_obs = extract_past_observations(str(row["input"]))

    storm = {
        "id": row.get("storm_id", "UNKNOWN"),
        "basin": "Atlantic",
        "time": row.get("ref_time", ""),
        "lat": float(row.get("last_lat")),
        "lon": float(row.get("last_lon")),
        "wind": float(row.get("last_wind")),
        "pressure": "unknown",  # not present in the toy dataset
        "motion": "unknown",    # not present in the toy dataset
    }

    environment = {
        "Past 48h track/intensity (6-hourly)": past_obs
        or "[past observations block not parsed]",
    }

    large_scale = {
        "Context": (
            "Large-scale pattern not included in this toy dataset; "
            "model should lean mainly on the recent track/intensity history."
        )
    }

    analogs: list[dict[str, str]] = []
    guidance = [
        "Focus on the extracted 48h HURDAT2 track and intensity history.",
        "Produce 24h/48h/72h forecasts consistent with the recent motion and climatology.",
    ]

    payload = {
        "storm": storm,
        "environment": environment,
        "large_scale": large_scale,
        "analogs": analogs,
        "guidance": guidance,
    }

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote payload for index {args.index} to {args.out}")


if __name__ == "__main__":
    main()
