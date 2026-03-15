#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def summarize(dist_km: list[float], wind_err: list[float]) -> dict:
    if not dist_km:
        return {"samples": 0}
    dist_sorted = sorted(dist_km)
    wind_sorted = sorted(wind_err)
    n = len(dist_km)
    return {
        "samples": n,
        "track_mae_km": {
            "mean": sum(dist_km) / n,
            "median": dist_sorted[n // 2],
        },
        "wind_mae_kt": {
            "mean": sum(wind_err) / n,
            "median": wind_sorted[n // 2],
        },
    }


def eval_split(df: pd.DataFrame) -> dict:
    dist = []
    for a, b, c, d in zip(df["last_lat"], df["last_lon"], df["target_lat"], df["target_lon"]):
        if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in [a, b, c, d]):
            continue
        dist.append(haversine_km(float(a), float(b), float(c), float(d)))

    wind = []
    for a, b in zip(df["last_wind"], df["target_wind"]):
        if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in [a, b]):
            continue
        wind.append(abs(float(a) - float(b)))
    return summarize(dist, wind)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistence baseline for HURDAT2 toy parquet.")
    parser.add_argument("--parquet", type=Path, default=Path("hurdat2_llm_toy/all_samples.parquet"))
    parser.add_argument("--out-json", type=Path, default=Path("hurdat2_persistence_metrics.json"))
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if "season" not in df.columns:
        raise SystemExit("parquet missing 'season' column")

    train_years = list(range(1980, 2015))
    val_years = list(range(2015, 2019))
    test_years = list(range(2019, 2023))

    out = {
        "train": eval_split(df[df["season"].isin(train_years)]),
        "val": eval_split(df[df["season"].isin(val_years)]),
        "test": eval_split(df[df["season"].isin(test_years)]),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    for split in ["val", "test"]:
        m = out[split]
        print(
            f"{split}: samples={m.get('samples', 0)} "
            f"track_mae_km_mean={m.get('track_mae_km', {}).get('mean', 0):.1f} "
            f"wind_mae_kt_mean={m.get('wind_mae_kt', {}).get('mean', 0):.1f}"
        )


if __name__ == "__main__":
    main()
