#!/usr/bin/env python3
"""
Zero-shot TinyTimeMixer (TTM) baseline for simulated hurricane time series.

Notes:
- TTM public checkpoints are univariate (num_input_channels=1).
- We run three separate passes (lat, lon, wind) and merge them per sample.
- Default resample is hourly so prediction_length=96 covers 96 hours.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from tsfm_public import TimeSeriesPreprocessor, TinyTimeMixerForPrediction
from tsfm_public.toolkit.dataset import ForecastDFDataset


def detect_base_minutes(times: pd.Series) -> int:
    deltas = times.diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return 0
    return int(round(deltas.median()))


def lon_360_to_180(lon: float) -> float:
    return ((lon + 180) % 360) - 180


def resample_df(df: pd.DataFrame, sample_minutes: int) -> pd.DataFrame:
    base_minutes = detect_base_minutes(df["datetime"])
    if base_minutes > 0 and sample_minutes % base_minutes == 0:
        factor = max(1, int(round(sample_minutes / base_minutes)))
        return df.iloc[::factor].reset_index(drop=True)
    return (
        df.set_index("datetime")
        .resample(f"{sample_minutes}min")
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )


def lead_to_index(lead_hours: int, sample_minutes: int) -> int:
    steps = int(round((lead_hours * 60) / sample_minutes))
    return max(0, steps - 1)


def build_dataset(
    df: pd.DataFrame,
    target: str,
    *,
    context_length: int,
    prediction_length: int,
    stride: int,
    scaling: bool,
) -> tuple[TimeSeriesPreprocessor | None, ForecastDFDataset, object | None]:
    tsp = None
    scaler = None
    if scaling:
        tsp = TimeSeriesPreprocessor(
            timestamp_column="datetime",
            id_columns=[],
            target_columns=[target],
            context_length=context_length,
            prediction_length=prediction_length,
            scaling=True,
            encode_categorical=False,
        )
        tsp.train(df)
        df = tsp.preprocess(df)
        scaler = next(iter(tsp.target_scaler_dict.values()))

    dset = ForecastDFDataset(
        data=df,
        timestamp_column="datetime",
        target_columns=[target],
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
    )
    return tsp, dset, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot TTM baseline on simulated CSV.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("sim_outputs_ttm"))
    parser.add_argument("--sample-minutes", type=int, default=60)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--lead-hours", type=str, default="24,48,72")
    parser.add_argument("--stride-hours", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-scale", action="store_true", help="Disable scaling (faster, but less stable).")
    args = parser.parse_args()

    leads = [int(x.strip()) for x in args.lead_hours.split(",") if x.strip()]
    if not leads:
        raise SystemExit("lead-hours must include at least one integer (e.g., 24,48,72).")

    df = pd.read_csv(args.csv, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if "longstore" in df.columns and df["longstore"].max() > 180:
        df["longstore"] = df["longstore"].apply(lon_360_to_180)
    df = resample_df(df, args.sample_minutes)

    stride_steps = max(1, int(round((args.stride_hours * 60) / args.sample_minutes)))

    targets = {
        "latstore": "lat",
        "longstore": "lon",
        "vstore": "wind",
    }

    datasets = {}
    scalers = {}
    for col in targets:
        _, dset, scaler = build_dataset(
            df,
            col,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            stride=stride_steps,
            scaling=not args.no_scale,
        )
        datasets[col] = dset
        scalers[col] = scaler

    n = min(len(datasets["latstore"]), len(datasets["longstore"]), len(datasets["vstore"]))
    if args.max_samples and args.max_samples > 0:
        n = min(n, args.max_samples)

    device = torch.device(args.device)
    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",
        revision="main",
    ).to(device)
    model.eval()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    truth_path = out_dir / "truth.jsonl"

    lead_indices = {h: lead_to_index(h, args.sample_minutes) for h in leads}

    with pred_path.open("w", encoding="utf-8") as f_pred, truth_path.open("w", encoding="utf-8") as f_truth:
        for i in range(n):
            preds = {}
            trues = {}
            for col in targets:
                sample = datasets[col][i]
                past = sample["past_values"].unsqueeze(0).to(device)
                mask = sample["past_observed_mask"].unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(past_values=past, past_observed_mask=mask, return_loss=False)
                pred_seq = out.prediction_outputs.squeeze(0).cpu().numpy()  # pred_len x 1
                true_seq = sample["future_values"].numpy()

                scaler = scalers[col]
                if scaler is not None:
                    pred_seq = scaler.inverse_transform(pred_seq)
                    true_seq = scaler.inverse_transform(true_seq)

                preds[col] = pred_seq[:, 0]
                trues[col] = true_seq[:, 0]

            forecast = []
            truth = {}
            for h in leads:
                idx = lead_indices[h]
                forecast.append(
                    {
                        "lead_hours": h,
                        "lat": float(preds["latstore"][idx]),
                        "lon": float(preds["longstore"][idx]),
                        "wind": float(preds["vstore"][idx]),
                    }
                )
                truth[f"lat_{h}h"] = float(trues["latstore"][idx])
                truth[f"lon_{h}h"] = float(trues["longstore"][idx])
                truth[f"wind_{h}h"] = float(trues["vstore"][idx])

            f_pred.write(json.dumps({"forecast": forecast}) + "\n")
            f_truth.write(json.dumps(truth) + "\n")

            if (i + 1) % 25 == 0 or i == n - 1:
                print(f"Processed {i + 1}/{n}")

    print(f"Wrote {pred_path} and {truth_path}")


if __name__ == "__main__":
    main()
