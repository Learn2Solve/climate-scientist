#!/usr/bin/env python3
"""
Aligned TTM baseline: match LLM payload times and output predictions in the same order.

This script:
- Loads simulated hurricane CSV
- Resamples to the same cadence used for LLM payloads
- For each payload timestamp, builds a context window (pads with zeros if needed)
- Runs zero-shot TTM (univariate) for lat/lon/wind separately
- Writes predictions JSONL aligned to payload order
- Optionally writes a filtered truth JSONL aligned to the same indices
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from tsfm_public import TimeSeriesPreprocessor, TinyTimeMixerForPrediction


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


def load_payload_times(path: Path, start: int, limit: int) -> tuple[list[pd.Timestamp], list[int]]:
    times: list[pd.Timestamp] = []
    idxs: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit and len(times) >= limit:
                break
            obj = json.loads(line)
            storm = obj.get("storm", {})
            t = storm.get("time")
            if not t:
                continue
            times.append(pd.to_datetime(t))
            idxs.append(i)
    return times, idxs


def build_scaled_series(df: pd.DataFrame, column: str, scaling: bool) -> tuple[np.ndarray, object | None]:
    df_var = df[["datetime", column]].copy()
    if not scaling:
        return df_var[column].to_numpy(dtype=np.float32), None

    tsp = TimeSeriesPreprocessor(
        timestamp_column="datetime",
        id_columns=[],
        target_columns=[column],
        context_length=1,
        prediction_length=1,
        scaling=True,
        encode_categorical=False,
    )
    tsp.train(df_var)
    df_scaled = tsp.preprocess(df_var)
    scaler = next(iter(tsp.target_scaler_dict.values()))
    return df_scaled[column].to_numpy(dtype=np.float32), scaler


def build_context_arrays(
    series: np.ndarray,
    indices: list[int],
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(indices)
    past = np.zeros((n, context_length, 1), dtype=np.float32)
    mask = np.zeros((n, context_length, 1), dtype=np.float32)
    for i, idx in enumerate(indices):
        start = idx - context_length + 1
        if start < 0:
            pad = -start
            slice_ = series[0 : idx + 1]
            past[i, pad:, 0] = slice_
            mask[i, pad:, 0] = 1.0
        else:
            slice_ = series[start : idx + 1]
            past[i, :, 0] = slice_
            mask[i, :, 0] = 1.0
    return past, mask


def run_model(
    model: TinyTimeMixerForPrediction,
    past: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    preds = []
    model.eval()
    for i in range(0, past.shape[0], batch_size):
        pv = torch.from_numpy(past[i : i + batch_size]).to(device)
        pm = torch.from_numpy(mask[i : i + batch_size]).to(device)
        with torch.no_grad():
            out = model(past_values=pv, past_observed_mask=pm, return_loss=False)
        pred = out.prediction_outputs.cpu().numpy()  # [bs, pred_len, 1]
        preds.append(pred[:, :, 0])
    return np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligned TTM baseline vs LLM payloads.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=None, help="Optional truth JSONL to align.")
    parser.add_argument("--out-dir", type=Path, default=Path("sim_outputs_ttm_aligned"))
    parser.add_argument("--sample-minutes", type=int, default=60)
    parser.add_argument("--lead-hours", type=str, default="24,48,72")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--prediction-length", type=int, default=0)
    args = parser.parse_args()

    leads = [int(x.strip()) for x in args.lead_hours.split(",") if x.strip()]
    if not leads:
        raise SystemExit("lead-hours must include at least one integer (e.g., 24,48,72).")

    payload_times, payload_idxs = load_payload_times(args.payloads, args.start, args.limit)
    if not payload_times:
        raise SystemExit("No payload times found. Check --payloads/--start/--limit.")

    df = pd.read_csv(args.csv, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if "longstore" in df.columns and df["longstore"].max() > 180:
        df["longstore"] = df["longstore"].apply(lon_360_to_180)
    df = resample_df(df, args.sample_minutes)

    index_map = pd.Series(df.index.to_numpy(), index=df["datetime"]).to_dict()
    valid_indices: list[int] = []
    valid_payload_idxs: list[int] = []
    for t, idx in zip(payload_times, payload_idxs):
        row = index_map.get(t)
        if row is None:
            continue
        valid_indices.append(int(row))
        valid_payload_idxs.append(idx)

    if not valid_indices:
        raise SystemExit("No payload timestamps matched resampled CSV datetimes.")

    device = torch.device(args.device)
    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",
        revision="main",
    ).to(device)

    context_length = args.context_length or int(model.config.context_length)
    prediction_length = args.prediction_length or int(model.config.prediction_length)

    if context_length != int(model.config.context_length):
        print(f"Warning: overriding context_length to model config={model.config.context_length}")
        context_length = int(model.config.context_length)
    if prediction_length != int(model.config.prediction_length):
        print(f"Warning: overriding prediction_length to model config={model.config.prediction_length}")
        prediction_length = int(model.config.prediction_length)

    targets = {
        "latstore": "lat",
        "longstore": "lon",
        "vstore": "wind",
    }

    series_scaled: dict[str, np.ndarray] = {}
    scalers: dict[str, object | None] = {}
    for col in targets:
        series_scaled[col], scalers[col] = build_scaled_series(df, col, scaling=not args.no_scale)

    preds_out: dict[str, np.ndarray] = {}
    for col in targets:
        past, mask = build_context_arrays(series_scaled[col], valid_indices, context_length)
        pred = run_model(model, past, mask, args.batch_size, device)
        scaler = scalers[col]
        if scaler is not None:
            pred = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
        preds_out[col] = pred

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    truth_path = out_dir / "truth.jsonl"

    lead_indices = {h: lead_to_index(h, args.sample_minutes) for h in leads}

    with pred_path.open("w", encoding="utf-8") as f_pred:
        for i in range(len(valid_indices)):
            forecast = []
            for h in leads:
                idx = lead_indices[h]
                forecast.append(
                    {
                        "lead_hours": h,
                        "lat": float(preds_out["latstore"][i, idx]),
                        "lon": float(preds_out["longstore"][i, idx]),
                        "wind": float(preds_out["vstore"][i, idx]),
                    }
                )
            f_pred.write(json.dumps({"forecast": forecast}) + "\n")

    if args.truth:
        lines = args.truth.read_text(encoding="utf-8").strip().splitlines()
        with truth_path.open("w", encoding="utf-8") as f_truth:
            for idx in valid_payload_idxs:
                if idx < len(lines):
                    f_truth.write(lines[idx].strip() + "\n")

    print(f"Wrote {pred_path}")
    if args.truth:
        print(f"Wrote {truth_path}")
    print(f"aligned_samples: {len(valid_indices)} (from {len(payload_times)} payloads)")


if __name__ == "__main__":
    main()
