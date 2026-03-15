#!/usr/bin/env python3
"""Run Aurora rollout on ERA5 data and extract TC proxy forecasts.

This is a *pipeline demo* to go from gridded Aurora output to
(lat, lon, wind) at specified lead hours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from aurora import AuroraSmall, Batch, Metadata, rollout


def lon_wrap_diff(lon: np.ndarray, lon0: float) -> np.ndarray:
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def haversine_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    r = 6371.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    dlat = lat_rad - lat0_rad
    dlon = np.deg2rad(lon_wrap_diff(lon, lon0))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return r * c


def lon_360_to_180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def align_lat_lon(lat: np.ndarray, lon: np.ndarray, msl_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if lat.shape[0] != msl_shape[0]:
        if lat.shape[0] == msl_shape[0] + 1:
            lat = lat[: msl_shape[0]]
        elif lat.shape[0] == msl_shape[0] - 1:
            lat = np.pad(lat, (0, 1), mode="edge")
        else:
            raise ValueError(f"lat length {lat.shape[0]} does not match msl shape {msl_shape[0]}")
    if lon.shape[0] != msl_shape[1]:
        if lon.shape[0] == msl_shape[1] + 1:
            lon = lon[: msl_shape[1]]
        elif lon.shape[0] == msl_shape[1] - 1:
            lon = np.pad(lon, (0, 1), mode="edge")
        else:
            raise ValueError(f"lon length {lon.shape[0]} does not match msl shape {msl_shape[1]}")
    return lat, lon


def extract_tc_from_surf(
    surf_vars: dict,
    lat: np.ndarray,
    lon: np.ndarray,
    radius_km: float,
    to_knots: bool,
    lon_180: bool,
) -> dict:
    msl = surf_vars["msl"][0, 0].cpu().numpy()
    u10 = surf_vars["10u"][0, 0].cpu().numpy()
    v10 = surf_vars["10v"][0, 0].cpu().numpy()

    lat, lon = align_lat_lon(lat, lon, msl.shape)

    idx = np.unravel_index(np.argmin(msl), msl.shape)
    lat0 = float(lat[idx[0]])
    lon0 = float(lon[idx[1]])

    wind = np.sqrt(u10 ** 2 + v10 ** 2)
    if radius_km and radius_km > 0:
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        dist = haversine_km(lat_grid, lon_grid, lat0, lon0)
        mask = dist <= radius_km
        wind_max = float(wind[mask].max()) if mask.any() else float(wind.max())
    else:
        wind_max = float(wind.max())

    if to_knots:
        wind_max *= 1.943844

    lon_out = lon_360_to_180(lon0) if lon_180 else lon0

    return {
        "lat": float(lat0),
        "lon": float(lon_out),
        "wind": float(wind_max),
        "msl_min": float(msl[idx]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurora rollout -> TC proxy JSONL.")
    parser.add_argument("--download-dir", type=Path, default=Path("results/aurora_data"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/aurora/forecast.jsonl"))
    parser.add_argument("--time-index", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--step-hours", type=int, default=6, help="Hours per rollout step.")
    parser.add_argument("--lead-hours", type=str, default="24,48,72")
    parser.add_argument("--radius-km", type=float, default=500.0)
    parser.add_argument("--to-knots", action="store_true")
    parser.add_argument("--lon-180", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    static_path = args.download_dir / "static.nc"
    surf_path = args.download_dir / "2023-01-01-surface-level.nc"
    atmos_path = args.download_dir / "2023-01-01-atmospheric.nc"
    if not (static_path.exists() and surf_path.exists() and atmos_path.exists()):
        raise SystemExit("ERA5 files missing; run src/aurora_era5_smoke.py first to download.")

    static_vars_ds = xr.open_dataset(static_path, engine="netcdf4")
    surf_vars_ds = xr.open_dataset(surf_path, engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(atmos_path, engine="netcdf4")

    lat = surf_vars_ds.latitude.values
    lon = surf_vars_ds.longitude.values

    i = args.time_index
    batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i - 1, i]][None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i - 1, i]][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i - 1, i]][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i - 1, i]][None]),
        },
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[[i - 1, i]][None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[[i - 1, i]][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[[i - 1, i]][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[[i - 1, i]][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[[i - 1, i]][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(lat),
            lon=torch.from_numpy(lon),
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    )

    model = AuroraSmall().to(args.device)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")

    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=args.steps)]

    lead_hours = [int(x.strip()) for x in args.lead_hours.split(",") if x.strip()]
    step_indices = {h: int(round(h / args.step_hours)) - 1 for h in lead_hours}

    forecast = []
    for h in lead_hours:
        idx = step_indices[h]
        if idx < 0 or idx >= len(preds):
            raise SystemExit(f"Lead {h}h requires step index {idx}, but only {len(preds)} steps were run.")
        tc = extract_tc_from_surf(
            preds[idx].surf_vars,
            lat,
            lon,
            radius_km=args.radius_km,
            to_knots=args.to_knots,
            lon_180=args.lon_180,
        )
        forecast.append({"lead_hours": h, "lat": tc["lat"], "lon": tc["lon"], "wind": tc["wind"]})

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text(json.dumps({"forecast": forecast}) + "\n", encoding="utf-8")
    print("Wrote", args.out_jsonl)


if __name__ == "__main__":
    main()
