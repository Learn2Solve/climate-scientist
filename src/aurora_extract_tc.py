#!/usr/bin/env python3
"""Extract a simple TC proxy (min MSLP + max 10m wind) from Aurora surf grids.

Example:
  uv run --with xarray --with netcdf4 python src/aurora_extract_tc.py \
    --pred-pt results/aurora/pred_surf_step0.pt \
    --latlon-from results/aurora_data/2023-01-01-surface-level.nc \
    --radius-km 500 \
    --to-knots \
    --out-json results/aurora/track_step0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr


def lon_wrap_diff(lon: np.ndarray, lon0: float) -> np.ndarray:
    """Shortest lon difference in degrees (handles 0..360 wrap)."""
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def haversine_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Vectorized great-circle distance (km)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TC proxy from Aurora surf vars.")
    parser.add_argument("--pred-pt", type=Path, required=True)
    parser.add_argument("--latlon-from", type=Path, required=True)
    parser.add_argument("--radius-km", type=float, default=0.0, help="Max wind within radius (km). 0 = global max.")
    parser.add_argument("--to-knots", action="store_true", help="Convert wind from m/s to knots.")
    parser.add_argument("--lon-180", action="store_true", help="Convert lon to [-180, 180].")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    pred = torch.load(args.pred_pt, map_location="cpu")
    msl = pred["msl"][0, 0].numpy()
    u10 = pred["10u"][0, 0].numpy()
    v10 = pred["10v"][0, 0].numpy()

    ds = xr.open_dataset(args.latlon_from)
    lat = ds.latitude.values
    lon = ds.longitude.values
    # Align lat/lon to prediction grid if needed (Aurora output may drop one latitude row)
    if lat.shape[0] != msl.shape[0]:
        if lat.shape[0] == msl.shape[0] + 1:
            lat = lat[: msl.shape[0]]
        elif lat.shape[0] == msl.shape[0] - 1:
            lat = np.pad(lat, (0, 1), mode="edge")
        else:
            raise ValueError(f"lat length {lat.shape[0]} does not match msl shape {msl.shape[0]}")
    if lon.shape[0] != msl.shape[1]:
        if lon.shape[0] == msl.shape[1] + 1:
            lon = lon[: msl.shape[1]]
        elif lon.shape[0] == msl.shape[1] - 1:
            lon = np.pad(lon, (0, 1), mode="edge")
        else:
            raise ValueError(f"lon length {lon.shape[0]} does not match msl shape {msl.shape[1]}")

    # find min MSLP center
    idx = np.unravel_index(np.argmin(msl), msl.shape)
    lat0 = float(lat[idx[0]])
    lon0 = float(lon[idx[1]])

    wind = np.sqrt(u10 ** 2 + v10 ** 2)

    if args.radius_km and args.radius_km > 0:
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        dist = haversine_km(lat_grid, lon_grid, lat0, lon0)
        mask = dist <= args.radius_km
        if mask.any():
            wind_max = float(wind[mask].max())
        else:
            wind_max = float(wind.max())
    else:
        wind_max = float(wind.max())

    if args.to_knots:
        wind_max = wind_max * 1.943844

    lon_out = lon_360_to_180(lon0) if args.lon_180 else lon0

    out = {
        "lat": float(lat0),
        "lon": float(lon_out),
        "wind": float(wind_max),
        "msl_min": float(msl[idx]),
        "units": {
            "wind": "kt" if args.to_knots else "m/s",
            "msl": "Pa",
        },
        "radius_km": float(args.radius_km) if args.radius_km else 0.0,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
