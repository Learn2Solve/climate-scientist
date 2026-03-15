#!/usr/bin/env python3
"""Extract a simple TC proxy track from ERA5 surface files.

Uses min MSLP for center and max 10m wind within radius for intensity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TC proxy track from ERA5 surface files.")
    parser.add_argument("--surf-files", type=str, required=True, help="Comma-separated surface nc files")
    parser.add_argument(
        "--center-search-radius-km",
        type=float,
        default=0.0,
        help="If >0, pick min MSLP within this radius of previous center (or init center).",
    )
    parser.add_argument("--init-lat", type=float, default=None)
    parser.add_argument("--init-lon", type=float, default=None)
    parser.add_argument("--radius-km", type=float, default=500.0)
    parser.add_argument("--to-knots", action="store_true")
    parser.add_argument("--lon-180", action="store_true")
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/era5_dorian/track.jsonl"))
    args = parser.parse_args()

    surf_files = [Path(p.strip()) for p in args.surf_files.split(",") if p.strip()]
    if len(surf_files) == 1:
        ds = xr.open_dataset(surf_files[0], engine="netcdf4")
    else:
        parts = [xr.open_dataset(p, engine="netcdf4") for p in surf_files]
        ds = xr.concat(parts, dim="valid_time")
        ds = ds.sortby("valid_time")

    lat = ds.latitude.values
    lon = ds.longitude.values

    out = []
    prev_lat = args.init_lat
    prev_lon = args.init_lon
    for t in range(ds.dims["valid_time"]):
        msl = ds["msl"].isel(valid_time=t).values
        u10 = ds["u10"].isel(valid_time=t).values
        v10 = ds["v10"].isel(valid_time=t).values

        # center = min MSLP (global, or local around previous center)
        if args.center_search_radius_km and args.center_search_radius_km > 0 and prev_lat is not None and prev_lon is not None:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            dist = haversine_km(lat_grid, lon_grid, float(prev_lat), float(prev_lon))
            mask = dist <= args.center_search_radius_km
            if mask.any():
                masked = np.where(mask, msl, np.inf)
                idx = np.unravel_index(int(np.argmin(masked)), msl.shape)
            else:
                idx = np.unravel_index(int(np.argmin(msl)), msl.shape)
        else:
            idx = np.unravel_index(int(np.argmin(msl)), msl.shape)
        lat0 = float(lat[idx[0]])
        lon0 = float(lon[idx[1]])
        prev_lat, prev_lon = lat0, lon0

        wind = np.sqrt(u10 ** 2 + v10 ** 2)
        if args.radius_km and args.radius_km > 0:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            dist = haversine_km(lat_grid, lon_grid, lat0, lon0)
            mask = dist <= args.radius_km
            wind_max = float(wind[mask].max()) if mask.any() else float(wind.max())
        else:
            wind_max = float(wind.max())

        if args.to_knots:
            wind_max *= 1.943844

        lon_out = lon_360_to_180(lon0) if args.lon_180 else lon0

        out.append(
            {
                "time": str(ds.valid_time.values[t]),
                "lat": lat0,
                "lon": lon_out,
                "wind": wind_max,
                "msl_min": float(msl[idx]),
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print("Wrote", args.out_jsonl)


if __name__ == "__main__":
    main()
