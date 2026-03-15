#!/usr/bin/env python3
"""Batch Aurora rollout aligned to payload times and output JSONL forecasts."""

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


def extract_tc(
    surf_vars: dict,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    prev_center: tuple[float, float] | None,
    center_search_radius_km: float,
    radius_km: float,
    to_knots: bool,
    lon_180: bool,
) -> dict:
    msl = surf_vars["msl"][0, 0].cpu().numpy()
    u10 = surf_vars["10u"][0, 0].cpu().numpy()
    v10 = surf_vars["10v"][0, 0].cpu().numpy()

    lat, lon = align_lat_lon(lat, lon, msl.shape)

    # Center = minimum MSLP (optionally constrained near previous center to avoid jumping to other lows)
    if center_search_radius_km and center_search_radius_km > 0 and prev_center is not None:
        prev_lat, prev_lon = prev_center
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        dist = haversine_km(lat_grid, lon_grid, float(prev_lat), float(prev_lon))
        mask = dist <= center_search_radius_km
        if mask.any():
            masked = np.where(mask, msl, np.inf)
            idx = np.unravel_index(int(np.argmin(masked)), msl.shape)
        else:
            idx = np.unravel_index(int(np.argmin(msl)), msl.shape)
    else:
        idx = np.unravel_index(int(np.argmin(msl)), msl.shape)
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

    return {"lat": lat0, "lon": lon_out, "wind": wind_max, "msl_min": float(msl[idx])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Aurora rollout aligned to payloads.")
    parser.add_argument("--static-path", type=Path, required=True)
    parser.add_argument("--surf-files", type=str, required=True)
    parser.add_argument("--atmos-files", type=str, required=True)
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, default=Path("results/era5_dorian/aurora_preds.jsonl"))
    parser.add_argument("--lead-hours", type=str, default="24,48,72")
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument(
        "--center-search-radius-km",
        type=float,
        default=800.0,
        help="Min-MSLP search radius around previous center (km).",
    )
    parser.add_argument("--radius-km", type=float, default=500.0)
    parser.add_argument("--to-knots", action="store_true")
    parser.add_argument("--lon-180", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    surf_files = [Path(p.strip()) for p in args.surf_files.split(",") if p.strip()]
    atmos_files = [Path(p.strip()) for p in args.atmos_files.split(",") if p.strip()]

    static_vars_ds = xr.open_dataset(args.static_path, engine="netcdf4")
    if len(surf_files) == 1:
        surf_vars_ds = xr.open_dataset(surf_files[0], engine="netcdf4")
    else:
        surf_parts = [xr.open_dataset(p, engine="netcdf4") for p in surf_files]
        surf_vars_ds = xr.concat(surf_parts, dim="valid_time").sortby("valid_time")
    if len(atmos_files) == 1:
        atmos_vars_ds = xr.open_dataset(atmos_files[0], engine="netcdf4")
    else:
        atmos_parts = [xr.open_dataset(p, engine="netcdf4") for p in atmos_files]
        atmos_vars_ds = xr.concat(atmos_parts, dim="valid_time").sortby("valid_time")

    leads = [int(x.strip()) for x in args.lead_hours.split(",") if x.strip()]
    step_indices = {h: int(round(h / args.step_hours)) - 1 for h in leads}
    steps_needed = max(step_indices.values()) + 1

    model = AuroraSmall().to(args.device)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")

    # crop to multiples of patch size
    patch = int(model.patch_size)
    lat_len = surf_vars_ds.dims["latitude"]
    lon_len = surf_vars_ds.dims["longitude"]
    new_lat = (lat_len // patch) * patch
    new_lon = (lon_len // patch) * patch
    if new_lat != lat_len or new_lon != lon_len:
        surf_vars_ds = surf_vars_ds.isel(latitude=slice(0, new_lat), longitude=slice(0, new_lon))
        atmos_vars_ds = atmos_vars_ds.isel(latitude=slice(0, new_lat), longitude=slice(0, new_lon))

    static_z = static_vars_ds["z"].values[0]
    static_slt = static_vars_ds["slt"].values[0]
    static_lsm = static_vars_ds["lsm"].values[0]
    if static_z.shape[0] != new_lat or static_z.shape[1] != new_lon:
        static_z = static_z[:new_lat, :new_lon]
        static_slt = static_slt[:new_lat, :new_lon]
        static_lsm = static_lsm[:new_lat, :new_lon]

    lat = surf_vars_ds.latitude.values
    lon = surf_vars_ds.longitude.values
    if (lon < 0).any():
        lon = (lon + 360.0) % 360.0

    valid_times = surf_vars_ds.valid_time.values
    time_to_idx = {np.datetime64(t): i for i, t in enumerate(valid_times)}

    preds_out = []
    payload_lines = args.payloads.read_text().strip().splitlines()
    if args.limit:
        payload_lines = payload_lines[: args.limit]

    for line in payload_lines:
        obj = json.loads(line)
        storm = obj.get("storm", {}) or {}
        # Use real timestamp for alignment when payloads are anonymized.
        t_str = storm.get("valid_time") or storm.get("time")
        if not t_str:
            continue
        t = np.datetime64(t_str)
        idx = time_to_idx.get(t)
        if idx is None or idx < 1:
            continue

        batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(surf_vars_ds["t2m"].isel(valid_time=[idx - 1, idx]).values[None]),
                "10u": torch.from_numpy(surf_vars_ds["u10"].isel(valid_time=[idx - 1, idx]).values[None]),
                "10v": torch.from_numpy(surf_vars_ds["v10"].isel(valid_time=[idx - 1, idx]).values[None]),
                "msl": torch.from_numpy(surf_vars_ds["msl"].isel(valid_time=[idx - 1, idx]).values[None]),
            },
            static_vars={
                "z": torch.from_numpy(static_z),
                "slt": torch.from_numpy(static_slt),
                "lsm": torch.from_numpy(static_lsm),
            },
            atmos_vars={
                "t": torch.from_numpy(atmos_vars_ds["t"].isel(valid_time=[idx - 1, idx]).values[None]),
                "u": torch.from_numpy(atmos_vars_ds["u"].isel(valid_time=[idx - 1, idx]).values[None]),
                "v": torch.from_numpy(atmos_vars_ds["v"].isel(valid_time=[idx - 1, idx]).values[None]),
                "q": torch.from_numpy(atmos_vars_ds["q"].isel(valid_time=[idx - 1, idx]).values[None]),
                "z": torch.from_numpy(atmos_vars_ds["z"].isel(valid_time=[idx - 1, idx]).values[None]),
            },
            metadata=Metadata(
                lat=torch.from_numpy(lat),
                lon=torch.from_numpy(lon),
                time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[idx],),
                atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
            ),
        )

        with torch.inference_mode():
            preds = [pred.to("cpu") for pred in rollout(model, batch, steps=steps_needed)]

        # Track sequentially across rollout steps to avoid global-min jumps.
        try:
            prev_center = (float(storm.get("lat")), float(storm.get("lon")))
        except Exception:
            prev_center = None
        forecast_map: dict[int, dict] = {}
        for step in range(steps_needed):
            tc = extract_tc(
                preds[step].surf_vars,
                lat,
                lon,
                prev_center=prev_center,
                center_search_radius_km=args.center_search_radius_km,
                radius_km=args.radius_km,
                to_knots=args.to_knots,
                lon_180=args.lon_180,
            )
            prev_center = (tc["lat"], tc["lon"])
            for h, step_idx in step_indices.items():
                if step == step_idx:
                    forecast_map[h] = tc

        forecast = [{"lead_hours": h, "lat": forecast_map[h]["lat"], "lon": forecast_map[h]["lon"], "wind": forecast_map[h]["wind"]} for h in leads]

        preds_out.append({"forecast": forecast})

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in preds_out:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(preds_out)} predictions to {args.out_jsonl}")


if __name__ == "__main__":
    main()
