#!/usr/bin/env python3
"""Download ERA5 sample data and run a small Aurora rollout.

This mirrors the official Aurora notebook but keeps outputs minimal.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import xarray as xr

from aurora import AuroraSmall, Batch, Metadata, rollout


def download_era5(download_path: Path) -> None:
    import cdsapi
    download_path.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()

    static_path = download_path / "static.nc"
    if not static_path.exists():
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["geopotential", "land_sea_mask", "soil_type"],
                "year": "2023",
                "month": "01",
                "day": "01",
                "time": "00:00",
                "format": "netcdf",
            },
            str(static_path),
        )

    surf_path = download_path / "2023-01-01-surface-level.nc"
    if not surf_path.exists():
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",
                ],
                "year": "2023",
                "month": "01",
                "day": "01",
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
            },
            str(surf_path),
        )

    atmos_path = download_path / "2023-01-01-atmospheric.nc"
    if not atmos_path.exists():
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "specific_humidity",
                    "geopotential",
                ],
                "pressure_level": [
                    "50",
                    "100",
                    "150",
                    "200",
                    "250",
                    "300",
                    "400",
                    "500",
                    "600",
                    "700",
                    "850",
                    "925",
                    "1000",
                ],
                "year": "2023",
                "month": "01",
                "day": "01",
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
            },
            str(atmos_path),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurora ERA5 smoke test.")
    parser.add_argument("--download-dir", type=Path, default=Path("results/aurora_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/aurora"))
    parser.add_argument("--time-index", type=int, default=1, help="Index into ERA5 time dimension")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    static_path = args.download_dir / "static.nc"
    surf_path = args.download_dir / "2023-01-01-surface-level.nc"
    atmos_path = args.download_dir / "2023-01-01-atmospheric.nc"
    if not (static_path.exists() and surf_path.exists() and atmos_path.exists()):
        download_era5(args.download_dir)

    static_vars_ds = xr.open_dataset(static_path, engine="netcdf4")
    surf_vars_ds = xr.open_dataset(surf_path, engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(atmos_path, engine="netcdf4")

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
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    )

    model = AuroraSmall().to(args.device)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")

    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=args.steps)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "pred_surf_step0.pt"
    torch.save(preds[0].surf_vars, out_path)

    print("Saved surf vars for step 0 to", out_path)
    print("Example var shapes:")
    for k, v in preds[0].surf_vars.items():
        print(k, tuple(v.shape))


if __name__ == "__main__":
    main()
