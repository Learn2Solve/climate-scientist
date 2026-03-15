#!/usr/bin/env python3
"""Download a small ERA5 subset for an Aug/Sep window (e.g., a hurricane case study).

Defaults:
- Date window: Aug/Sep days (configurable via args)
- Times: 00,06,12,18
- Area: North Atlantic subset (40N, 120W) to (5N, 40W)

Outputs multiple files (Aug + Sep) so we can safely merge later.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def retrieve_single_levels(c, out_path: Path, year: str, month: str, days: list[str], times: list[str], area: list[float]) -> None:
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
            "year": year,
            "month": month,
            "day": days,
            "time": times,
            "area": area,
            "format": "netcdf",
        },
        str(out_path),
    )


def retrieve_pressure_levels(c, out_path: Path, year: str, month: str, days: list[str], times: list[str], area: list[float]) -> None:
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
            "year": year,
            "month": month,
            "day": days,
            "time": times,
            "area": area,
            "format": "netcdf",
        },
        str(out_path),
    )


def retrieve_static(c, out_path: Path, year: str, month: str, day: str, time: str, area: list[float]) -> None:
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["geopotential", "land_sea_mask", "soil_type"],
            "year": year,
            "month": month,
            "day": day,
            "time": time,
            "area": area,
            "format": "netcdf",
        },
        str(out_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an ERA5 subset for an Aug/Sep case window.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/era5_dorian"))
    parser.add_argument("--year", type=str, default="2019", help="Year for the requested window.")
    parser.add_argument("--times", type=str, default="00:00,06:00,12:00,18:00")
    parser.add_argument(
        "--area",
        type=str,
        default="40,-120,5,-40",
        help="north,west,south,east (deg). Use -120,-40 for Atlantic subset.",
    )
    parser.add_argument("--aug-days", type=str, default="26,27,28,29,30,31")
    parser.add_argument("--sep-days", type=str, default="01,02,03,04,05")
    parser.add_argument("--force", action="store_true", help="Redownload even if files exist.")
    args = parser.parse_args()

    # split Aug and Sep to avoid invalid day/month combos
    aug_days = [d.zfill(2) for d in args.aug_days.split(",") if d.strip()]
    sep_days = [d.zfill(2) for d in args.sep_days.split(",") if d.strip()]
    times = [t.strip() for t in args.times.split(",") if t.strip()]
    area = [float(x.strip()) for x in args.area.split(",")]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    static_path = args.out_dir / "static.nc"
    aug_surf = args.out_dir / f"surface_{args.year}_08.nc"
    sep_surf = args.out_dir / f"surface_{args.year}_09.nc"
    aug_atmos = args.out_dir / f"atmos_{args.year}_08.nc"
    sep_atmos = args.out_dir / f"atmos_{args.year}_09.nc"

    import cdsapi

    c = cdsapi.Client()

    if args.force or not static_path.exists():
        retrieve_static(c, static_path, args.year, "08", (aug_days[0] if aug_days else "01"), "00:00", area)

    if args.force or not aug_surf.exists():
        retrieve_single_levels(c, aug_surf, args.year, "08", aug_days, times, area)
    if args.force or not sep_surf.exists():
        retrieve_single_levels(c, sep_surf, args.year, "09", sep_days, times, area)

    if args.force or not aug_atmos.exists():
        retrieve_pressure_levels(c, aug_atmos, args.year, "08", aug_days, times, area)
    if args.force or not sep_atmos.exists():
        retrieve_pressure_levels(c, sep_atmos, args.year, "09", sep_days, times, area)

    print("Download complete:")
    print(static_path)
    print(aug_surf)
    print(sep_surf)
    print(aug_atmos)
    print(sep_atmos)


if __name__ == "__main__":
    main()
