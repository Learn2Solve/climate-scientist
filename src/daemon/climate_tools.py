"""Climate analysis toolkit - generates and executes sandbox scripts.

Provides structured climate analysis by generating Python scripts
that run inside the sandbox runner. Does not perform computations
directly; instead creates reproducible analysis scripts.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.climate_tools")

# ---------------------------------------------------------------------------
# Script templates
# ---------------------------------------------------------------------------

HURDAT2_PARSE_SCRIPT = textwrap.dedent("""\
    import json
    import re
    from pathlib import Path
    from datetime import datetime, timedelta

    data_path = Path("{data_path}")
    text = data_path.read_text(encoding="utf-8", errors="replace")
    lines = text.strip().splitlines()

    storms = []
    current_storm = None
    remaining = 0

    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if remaining == 0:
            # Header line: storm_id, name, num_entries
            storm_id = parts[0]
            name = parts[1].strip()
            num_entries = int(parts[2])
            current_storm = {{
                "storm_id": storm_id,
                "name": name,
                "records": [],
            }}
            remaining = num_entries
            continue

        # Data line
        date_str = parts[0]
        time_str = parts[1]
        record_id = parts[2]
        status = parts[3]
        lat_str = parts[4]
        lon_str = parts[5]
        max_wind = int(parts[6]) if parts[6] not in ("", "-999") else None
        min_pressure = int(parts[7]) if parts[7] not in ("", "-999") else None

        # Parse lat/lon
        lat_val = float(lat_str[:-1])
        if lat_str.endswith("S"):
            lat_val = -lat_val
        lon_val = float(lon_str[:-1])
        if lon_str.endswith("W"):
            lon_val = -lon_val

        dt_str = date_str + time_str.zfill(4)
        try:
            dt = datetime.strptime(dt_str, "%Y%m%d%H%M")
        except ValueError:
            dt = None

        current_storm["records"].append({{
            "datetime": dt.isoformat() if dt else None,
            "record_id": record_id,
            "status": status,
            "lat": lat_val,
            "lon": lon_val,
            "max_wind_kt": max_wind,
            "min_pressure_mb": min_pressure,
        }})

        remaining -= 1
        if remaining == 0:
            storms.append(current_storm)
            current_storm = None

    # Detect RI events (wind increase >= 30kt in 24h)
    ri_events = 0
    for storm in storms:
        recs = [r for r in storm["records"] if r["max_wind_kt"] is not None and r["datetime"] is not None]
        for i, r1 in enumerate(recs):
            dt1 = datetime.fromisoformat(r1["datetime"])
            for r2 in recs[i + 1:]:
                dt2 = datetime.fromisoformat(r2["datetime"])
                delta_h = (dt2 - dt1).total_seconds() / 3600
                if delta_h > 24:
                    break
                if 20 <= delta_h <= 28:
                    wind_change = r2["max_wind_kt"] - r1["max_wind_kt"]
                    if wind_change >= 30:
                        ri_events += 1
                        break

    # Date range
    all_dates = []
    for s in storms:
        for r in s["records"]:
            if r["datetime"]:
                all_dates.append(r["datetime"])
    all_dates.sort()

    results = {{
        "total_storms": len(storms),
        "total_records": sum(len(s["records"]) for s in storms),
        "date_range": {{
            "start": all_dates[0] if all_dates else None,
            "end": all_dates[-1] if all_dates else None,
        }},
        "ri_events_count": ri_events,
        "storms_summary": [
            {{
                "storm_id": s["storm_id"],
                "name": s["name"],
                "num_records": len(s["records"]),
            }}
            for s in storms[:50]  # first 50 for summary
        ],
    }}

    Path("results.json").write_text(json.dumps(results, indent=2))
    print(f"Parsed {{len(storms)}} storms, {{ri_events}} RI events")
""")


SST_ANOMALY_SCRIPT = textwrap.dedent("""\
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    try:
        data_path = Path("{data_path}")
        suffix = data_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(data_path)
            # Expect columns: date/time, sst or temperature
            temp_cols = [c for c in df.columns if any(
                k in c.lower() for k in ("sst", "temp", "temperature")
            )]
            if not temp_cols:
                raise ValueError(f"No temperature columns found in {{list(df.columns)}}")
            temp_col = temp_cols[0]
            values = df[temp_col].dropna().values
        elif suffix in (".nc", ".netcdf"):
            import xarray as xr
            ds = xr.open_dataset(data_path)
            var_names = [v for v in ds.data_vars if any(
                k in v.lower() for k in ("sst", "temp", "temperature")
            )]
            if not var_names:
                raise ValueError(f"No temperature variables found in {{list(ds.data_vars)}}")
            values = ds[var_names[0]].values.flatten()
            values = values[~np.isnan(values)]
        else:
            raise ValueError(f"Unsupported format: {{suffix}}")

        climatology_mean = float(np.mean(values))
        climatology_std = float(np.std(values))
        anomalies = values - climatology_mean

        results = {{
            "climatology_mean": round(climatology_mean, 4),
            "climatology_std": round(climatology_std, 4),
            "anomaly_mean": round(float(np.mean(anomalies)), 4),
            "anomaly_std": round(float(np.std(anomalies)), 4),
            "anomaly_min": round(float(np.min(anomalies)), 4),
            "anomaly_max": round(float(np.max(anomalies)), 4),
            "n_values": len(values),
        }}
    except Exception as e:
        results = {{"error": str(e)}}

    Path("results.json").write_text(json.dumps(results, indent=2))
""")


RI_DETECTION_SCRIPT = textwrap.dedent("""\
    import json
    from pathlib import Path
    from datetime import datetime

    try:
        data_path = Path("{data_path}")
        storm_data = json.loads(data_path.read_text())

        # Accept either a list of storms or a dict with "storms_summary"/"storms" key
        if isinstance(storm_data, list):
            storms = storm_data
        elif "storms" in storm_data:
            storms = storm_data["storms"]
        else:
            storms = [storm_data]

        ri_threshold_kt = {ri_threshold_kt}
        ri_window_h = {ri_window_h}

        all_ri_events = []
        for storm in storms:
            recs = storm.get("records", [])
            recs = [r for r in recs if r.get("max_wind_kt") is not None and r.get("datetime")]
            recs.sort(key=lambda r: r["datetime"])

            for i, r1 in enumerate(recs):
                dt1 = datetime.fromisoformat(r1["datetime"])
                for r2 in recs[i + 1:]:
                    dt2 = datetime.fromisoformat(r2["datetime"])
                    delta_h = (dt2 - dt1).total_seconds() / 3600
                    if delta_h > ri_window_h + 4:
                        break
                    if (ri_window_h - 4) <= delta_h <= (ri_window_h + 4):
                        wind_change = r2["max_wind_kt"] - r1["max_wind_kt"]
                        if wind_change >= ri_threshold_kt:
                            all_ri_events.append({{
                                "storm_id": storm.get("storm_id", "unknown"),
                                "storm_name": storm.get("name", "unknown"),
                                "start_time": r1["datetime"],
                                "end_time": r2["datetime"],
                                "start_wind_kt": r1["max_wind_kt"],
                                "end_wind_kt": r2["max_wind_kt"],
                                "wind_change_kt": wind_change,
                                "duration_h": round(delta_h, 1),
                            }})
                            break

        results = {{
            "ri_threshold_kt": ri_threshold_kt,
            "ri_window_h": ri_window_h,
            "total_ri_events": len(all_ri_events),
            "ri_events": all_ri_events[:200],
        }}
    except Exception as e:
        results = {{"error": str(e)}}

    Path("results.json").write_text(json.dumps(results, indent=2))
""")


CLIMATE_INDEX_SCRIPT = textwrap.dedent("""\
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    try:
        data_path = Path("{data_path}")
        index_type = "{index_type}"

        if index_type == "oni":
            # ONI: 3-month running mean of SST anomalies in Nino3.4 region
            df = pd.read_csv(data_path, delim_whitespace=True, comment="#",
                             na_values=["-99.9", "-99.99", "***", "-999"])
            # Flexible column detection
            temp_cols = [c for c in df.columns if any(
                k in c.lower() for k in ("anom", "sst", "temp")
            )]
            if temp_cols:
                values = df[temp_cols[0]].dropna().values
            else:
                # Try numeric columns after first (assume first is year/date)
                numeric = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric) > 1:
                    values = df[numeric[1]].dropna().values
                else:
                    values = df.iloc[:, -1].dropna().values

            # 3-month running mean
            if len(values) >= 3:
                oni = np.convolve(values, np.ones(3) / 3, mode="valid")
            else:
                oni = values

            el_nino = int(np.sum(oni >= 0.5))
            la_nina = int(np.sum(oni <= -0.5))
            neutral = int(len(oni) - el_nino - la_nina)

            results = {{
                "index_type": "oni",
                "n_values": len(oni),
                "mean": round(float(np.mean(oni)), 4),
                "std": round(float(np.std(oni)), 4),
                "min": round(float(np.min(oni)), 4),
                "max": round(float(np.max(oni)), 4),
                "el_nino_periods": el_nino,
                "la_nina_periods": la_nina,
                "neutral_periods": neutral,
            }}

        elif index_type in ("sst_anomaly", "wind_shear", "mpi"):
            # Generic: read data, compute basic stats
            df = pd.read_csv(data_path, comment="#",
                             na_values=["-99.9", "-99.99", "-999"])
            numeric = df.select_dtypes(include=[np.number])
            if numeric.empty:
                raise ValueError("No numeric columns found")

            stats = {{}}
            for col in numeric.columns[:10]:
                vals = numeric[col].dropna().values
                if len(vals) == 0:
                    continue
                stats[col] = {{
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "n": len(vals),
                }}
            results = {{"index_type": index_type, "column_stats": stats}}

        else:
            results = {{"error": f"Unknown index type: {{index_type}}"}}

    except Exception as e:
        results = {{"error": str(e)}}

    Path("results.json").write_text(json.dumps(results, indent=2))
""")


TC_ENVIRONMENT_SCRIPT = textwrap.dedent("""\
    import json
    import numpy as np
    from pathlib import Path

    try:
        storm_path = Path("{storm_data_path}")
        storm_data = json.loads(storm_path.read_text())

        era5_path_str = "{era5_data_path}"
        has_era5 = era5_path_str and era5_path_str != "None"

        # Basic TC statistics from storm data
        if isinstance(storm_data, list):
            storms = storm_data
        elif "storms" in storm_data:
            storms = storm_data["storms"]
        else:
            storms = [storm_data]

        all_winds = []
        all_pressures = []
        all_lats = []
        all_lons = []
        for storm in storms:
            for r in storm.get("records", []):
                if r.get("max_wind_kt") is not None:
                    all_winds.append(r["max_wind_kt"])
                if r.get("min_pressure_mb") is not None and r["min_pressure_mb"] > 0:
                    all_pressures.append(r["min_pressure_mb"])
                if r.get("lat") is not None:
                    all_lats.append(r["lat"])
                if r.get("lon") is not None:
                    all_lons.append(r["lon"])

        env_analysis = {{
            "n_storms": len(storms),
            "wind_stats_kt": {{
                "mean": round(float(np.mean(all_winds)), 1) if all_winds else None,
                "max": int(max(all_winds)) if all_winds else None,
                "std": round(float(np.std(all_winds)), 1) if all_winds else None,
            }},
            "pressure_stats_mb": {{
                "mean": round(float(np.mean(all_pressures)), 1) if all_pressures else None,
                "min": int(min(all_pressures)) if all_pressures else None,
                "std": round(float(np.std(all_pressures)), 1) if all_pressures else None,
            }},
            "spatial_extent": {{
                "lat_range": [round(min(all_lats), 1), round(max(all_lats), 1)] if all_lats else None,
                "lon_range": [round(min(all_lons), 1), round(max(all_lons), 1)] if all_lons else None,
            }},
        }}

        if has_era5:
            try:
                import xarray as xr
                era5 = xr.open_dataset(Path(era5_path_str))
                era5_vars = list(era5.data_vars)
                env_analysis["era5_variables"] = era5_vars

                # SST if available
                sst_names = [v for v in era5_vars if "sst" in v.lower() or "sea_surface" in v.lower()]
                if sst_names:
                    sst = era5[sst_names[0]].values.flatten()
                    sst = sst[~np.isnan(sst)]
                    if len(sst) > 0:
                        env_analysis["sst_stats_K"] = {{
                            "mean": round(float(np.mean(sst)), 2),
                            "min": round(float(np.min(sst)), 2),
                            "max": round(float(np.max(sst)), 2),
                        }}
            except Exception as era5_err:
                env_analysis["era5_error"] = str(era5_err)
        else:
            env_analysis["era5_note"] = "No ERA5 data provided"

        results = env_analysis

    except Exception as e:
        results = {{"error": str(e)}}

    Path("results.json").write_text(json.dumps(results, indent=2))
""")


# ---------------------------------------------------------------------------
# Script template registry
# ---------------------------------------------------------------------------

_SCRIPT_TEMPLATES: dict[str, str] = {
    "hurdat2_parse": HURDAT2_PARSE_SCRIPT,
    "sst_extraction": SST_ANOMALY_SCRIPT,
    "sst_anomaly": SST_ANOMALY_SCRIPT,
    "ri_detection": RI_DETECTION_SCRIPT,
    "wind_shear_calc": CLIMATE_INDEX_SCRIPT,
    "climate_index": CLIMATE_INDEX_SCRIPT,
    "tc_environment": TC_ENVIRONMENT_SCRIPT,
}

_REQUIREMENTS: dict[str, list[str]] = {
    "hurdat2_parse": [],
    "sst_extraction": ["numpy", "pandas", "xarray", "netCDF4"],
    "sst_anomaly": ["numpy", "pandas", "xarray", "netCDF4"],
    "ri_detection": [],
    "wind_shear_calc": ["numpy", "pandas"],
    "climate_index": ["numpy", "pandas"],
    "tc_environment": ["numpy", "xarray", "netCDF4"],
}


# ---------------------------------------------------------------------------
# ClimateToolkit
# ---------------------------------------------------------------------------

class ClimateToolkit:
    """Structured climate analysis via sandbox-executed scripts."""

    def __init__(self, config: Any, db: Any, sandbox_runner: Any) -> None:
        self._config = config
        self._db = db
        self._sandbox = sandbox_runner

    # -- public API ---------------------------------------------------------

    def parse_hurdat2(self, data_path: str) -> dict[str, Any]:
        """Parse a HURDAT2 file and return summary statistics."""
        script = self.generate_analysis_script("hurdat2_parse", data_path=data_path)
        result = self._sandbox.execute(
            script=script,
            requirements=_REQUIREMENTS.get("hurdat2_parse", []),
            timeout_s=getattr(self._config, "sandbox_timeout_default_s", 120),
            description=f"Parse HURDAT2: {data_path}",
        )
        if result.metrics:
            return result.metrics
        log.warning(
            "HURDAT2 parse returned no metrics (rc=%d): %s",
            result.returncode, result.stderr[:500],
        )
        return {"error": result.stderr[:1000], "returncode": result.returncode}

    def compute_climate_indices(
        self, dataset_path: str, index_type: str
    ) -> dict[str, Any]:
        """Compute a climate index from dataset. index_type: oni, sst_anomaly, wind_shear, mpi."""
        if index_type not in ("oni", "sst_anomaly", "wind_shear", "mpi"):
            return {"error": f"Unknown index type: {index_type}"}

        script = self.generate_analysis_script(
            "climate_index", data_path=dataset_path, index_type=index_type,
        )
        reqs = _REQUIREMENTS.get("climate_index", ["numpy", "pandas"])
        result = self._sandbox.execute(
            script=script,
            requirements=reqs,
            timeout_s=getattr(self._config, "sandbox_timeout_default_s", 120),
            description=f"Climate index ({index_type}): {dataset_path}",
        )
        if result.metrics:
            return result.metrics
        return {"error": result.stderr[:1000], "returncode": result.returncode}

    def analyze_tc_environment(
        self, storm_data_path: str, era5_data_path: str | None = None,
    ) -> dict[str, Any]:
        """Analyze tropical cyclone environmental conditions."""
        script = self.generate_analysis_script(
            "tc_environment",
            storm_data_path=storm_data_path,
            era5_data_path=era5_data_path or "None",
        )
        reqs = _REQUIREMENTS.get("tc_environment", ["numpy"])
        result = self._sandbox.execute(
            script=script,
            requirements=reqs,
            timeout_s=getattr(self._config, "sandbox_timeout_default_s", 120),
            description=f"TC environment analysis: {storm_data_path}",
        )
        if result.metrics:
            return result.metrics
        return {"error": result.stderr[:1000], "returncode": result.returncode}

    def detect_ri_events(
        self,
        storm_data_path: str,
        ri_threshold_kt: int = 30,
        ri_window_h: int = 24,
    ) -> dict[str, Any]:
        """Detect rapid intensification events in parsed TC data."""
        script = self.generate_analysis_script(
            "ri_detection",
            data_path=storm_data_path,
            ri_threshold_kt=ri_threshold_kt,
            ri_window_h=ri_window_h,
        )
        result = self._sandbox.execute(
            script=script,
            requirements=_REQUIREMENTS.get("ri_detection", []),
            timeout_s=getattr(self._config, "sandbox_timeout_default_s", 120),
            description=f"RI detection (>={ri_threshold_kt}kt/{ri_window_h}h)",
        )
        if result.metrics:
            return result.metrics
        return {"error": result.stderr[:1000], "returncode": result.returncode}

    def compute_sst_anomaly(self, dataset_path: str) -> dict[str, Any]:
        """Compute SST anomalies from a temperature dataset."""
        script = self.generate_analysis_script(
            "sst_anomaly", data_path=dataset_path,
        )
        reqs = _REQUIREMENTS.get("sst_anomaly", ["numpy", "pandas"])
        result = self._sandbox.execute(
            script=script,
            requirements=reqs,
            timeout_s=getattr(self._config, "sandbox_timeout_default_s", 120),
            description=f"SST anomaly: {dataset_path}",
        )
        if result.metrics:
            return result.metrics
        return {"error": result.stderr[:1000], "returncode": result.returncode}

    # -- script generation --------------------------------------------------

    def generate_analysis_script(self, analysis_type: str, **kwargs: Any) -> str:
        """Return a Python script string for the given analysis type.

        Supported analysis_type values:
            hurdat2_parse, sst_extraction, sst_anomaly, ri_detection,
            wind_shear_calc, climate_index, tc_environment
        """
        template = _SCRIPT_TEMPLATES.get(analysis_type)
        if template is None:
            raise ValueError(
                f"Unknown analysis type: {analysis_type!r}. "
                f"Available: {sorted(_SCRIPT_TEMPLATES)}"
            )
        # Apply keyword substitutions into the template
        try:
            return template.format(**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"Missing parameter {exc} for analysis type {analysis_type!r}"
            ) from exc
