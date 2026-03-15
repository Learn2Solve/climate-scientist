"""Authenticated data access - ERA5, ESGF, NOAA climate indices.

Wraps APIs that require credentials (ERA5/CDS) and freely available
indices (NOAA). Downloads are executed via sandbox scripts (ERA5) or
direct HTTP (NOAA/ESGF search).
"""

from __future__ import annotations

import json
import logging
import textwrap
import urllib.parse
from pathlib import Path
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.data_apis")

# ---------------------------------------------------------------------------
# NOAA index URL registry
# ---------------------------------------------------------------------------

NOAA_INDEX_URLS: dict[str, dict[str, str]] = {
    "oni": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        "description": "Oceanic Nino Index (3-month running mean SST anomaly, Nino3.4)",
    },
    "amo": {
        "url": "https://www.psl.noaa.gov/data/correlation/amon.us.data",
        "description": "Atlantic Multidecadal Oscillation unsmoothed index",
    },
    "pdo": {
        "url": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
        "description": "Pacific Decadal Oscillation index (ERSSTv5)",
    },
    "nao": {
        "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        "description": "North Atlantic Oscillation monthly index",
    },
    "soi": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
        "description": "Southern Oscillation Index",
    },
    "nino34": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
        "description": "Nino3.4 SST indices (weekly/monthly)",
    },
}


# ---------------------------------------------------------------------------
# ERA5 download script template
# ---------------------------------------------------------------------------

ERA5_DOWNLOAD_SCRIPT = textwrap.dedent("""\
    import json
    import cdsapi
    from pathlib import Path

    try:
        c = cdsapi.Client(url="{api_url}", key="{api_key}")
        request = {{
            "product_type": "reanalysis",
            "variable": {variables},
            "year": "{year}",
            "month": "{month:02d}",
            "day": [f"{{d:02d}}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": {area},
            "format": "netcdf",
        }}

        dataset = "{dataset}"
        output_file = "output.nc"

        {pressure_block}

        c.retrieve(dataset, request, output_file)

        out = Path(output_file)
        results = {{
            "success": True,
            "file": output_file,
            "size_bytes": out.stat().st_size if out.exists() else 0,
            "variables": {variables},
            "year": "{year}",
            "month": "{month:02d}",
        }}
    except Exception as e:
        results = {{"success": False, "error": str(e)}}

    Path("results.json").write_text(json.dumps(results, indent=2))
""")


# ---------------------------------------------------------------------------
# AuthenticatedDataAccess
# ---------------------------------------------------------------------------

class AuthenticatedDataAccess:
    """Access ERA5, ESGF, and NOAA climate data APIs."""

    def __init__(self, config: Any, db: Any) -> None:
        self._config = config
        self._db = db
        self._era5_api_key: str | None = getattr(config, "era5_api_key", None)
        self._era5_api_url: str = getattr(
            config, "era5_api_url", "https://cds.climate.copernicus.eu/api"
        )
        repo_root = Path(getattr(config, "repo_root", Path.cwd()))
        self._cache_dir = repo_root / "data" / "cache"

    # -- ERA5 ---------------------------------------------------------------

    def fetch_era5(
        self,
        variable: str | list[str],
        year: str | int,
        month: str | int,
        area: list[float] | None = None,
        pressure_levels: list[int] | None = None,
    ) -> dict[str, Any]:
        """Download ERA5 reanalysis data via the CDS API.

        Returns metadata dict with dataset_id and local_path on success,
        or an error dict if the API key is not configured or download fails.
        """
        if not self._era5_api_key:
            return {
                "error": "ERA5 API key not configured. "
                "Set era5_api_key in .agent/config.yml or CDS_API_KEY env var.",
                "configured": False,
            }

        variables = [variable] if isinstance(variable, str) else list(variable)
        year_str = str(year)
        month_int = int(month)
        area_list = area or [90, -180, -90, 180]  # global

        # Determine dataset and pressure level block
        if pressure_levels:
            dataset = "reanalysis-era5-pressure-levels"
            pressure_block = f"request['pressure_level'] = {pressure_levels!r}"
        else:
            dataset = "reanalysis-era5-single-levels"
            pressure_block = "# Single levels - no pressure_level needed"

        script = ERA5_DOWNLOAD_SCRIPT.format(
            api_url=self._era5_api_url,
            api_key=self._era5_api_key,
            variables=variables,
            year=year_str,
            month=month_int,
            area=area_list,
            dataset=dataset,
            pressure_block=pressure_block,
        )

        # Execute via sandbox
        from .sandbox import SandboxRunner
        sandbox: SandboxRunner | None = None

        # Try to get sandbox from db's parent context; fall back to constructing one
        try:
            sandbox = SandboxRunner(self._config, self._db)
        except Exception as exc:
            log.error("Could not create sandbox runner: %s", exc)
            return {"error": f"Sandbox unavailable: {exc}"}

        result = sandbox.execute(
            script=script,
            requirements=["cdsapi", "netCDF4"],
            timeout_s=min(
                300,
                getattr(self._config, "sandbox_timeout_max_s", 600),
            ),
            description=f"ERA5 download: {variables} {year_str}-{month_int:02d}",
        )

        if result.metrics and result.metrics.get("success"):
            dataset_id = str(ULID())
            local_path = str(
                Path(result.metrics.get("file", "output.nc"))
            )
            record = {
                "id": dataset_id,
                "name": f"era5_{'-'.join(variables)}_{year_str}_{month_int:02d}",
                "source_url": f"{self._era5_api_url}/retrieve/{dataset}",
                "format": "nc",
                "local_path": local_path,
                "sha256": "",
                "size_bytes": result.metrics.get("size_bytes", 0),
                "description": f"ERA5 {variables} for {year_str}-{month_int:02d}",
                "verified": False,
            }
            try:
                self._db.insert_dataset(record)
            except Exception as db_err:
                log.warning("Failed to store ERA5 dataset record: %s", db_err)

            return {
                "dataset_id": dataset_id,
                "local_path": local_path,
                "variables": variables,
                "year": year_str,
                "month": f"{month_int:02d}",
                "size_bytes": result.metrics.get("size_bytes", 0),
            }

        error_msg = (
            result.metrics.get("error", "") if result.metrics
            else result.stderr[:1000]
        )
        return {"error": error_msg, "returncode": result.returncode}

    # -- ESGF ---------------------------------------------------------------

    def search_esgf(
        self,
        query: str,
        project: str = "CMIP6",
        experiment: str | None = None,
        variable: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search ESGF for climate model datasets.

        Uses the ESGF search API (no authentication required).
        Returns a dict with count and dataset list.
        """
        from agent.web_tools import http_get

        params: dict[str, str] = {
            "type": "Dataset",
            "format": "application/solr+json",
            "limit": str(min(limit, 100)),
            "project": project,
            "query": query,
        }
        if experiment:
            params["experiment_id"] = experiment
        if variable:
            params["variable_id"] = variable

        base_url = "https://esgf-node.llnl.gov/esg-search/search"
        url = base_url + "?" + urllib.parse.urlencode(params)

        try:
            resp = http_get(url, timeout_s=30.0, max_bytes=2_000_000)
            data = json.loads(resp.body.decode("utf-8", errors="replace"))
        except Exception as exc:
            log.error("ESGF search failed: %s", exc)
            return {"error": str(exc), "count": 0, "datasets": []}

        response = data.get("response", {})
        num_found = response.get("numFound", 0)
        docs = response.get("docs", [])

        datasets = []
        for doc in docs[:limit]:
            datasets.append({
                "id": doc.get("id", ""),
                "title": doc.get("title", doc.get("id", "")),
                "project": doc.get("project", [project])[0]
                    if isinstance(doc.get("project"), list)
                    else doc.get("project", project),
                "experiment": doc.get("experiment_id", [""])[0]
                    if isinstance(doc.get("experiment_id"), list)
                    else doc.get("experiment_id", ""),
                "variables": doc.get("variable_id", [])
                    if isinstance(doc.get("variable_id"), list)
                    else [doc.get("variable_id", "")],
                "url": doc.get("url", [""])[0]
                    if isinstance(doc.get("url"), list)
                    else doc.get("url", ""),
                "size": doc.get("size", 0),
                "replica": doc.get("replica", False),
            })

        return {
            "query": query,
            "project": project,
            "count": num_found,
            "datasets": datasets,
        }

    # -- NOAA indices -------------------------------------------------------

    def fetch_noaa_indices(self, index_name: str) -> dict[str, Any]:
        """Download a NOAA climate index file.

        Supported indices: oni, amo, pdo, nao, soi, nino34.
        Returns dict with local_path and description.
        """
        index_lower = index_name.lower()
        info = NOAA_INDEX_URLS.get(index_lower)
        if info is None:
            return {
                "error": f"Unknown index: {index_name!r}. "
                f"Available: {sorted(NOAA_INDEX_URLS)}",
            }

        url = info["url"]
        description = info["description"]

        # Check if already cached in DB
        cached = self._db.get_dataset_by_name(f"noaa_{index_lower}")
        if cached:
            local = Path(cached["local_path"])
            if local.exists():
                log.info("NOAA %s index already cached at %s", index_lower, local)
                return {
                    "local_path": cached["local_path"],
                    "description": description,
                    "cached": True,
                    "dataset_id": cached["id"],
                }

        # Download via DataAcquisition
        try:
            from .data_acquisition import DataAcquisition
            da = DataAcquisition(self._config, self._db)
            record = da.fetch_dataset(
                url=url,
                name=f"noaa_{index_lower}",
                format_hint="txt",
                description=description,
            )
            return {
                "local_path": record["local_path"],
                "description": description,
                "cached": False,
                "dataset_id": record["id"],
            }
        except Exception as exc:
            log.error("Failed to download NOAA %s index: %s", index_lower, exc)
            return {"error": str(exc)}

    # -- API listing --------------------------------------------------------

    def list_available_apis(self) -> dict[str, Any]:
        """Report which data APIs are configured and available."""
        return {
            "era5": {
                "configured": bool(self._era5_api_key),
                "api_url": self._era5_api_url,
                "description": "ERA5 reanalysis via Copernicus CDS API",
            },
            "esgf": {
                "configured": True,
                "api_url": "https://esgf-node.llnl.gov/esg-search/search",
                "description": "ESGF CMIP6/CMIP5 model output search (no auth)",
            },
            "noaa": {
                "configured": True,
                "indices": sorted(NOAA_INDEX_URLS),
                "description": "NOAA climate indices (ONI, AMO, PDO, NAO, SOI, Nino3.4)",
            },
        }
