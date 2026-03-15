"""Data acquisition - download, cache, and manage climate datasets.

Uses web_tools.http_get() for downloads with SSRF protection.
Files cached by URL (UNIQUE constraint) in data/cache/{name}/.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ulid import ULID

log = logging.getLogger("daemon.data_acquisition")


KNOWN_CLIMATE_SOURCES: dict[str, dict[str, Any]] = {
    "hurdat2": {
        "url": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt",
        "format": "txt",
        "description": "HURDAT2 Atlantic hurricane database (1851-2023)",
    },
    "hurdat2_pacific": {
        "url": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2023-042624.txt",
        "format": "txt",
        "description": "HURDAT2 NE/Central Pacific hurricane database (1949-2023)",
    },
    "ibtracs_csv": {
        "url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv",
        "format": "csv",
        "description": "IBTrACS global tropical cyclone best track data (CSV)",
    },
    "noaa_sst_monthly": {
        "url": "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/oisst-avhrr-v02r01.monthly.nc",
        "format": "nc",
        "description": "NOAA OISSTv2.1 monthly sea surface temperature (NetCDF)",
    },
    "oni_index": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        "format": "txt",
        "description": "Oceanic Nino Index (ONI) - El Nino/La Nina indicator",
    },
}


class DataAcquisition:
    """Download, cache, and manage climate datasets."""

    def __init__(self, config: Any, db: Any) -> None:
        self._config = config
        self._db = db
        repo_root = Path(getattr(config, "repo_root", Path.cwd()))
        self._cache_dir = repo_root / "data" / "cache"
        self._max_bytes = getattr(config, "max_dataset_size_bytes", 100_000_000)

    def fetch_dataset(
        self,
        url: str,
        name: str,
        format_hint: str = "csv",
        max_bytes: int | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        """Download and cache a dataset. Returns DatasetRecord dict."""
        from agent.web_tools import validate_http_url

        # SSRF check
        ok, err = validate_http_url(url)
        if not ok:
            raise ValueError(f"URL blocked: {err}")

        # Check cache
        cached = self._db.get_dataset_by_url(url)
        if cached:
            local = Path(cached["local_path"])
            if local.exists():
                log.info("Dataset '%s' already cached at %s", name, local)
                return cached
            # File missing, re-download
            log.info("Cached record exists but file missing, re-downloading '%s'", name)

        limit = min(max_bytes or self._max_bytes, self._max_bytes)

        # Prepare target directory
        safe_name = _sanitize_name(name)
        target_dir = self._cache_dir / safe_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename from URL
        parsed = urllib.parse.urlparse(url)
        url_filename = Path(parsed.path).name or f"{safe_name}.{format_hint}"
        target_path = target_dir / url_filename

        # Download with streaming
        log.info("Downloading dataset '%s' from %s", name, url)
        sha256, size = _stream_download(url, target_path, limit)

        # Store record
        record = {
            "id": str(ULID()),
            "name": name,
            "source_url": url,
            "format": format_hint,
            "local_path": str(target_path),
            "sha256": sha256,
            "size_bytes": size,
            "description": description,
            "verified": True,
        }
        self._db.insert_dataset(record)
        log.info(
            "Dataset '%s' cached: %d bytes, sha256=%s",
            name, size, sha256[:12],
        )
        return record

    def list_cached(self) -> list[dict[str, Any]]:
        """List all cached datasets."""
        return self._db.get_all_datasets()

    def search_known_sources(self, query: str) -> list[dict[str, Any]]:
        """Search known climate data sources by keyword."""
        query_lower = query.lower()
        results = []
        for key, info in KNOWN_CLIMATE_SOURCES.items():
            searchable = f"{key} {info.get('description', '')} {info.get('format', '')}".lower()
            if any(w in searchable for w in query_lower.split()):
                results.append({
                    "name": key,
                    "url": info["url"],
                    "format": info.get("format", "csv"),
                    "description": info.get("description", ""),
                })

        # Also search cached datasets
        cached = self._db.get_all_datasets()
        for ds in cached:
            searchable = f"{ds['name']} {ds.get('description', '')}".lower()
            if any(w in searchable for w in query_lower.split()):
                results.append({
                    "name": ds["name"],
                    "url": ds["source_url"],
                    "format": ds.get("format", "csv"),
                    "description": ds.get("description", ""),
                    "cached": True,
                    "local_path": ds.get("local_path", ""),
                    "size_bytes": ds.get("size_bytes", 0),
                })

        return results


def _sanitize_name(name: str) -> str:
    """Create a filesystem-safe name."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return safe[:100] or "dataset"


def _stream_download(url: str, target: Path, max_bytes: int) -> tuple[str, int]:
    """Stream-download a URL to disk. Returns (sha256, size_bytes).

    Aborts if content-length or actual size exceeds max_bytes.
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": "climate-scientist-agent/0.1"}
    )
    hasher = hashlib.sha256()
    size = 0

    with urllib.request.urlopen(req, timeout=60) as resp:
        # Check content-length if available
        cl = resp.headers.get("Content-Length")
        if cl and int(cl) > max_bytes:
            raise ValueError(
                f"Dataset too large: {int(cl)} bytes > {max_bytes} byte limit"
            )

        with open(target, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    target.unlink(missing_ok=True)
                    raise ValueError(
                        f"Download exceeded {max_bytes} byte limit at {size} bytes"
                    )
                hasher.update(chunk)
                f.write(chunk)

    return hasher.hexdigest(), size
