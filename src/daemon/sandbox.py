"""Sandbox execution - Docker primary, local subprocess fallback.

Runs Python scripts in isolation, captures stdout/stderr/metrics,
stores results in sandbox_executions table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ulid import ULID

log = logging.getLogger("daemon.sandbox")


@dataclass
class SandboxResult:
    execution_id: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    artifacts: list[str]
    metrics: dict | None
    backend: str  # "docker" | "local"


class SandboxBackend(Protocol):
    def run(
        self, script_path: Path, work_dir: Path, timeout_s: int
    ) -> SandboxResult: ...

    def is_available(self) -> bool: ...


class DockerSandboxBackend:
    """Run scripts in a Docker container with resource limits."""

    def __init__(self, config: Any) -> None:
        self._image = getattr(config, "sandbox_docker_image", "python:3.13-slim")
        self._memory = getattr(config, "sandbox_docker_memory", "2g")
        self._cpus = getattr(config, "sandbox_docker_cpus", "2")
        self._network = getattr(config, "sandbox_docker_network", "none")
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            self._available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            self._available = False
        log.info("Docker available: %s", self._available)
        return self._available

    def run(
        self, script_path: Path, work_dir: Path, timeout_s: int
    ) -> SandboxResult:
        execution_id = str(ULID())
        start = time.monotonic()

        # Build docker command
        cmd = [
            "docker", "run", "--rm",
            f"--memory={self._memory}",
            f"--cpus={self._cpus}",
            f"--network={self._network}",
            "-v", f"{work_dir.resolve()}:/workspace",
            "-w", "/workspace",
            self._image,
            "bash", "-c",
            "pip install -q -r requirements.txt 2>/dev/null; python script.py",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            returncode = proc.returncode
            stdout = proc.stdout[:8000]
            stderr = proc.stderr[:4000]
        except subprocess.TimeoutExpired:
            returncode = -1
            stdout = ""
            stderr = f"Timeout after {timeout_s}s"
        except Exception as exc:
            returncode = -2
            stdout = ""
            stderr = str(exc)[:4000]

        duration = time.monotonic() - start

        # Parse results.json if present
        metrics = _parse_results_json(work_dir)

        # Collect artifacts
        artifacts = _collect_artifacts(work_dir)

        return SandboxResult(
            execution_id=execution_id,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=round(duration, 2),
            artifacts=artifacts,
            metrics=metrics,
            backend="docker",
        )


class LocalSandboxBackend:
    """Run scripts locally via uv run as fallback."""

    def is_available(self) -> bool:
        return True

    def run(
        self, script_path: Path, work_dir: Path, timeout_s: int
    ) -> SandboxResult:
        execution_id = str(ULID())
        start = time.monotonic()

        # Install requirements if present
        req_file = work_dir / "requirements.txt"
        if req_file.exists() and req_file.read_text().strip():
            try:
                subprocess.run(
                    ["uv", "pip", "install", "-q", "-r", str(req_file)],
                    capture_output=True,
                    timeout=60,
                    cwd=str(work_dir),
                )
            except Exception as exc:
                log.warning("Failed to install requirements: %s", exc)

        # Run the script
        cmd = ["uv", "run", "--no-project", "python", "script.py"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(work_dir),
            )
            returncode = proc.returncode
            stdout = proc.stdout[:8000]
            stderr = proc.stderr[:4000]
        except subprocess.TimeoutExpired:
            returncode = -1
            stdout = ""
            stderr = f"Timeout after {timeout_s}s"
        except Exception as exc:
            returncode = -2
            stdout = ""
            stderr = str(exc)[:4000]

        duration = time.monotonic() - start

        metrics = _parse_results_json(work_dir)
        artifacts = _collect_artifacts(work_dir)

        return SandboxResult(
            execution_id=execution_id,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=round(duration, 2),
            artifacts=artifacts,
            metrics=metrics,
            backend="local",
        )


class SandboxRunner:
    """High-level sandbox executor with Docker/local backend selection."""

    def __init__(self, config: Any, db: Any) -> None:
        self._config = config
        self._db = db
        self._docker = DockerSandboxBackend(config)
        self._local = LocalSandboxBackend()
        self._sandbox_root = Path(
            getattr(config, "repo_root", Path.cwd())
        ) / ".agent" / "sandbox"

    def execute(
        self,
        script: str,
        requirements: list[str] | None = None,
        data_files: dict[str, str] | None = None,
        timeout_s: int | None = None,
        description: str = "",
        linked_experiment_id: str | None = None,
    ) -> SandboxResult:
        """Execute a Python script in sandbox, store result in DB."""
        max_timeout = getattr(self._config, "sandbox_timeout_max_s", 600)
        default_timeout = getattr(self._config, "sandbox_timeout_default_s", 120)
        timeout = min(timeout_s or default_timeout, max_timeout)

        # Create workspace
        execution_id = str(ULID())
        work_dir = self._sandbox_root / execution_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write script
        script_path = work_dir / "script.py"
        script_path.write_text(script, encoding="utf-8")

        # Write requirements
        req_path = work_dir / "requirements.txt"
        req_path.write_text("\n".join(requirements or []), encoding="utf-8")

        # Write data files
        if data_files:
            for filename, content in data_files.items():
                fpath = work_dir / filename
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content, encoding="utf-8")

        # Select backend
        backend: SandboxBackend
        if self._docker.is_available():
            backend = self._docker
        else:
            backend = self._local
            log.info("Docker unavailable, using local backend")

        # Execute
        result = backend.run(script_path, work_dir, timeout)
        result.execution_id = execution_id

        # Store in DB
        script_hash = hashlib.sha256(script.encode()).hexdigest()[:16]
        self._db.insert_sandbox_execution({
            "id": execution_id,
            "script_hash": script_hash,
            "description": description,
            "backend": result.backend,
            "returncode": result.returncode,
            "stdout_preview": result.stdout[:2000],
            "stderr_preview": result.stderr[:1000],
            "metrics": result.metrics,
            "artifacts": result.artifacts,
            "duration_s": result.duration_s,
            "work_dir": str(work_dir),
            "linked_experiment_id": linked_experiment_id,
        })

        return result

    def cleanup_old_workspaces(self, max_age_days: int = 7, keep_last: int = 50) -> int:
        """Delete old sandbox workspaces. Returns number deleted."""
        if not self._sandbox_root.exists():
            return 0

        # Get recent execution IDs to keep
        recent = self._db.get_recent_sandbox_executions(limit=keep_last)
        keep_ids = {e["id"] for e in recent}

        deleted = 0
        for child in sorted(self._sandbox_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name in keep_ids:
                continue
            # Check age
            try:
                age_days = (time.time() - child.stat().st_mtime) / 86400
                if age_days > max_age_days:
                    shutil.rmtree(child, ignore_errors=True)
                    deleted += 1
            except OSError:
                pass

        return deleted


def _parse_results_json(work_dir: Path) -> dict | None:
    """Parse results.json from workspace if present."""
    results_file = work_dir / "results.json"
    if results_file.exists():
        try:
            return json.loads(results_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _collect_artifacts(work_dir: Path) -> list[str]:
    """List files in workspace that are artifacts (not script/requirements)."""
    skip = {"script.py", "requirements.txt"}
    artifacts = []
    for f in work_dir.iterdir():
        if f.name not in skip and f.is_file():
            artifacts.append(f.name)
    return artifacts
