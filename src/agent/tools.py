from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


@dataclass(frozen=True)
class ShellResult:
    cmd: str
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    started_at: str
    ended_at: str
    duration_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_shell(cmd: str, cwd: Path, timeout_s: int | None) -> ShellResult:
    started_at = utc_now()
    t0 = time.time()
    env = os.environ.copy()
    # In sandboxed environments, the default uv cache dir may be unreadable. Use a safe default.
    env.setdefault("UV_CACHE_DIR", str(Path("/tmp/uv-cache").resolve()))
    proc = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        cwd=str(cwd),
        env=env,
        timeout=timeout_s,
    )
    duration_s = time.time() - t0
    ended_at = utc_now()
    return ShellResult(
        cmd=cmd,
        cwd=str(cwd),
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
    )
