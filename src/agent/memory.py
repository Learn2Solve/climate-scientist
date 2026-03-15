from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    artifacts_dir: Path
    plan_json: Path
    tool_calls_jsonl: Path
    env_lock: Path
    data_manifest_json: Path
    metrics_json: Path
    report_md: Path


class RunMemory:
    def __init__(self, run_dir: Path):
        artifacts_dir = run_dir / "artifacts"
        self.paths = RunPaths(
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            plan_json=run_dir / "plan.json",
            tool_calls_jsonl=run_dir / "tool_calls.jsonl",
            env_lock=run_dir / "env.lock",
            data_manifest_json=run_dir / "data_manifest.json",
            metrics_json=run_dir / "metrics.json",
            report_md=run_dir / "report.md",
        )
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def append_jsonl(self, path: Path, obj: Any) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def write_text(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    def log_tool_call(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", utc_now())
        self.append_jsonl(self.paths.tool_calls_jsonl, record)

    def record_env_lock(self, repo_root: Path) -> None:
        lines = []
        lines.append(f"created_at: {utc_now()}")
        lines.append(f"python_executable: {sys.executable}")
        lines.append(f"python_version: {sys.version.replace(chr(10), ' ')}")
        lines.append(f"platform: {platform.platform()}")
        lines.append("")

        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
            lines.append(f"git_commit: {sha}")
        except Exception:
            lines.append("git_commit: <unknown>")

        try:
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), text=True)
            status = status.strip()
            lines.append(f"git_dirty: {bool(status)}")
        except Exception:
            lines.append("git_dirty: <unknown>")

        lines.append("")
        lines.append("python_packages:")
        try:
            import importlib.metadata as importlib_metadata

            pkgs = []
            for dist in importlib_metadata.distributions():
                name = dist.metadata.get("Name") or getattr(dist, "name", "")
                version = getattr(dist, "version", "")
                if name and version:
                    pkgs.append(f"{name}=={version}")
            for item in sorted(set(pkgs), key=str.lower):
                lines.append(item)
        except Exception as exc:
            lines.append(f"<package listing failed: {exc}>")
        lines.append("")

        self.write_text(self.paths.env_lock, "\n".join(lines))

    def record_data_manifest(self, repo_root: Path, files: list[Path], extra: dict[str, Any] | None = None) -> None:
        items = []
        for p in files:
            p = p.resolve()
            if not p.exists() or not p.is_file():
                items.append({"path": safe_relpath(p, repo_root), "exists": False})
                continue
            st = p.stat()
            items.append(
                {
                    "path": safe_relpath(p, repo_root),
                    "exists": True,
                    "bytes": int(st.st_size),
                    "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                    "sha256": sha256_file(p),
                }
            )
        manifest: dict[str, Any] = {"created_at": utc_now(), "files": items}
        if extra:
            manifest["extra"] = extra
        self.write_json(self.paths.data_manifest_json, manifest)
