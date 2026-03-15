from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_path(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def verify_checks(checks: list[dict[str, Any]], repo_root: Path) -> list[str]:
    errors: list[str] = []
    for check in checks:
        if not isinstance(check, dict) or len(check) != 1:
            errors.append(f"invalid check: {check}")
            continue
        key = next(iter(check.keys()))
        val = check[key]

        if key == "path_exists":
            p = resolve_path(str(val), repo_root)
            if not p.exists():
                errors.append(f"missing path: {p}")
        elif key == "non_empty":
            p = resolve_path(str(val), repo_root)
            if not p.exists():
                errors.append(f"missing file: {p}")
            elif p.is_file() and p.stat().st_size == 0:
                errors.append(f"empty file: {p}")
        elif key == "json_parse":
            p = resolve_path(str(val), repo_root)
            if not p.exists():
                errors.append(f"missing json file: {p}")
            else:
                try:
                    json.loads(p.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"json parse failed: {p} ({exc})")
        else:
            errors.append(f"unknown check: {key}")

    return errors

