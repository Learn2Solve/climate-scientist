from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency 'pyyaml'. Install with `pip install -r requirements.txt`.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Workflow file must be a mapping: {path}")
    return data


def find_workflow_file(workflow: str, workflows_dir: Path) -> Path:
    p = Path(workflow)
    if p.suffix in {".yaml", ".yml", ".json"}:
        if p.is_absolute():
            return p.resolve()
        repo_root = workflows_dir.parent.parent
        return (repo_root / p).resolve()
    return (workflows_dir / f"{workflow}.yaml").resolve()


def _format_obj(obj: Any, ctx: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        try:
            return obj.format(**ctx)
        except Exception:
            return obj
    if isinstance(obj, list):
        return [_format_obj(x, ctx) for x in obj]
    if isinstance(obj, dict):
        return {k: _format_obj(v, ctx) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class Plan:
    workflow: dict[str, Any]
    steps: list[dict[str, Any]]
    data_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"workflow": self.workflow, "steps": self.steps, "data_files": self.data_files}


def build_plan(workflow_path: Path, ctx: dict[str, Any]) -> Plan:
    if workflow_path.suffix == ".json":
        spec = json.loads(workflow_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise SystemExit(f"Workflow JSON must be an object: {workflow_path}")
    else:
        spec = _load_yaml(workflow_path)

    steps_raw = spec.get("steps", [])
    if not isinstance(steps_raw, list):
        raise SystemExit(f"'steps' must be a list: {workflow_path}")

    steps: list[dict[str, Any]] = []
    prev_id: str | None = None
    for i, step in enumerate(steps_raw):
        if not isinstance(step, dict):
            raise SystemExit(f"Step must be an object (index {i}): {workflow_path}")
        step_id = str(step.get("id") or f"step_{i+1}")
        resolved = _format_obj(step, ctx)
        resolved["id"] = step_id
        resolved.setdefault("type", "note")
        if prev_id:
            resolved.setdefault("depends_on", [prev_id])
        else:
            resolved.setdefault("depends_on", [])
        prev_id = step_id
        steps.append(resolved)

    workflow_meta = {
        "name": str(spec.get("name") or workflow_path.stem),
        "description": str(spec.get("description") or ""),
        "source": str(workflow_path),
    }

    data_files_raw = spec.get("data_files", []) or []
    if not isinstance(data_files_raw, list):
        raise SystemExit(f"'data_files' must be a list: {workflow_path}")
    data_files = [str(_format_obj(x, ctx)) for x in data_files_raw]

    return Plan(workflow=workflow_meta, steps=steps, data_files=data_files)
