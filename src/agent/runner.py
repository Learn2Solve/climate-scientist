from __future__ import annotations

import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory import RunMemory
from .safety import SafetyConfig, check_shell_safe
from .tools import run_shell, tail
from .verifier import resolve_path, verify_checks


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StepResult:
    step_id: str
    step_type: str
    ok: bool
    started_at: str
    ended_at: str
    error: str | None = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_plan(
    plan: dict[str, Any],
    repo_root: Path,
    memory: RunMemory,
    safety: SafetyConfig,
    stop_on_error: bool = True,
) -> dict[str, Any]:
    steps = plan.get("steps") or []
    if not isinstance(steps, list):
        raise SystemExit("plan['steps'] must be a list")
    if len(steps) > safety.max_steps:
        raise SystemExit(f"plan has {len(steps)} steps, exceeds max_steps={safety.max_steps}")

    results: list[StepResult] = []
    ok_all = True

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise SystemExit(f"step {idx} is not an object")

        step_id = str(step.get("id") or f"step_{idx+1}")
        step_type = str(step.get("type") or "note")
        started_at = utc_now()

        err: str | None = None
        ok = True

        if step_type == "verify":
            checks = step.get("checks") or []
            if not isinstance(checks, list):
                err = "verify checks must be a list"
                ok = False
            else:
                errors = verify_checks(checks, repo_root)
                if errors:
                    ok = False
                    err = "; ".join(errors)

            memory.log_tool_call(
                {
                    "step_id": step_id,
                    "type": "verify",
                    "ok": ok,
                    "error": err,
                }
            )
        elif step_type == "shell":
            cmd = str(step.get("cmd") or "").strip()
            safe, reason = check_shell_safe(cmd)
            if not safe:
                ok = False
                err = reason
                memory.log_tool_call(
                    {
                        "step_id": step_id,
                        "type": "shell",
                        "cmd": cmd,
                        "ok": ok,
                        "error": err,
                    }
                )
            else:
                try:
                    tokens = shlex.split(cmd)
                except Exception:
                    tokens = []
                if tokens and tokens[0] in {"python", "python3"}:
                    ok = False
                    err = "blocked: use `uv run --no-project python <script>` (uv-managed venv required)"
                    memory.log_tool_call(
                        {
                            "step_id": step_id,
                            "type": "shell",
                            "cmd": cmd,
                            "ok": ok,
                            "error": err,
                        }
                    )
                    ended_at = utc_now()
                    results.append(
                        StepResult(
                            step_id=step_id,
                            step_type=step_type,
                            ok=ok,
                            started_at=started_at,
                            ended_at=ended_at,
                            error=err,
                        )
                    )
                    ok_all = False
                    if stop_on_error:
                        break
                    continue
                timeout_s = step.get("timeout_s")
                if timeout_s is None:
                    timeout_s = safety.default_timeout_s
                try:
                    res = run_shell(cmd, cwd=repo_root, timeout_s=int(timeout_s) if timeout_s else None)
                    out_path = memory.paths.artifacts_dir / f"{idx+1:02d}_{step_id}.stdout.txt"
                    err_path = memory.paths.artifacts_dir / f"{idx+1:02d}_{step_id}.stderr.txt"
                    _ensure_parent(out_path)
                    out_path.write_text(res.stdout, encoding="utf-8")
                    err_path.write_text(res.stderr, encoding="utf-8")
                    ok = res.returncode == 0
                    if not ok:
                        err = f"returncode={res.returncode}"
                    memory.log_tool_call(
                        {
                            "step_id": step_id,
                            "type": "shell",
                            "cmd": cmd,
                            "cwd": str(repo_root),
                            "returncode": res.returncode,
                            "ok": ok,
                            "stdout_tail": tail(res.stdout, safety.max_log_chars),
                            "stderr_tail": tail(res.stderr, safety.max_log_chars),
                            "stdout_file": str(out_path),
                            "stderr_file": str(err_path),
                            "started_at": res.started_at,
                            "ended_at": res.ended_at,
                            "duration_s": res.duration_s,
                        }
                    )
                except Exception as exc:
                    ok = False
                    err = str(exc)
                    memory.log_tool_call(
                        {
                            "step_id": step_id,
                            "type": "shell",
                            "cmd": cmd,
                            "ok": ok,
                            "error": err,
                        }
                    )
        elif step_type == "note":
            memory.log_tool_call(
                {
                    "step_id": step_id,
                    "type": "note",
                    "message": step.get("message", ""),
                    "ok": True,
                }
            )
        else:
            ok = False
            err = f"unknown step type: {step_type}"
            memory.log_tool_call(
                {
                    "step_id": step_id,
                    "type": "error",
                    "ok": ok,
                    "error": err,
                }
            )

        outputs = step.get("outputs") or []
        if ok and outputs:
            if not isinstance(outputs, list):
                ok = False
                err = "outputs must be a list"
            else:
                missing = []
                for out_str in outputs:
                    p = resolve_path(str(out_str), repo_root)
                    if not p.exists():
                        missing.append(str(p))
                    elif p.is_file() and p.stat().st_size == 0:
                        missing.append(f"{p} (empty)")
                if missing:
                    ok = False
                    err = f"missing outputs: {missing}"

        ended_at = utc_now()
        results.append(
            StepResult(
                step_id=step_id,
                step_type=step_type,
                ok=ok,
                started_at=started_at,
                ended_at=ended_at,
                error=err,
            )
        )

        if not ok:
            ok_all = False
            if stop_on_error:
                break

    return {
        "ok": ok_all,
        "steps": [
            {
                "id": r.step_id,
                "type": r.step_type,
                "ok": r.ok,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "error": r.error,
            }
            for r in results
        ],
    }
