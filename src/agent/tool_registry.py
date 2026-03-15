from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from .config import AgentConfig
from .memory import RunMemory
from .safety import check_shell_safe
from .tools import run_shell, tail
from .web_tools import ensure_dir, html_to_text, openalex_search_works, sha256_bytes, validate_http_url, http_get


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _resolve_under_repo(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def _blocked_read_path(path: Path) -> bool:
    # Avoid leaking secrets into prompts/logs.
    return path.name == ".env"


def _allowed_write_roots(cfg: AgentConfig) -> list[Path]:
    roots = [cfg.docs_root, cfg.run_dir]
    if cfg.allow_code_edit:
        roots.append((cfg.repo_root / "src").resolve())
    return roots


def _check_write_path(path: Path, cfg: AgentConfig) -> tuple[bool, str]:
    if not _is_within(path, cfg.repo_root):
        return False, f"path outside repo: {path}"
    for root in _allowed_write_roots(cfg):
        if _is_within(path, root):
            return True, ""
    return False, f"write blocked by policy: {path}"


def _diff_text(old: str, new: str, rel_path: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )


def _write_artifact_text(run_dir: Path, name: str, content: str) -> Path:
    p = (run_dir / "artifacts" / name).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _validate_git_apply_paths(patch_text: str, cfg: AgentConfig) -> tuple[bool, str, list[str]]:
    touched: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a_path = parts[2].removeprefix("a/")
                b_path = parts[3].removeprefix("b/")
                if a_path == "/dev/null":
                    touched.append(b_path)
                else:
                    touched.append(a_path)
                    touched.append(b_path)
    touched = [p for p in touched if p and p != "/dev/null"]
    for p_str in touched:
        p = _resolve_under_repo(p_str, cfg.repo_root)
        ok, reason = _check_write_path(p, cfg)
        if not ok:
            return False, reason, touched
    return True, "", touched


def validate_cmd(cmd: str, cfg: AgentConfig) -> tuple[bool, str]:
    safe, reason = check_shell_safe(cmd)
    if not safe:
        return False, reason

    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        return False, f"cmd parse error: {exc}"
    if not tokens:
        return False, "empty command"

    # Prevent accidental secret leakage.
    if any(t.endswith("/.env") or t == ".env" for t in tokens):
        return False, "blocked: cannot access .env"

    python_execs = {"python", "python3"}
    allowed_first = {"ls", "cat", "head", "tail", "rg", "git", "uv", "find", "wc", "sort", "grep"} | python_execs

    first = tokens[0]
    first_path = None
    if "/" in first or first.startswith("."):
        try:
            first_path = Path(first).resolve()
        except Exception:
            first_path = None

    if first not in allowed_first:
        # Allow calling the repo venv python explicitly.
        if first_path and first_path.name.startswith("python"):
            venv_py = (cfg.repo_root / ".venv" / "bin" / "python").resolve()
            if first_path == venv_py:
                tokens[0] = "python"
            else:
                return False, f"blocked: python executable not allowed ({first_path})"
        else:
            return False, f"blocked: command not allowed ({first})"

    if tokens[0] in {"cat", "head", "tail"}:
        return True, ""

    if tokens[0] == "ls":
        return True, ""

    if tokens[0] in {"rg", "find", "wc", "sort", "grep"}:
        return True, ""

    if tokens[0] in {"python", "python3"}:
        return False, "blocked: use `uv run --no-project python <script>` (uv-managed venv required)"

    if tokens[0] == "uv":
        # Allow a minimal subset: uv run python[3] src/...
        if len(tokens) < 4 or tokens[1] != "run":
            return False, "blocked: only 'uv run ...' allowed"
        # Skip optional flags like --with <pkg>
        j = 2
        while j < len(tokens) and tokens[j].startswith("-"):
            if tokens[j] in {"--with"}:
                j += 2
                continue
            if tokens[j] in {"--no-sync"}:
                j += 1
                continue
            if tokens[j] in {"--no-project", "--active", "--isolated", "--no-env-file"}:
                j += 1
                continue
            return False, f"blocked: uv flag not allowed ({tokens[j]})"
        if j >= len(tokens):
            return False, "blocked: uv run missing command"
        inner_tokens = tokens[j:]
        return _validate_inner_tokens(inner_tokens, cfg)

    if tokens[0] == "git":
        if len(tokens) < 2:
            return False, "blocked: git subcommand required"
        sub = tokens[1]
        allowed_git = {
            "status",
            "diff",
            "log",
            "rev-parse",
            "show",
            "checkout",
            "switch",
            "add",
            "commit",
            "apply",
        }
        if sub not in allowed_git:
            return False, f"blocked: git subcommand not allowed ({sub})"
        # Disallow destructive flags.
        if any(x in tokens for x in ["--hard", "--force", "-f"]):
            return False, "blocked: destructive git flags not allowed"
        return True, ""

    return False, "blocked: unknown command"


def _validate_inner_tokens(tokens: list[str], cfg: AgentConfig) -> tuple[bool, str]:
    if not tokens:
        return False, "empty command"

    # For uv-run we allow python as the inner command.
    if tokens[0] in {"python", "python3"}:
        if len(tokens) < 2:
            return False, "python requires a script path or -c"
        # Allow python -c "..." for inline scripts
        if tokens[1] == "-c":
            return True, ""
        if tokens[1].startswith("-"):
            return False, "blocked: python flags not allowed (use a script path or -c)"
        script = _resolve_under_repo(tokens[1], cfg.repo_root)
        if not script.exists() or not script.is_file():
            return False, f"script not found: {script}"
        if not str(script).endswith(".py"):
            return False, "blocked: only .py scripts allowed"
        if not _is_within(script, cfg.repo_root):
            return False, "blocked: python script must be under repo root"
        return True, ""

    # Everything else must still pass the top-level allowlist.
    return validate_cmd(" ".join(tokens), cfg)


ToolName = Literal[
    "repo_list",
    "repo_read",
    "repo_search",
    "run_cmd",
    "web_search_papers",
    "web_fetch",
    "write_text",
    "append_text",
    "apply_unified_diff",
    "make_patch",
    "git_branch_and_commit",
]


@dataclass
class ToolContext:
    cfg: AgentConfig
    memory: RunMemory
    touched_paths: set[Path]


Handler = Callable[[ToolContext, dict[str, Any]], dict[str, Any]]


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "repo_list",
                "description": "List a repo directory (non-recursive).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_entries": {"type": "integer", "default": 200},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "repo_read",
                "description": "Read a UTF-8 text file from the repo (truncated).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_chars": {"type": "integer", "default": 6000},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "repo_search",
                "description": "Search text in the repo (ripgrep).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "max_results": {"type": "integer", "default": 50},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_cmd",
                "description": "Run a safe shell command in the repo root (allowlisted).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "timeout_s": {"type": "integer"},
                    },
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search_papers",
                "description": "Search scholarly papers via OpenAlex (requires --allow-web).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10},
                        "from_publication_date": {"type": "string", "default": "2020-01-01"},
                        "sort": {"type": "string", "default": "publication_date:desc,cited_by_count:desc"},
                        "require_abstract": {"type": "boolean", "default": True},
                        "require_oa": {"type": "boolean", "default": False},
                        "require_pdf": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch a URL (text/HTML) and save an auditable artifact (requires --allow-web).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "timeout_s": {"type": "integer", "default": 20},
                        "max_bytes": {"type": "integer", "default": 2000000},
                        "extract_text": {"type": "boolean", "default": True},
                        "max_chars": {"type": "integer", "default": 8000},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_text",
                "description": "Write a UTF-8 text file (docs/ and run_dir/ by default).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_text",
                "description": "Append to a UTF-8 text file (docs/ and run_dir/ by default).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_unified_diff",
                "description": "Apply a unified diff via 'git apply' (restricted to docs/ and run_dir/ by default).",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "make_patch",
                "description": "Write a git patch of current changes to runs/<run_id>/artifacts/changes.patch.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "out_path": {"type": "string", "default": ""},
                        "paths": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_branch_and_commit",
                "description": "Create/switch to a branch and commit specified paths (docs/ by default).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "branch": {"type": "string"},
                        "message": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "required": ["branch", "message"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def handlers() -> dict[str, Handler]:
    return {
        "repo_list": _repo_list,
        "repo_read": _repo_read,
        "repo_search": _repo_search,
        "run_cmd": _run_cmd,
        "web_search_papers": _web_search_papers,
        "web_fetch": _web_fetch,
        "write_text": _write_text,
        "append_text": _append_text,
        "apply_unified_diff": _apply_unified_diff,
        "make_patch": _make_patch,
        "git_branch_and_commit": _git_branch_and_commit,
    }


def _repo_list(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under_repo(str(args.get("path", ".")), ctx.cfg.repo_root)
    max_entries = int(args.get("max_entries") or 200)
    if not _is_within(path, ctx.cfg.repo_root):
        return {"ok": False, "error": f"path outside repo: {path}"}
    if not path.exists() or not path.is_dir():
        return {"ok": False, "error": f"not a directory: {path}"}
    items = []
    for p in sorted(path.iterdir(), key=lambda x: x.name)[:max_entries]:
        items.append({"name": p.name, "type": "dir" if p.is_dir() else "file"})
    return {"ok": True, "path": str(path), "items": items, "truncated": len(list(path.iterdir())) > max_entries}


def _repo_read(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under_repo(str(args.get("path")), ctx.cfg.repo_root)
    max_chars = int(args.get("max_chars") or 6000)
    if not _is_within(path, ctx.cfg.repo_root):
        return {"ok": False, "error": f"path outside repo: {path}"}
    if _blocked_read_path(path):
        return {"ok": False, "error": "blocked: cannot read .env"}
    if not path.exists() or not path.is_file():
        return {"ok": False, "error": f"file not found: {path}"}
    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    rel = os.path.relpath(path, ctx.cfg.repo_root)
    return {"ok": True, "path": rel, "sha256": _sha256_text(text), "truncated": truncated, "content": text}


def _repo_search(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    pattern = str(args.get("pattern") or "")
    path_str = str(args.get("path") or ".")
    max_results = int(args.get("max_results") or 50)
    root = _resolve_under_repo(path_str, ctx.cfg.repo_root)
    if not _is_within(root, ctx.cfg.repo_root):
        return {"ok": False, "error": f"path outside repo: {root}"}
    if shutil.which("rg") is None:
        return {"ok": False, "error": "rg not found in PATH"}
    cmd = ["rg", "-n", pattern, str(root)]
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(ctx.cfg.repo_root))
    out = (proc.stdout or "").strip().splitlines()
    return {
        "ok": proc.returncode in (0, 1),
        "pattern": pattern,
        "path": str(root),
        "results": out[:max_results],
        "truncated": len(out) > max_results,
        "stderr_tail": tail(proc.stderr or "", 1000),
    }


def _run_cmd(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    cmd = str(args.get("cmd") or "").strip()
    # Strip unnecessary "cd <path> && " prefix - run_cmd already runs in repo_root
    import re as _re
    cmd = _re.sub(r"^cd\s+\S+\s*&&\s*", "", cmd).strip()
    timeout_s = args.get("timeout_s")
    if timeout_s is None:
        timeout_s = ctx.cfg.timeout_s
    ok, reason = validate_cmd(cmd, ctx.cfg)
    if not ok:
        ctx.memory.log_tool_call({"type": "run_cmd", "cmd": cmd, "ok": False, "error": reason})
        return {"ok": False, "error": reason}

    try:
        res = run_shell(cmd, cwd=ctx.cfg.repo_root, timeout_s=int(timeout_s) if timeout_s else None)
    except Exception as exc:
        ctx.memory.log_tool_call({"type": "run_cmd", "cmd": cmd, "ok": False, "error": str(exc)})
        return {"ok": False, "error": str(exc)}

    name = f"cmd_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    out_path = _write_artifact_text(ctx.cfg.run_dir, name, (res.stdout or "") + ("\n\n" + (res.stderr or "") if res.stderr else ""))
    record = {
        "type": "run_cmd",
        "cmd": cmd,
        "ok": res.returncode == 0,
        "returncode": res.returncode,
        "stdout_tail": tail(res.stdout, 2000),
        "stderr_tail": tail(res.stderr, 2000),
        "artifact": str(out_path),
        "started_at": res.started_at,
        "ended_at": res.ended_at,
        "duration_s": res.duration_s,
    }
    ctx.memory.log_tool_call(record)
    return {"ok": record["ok"], "returncode": res.returncode, "stdout_tail": record["stdout_tail"], "stderr_tail": record["stderr_tail"], "artifact": str(out_path)}


def _web_search_papers(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if not ctx.cfg.allow_web:
        return {"ok": False, "error": "web tools disabled (pass --allow-web)"}

    query = str(args.get("query") or "").strip()
    if not query:
        return {"ok": False, "error": "query required"}

    max_results = int(args.get("max_results") or 10)
    from_date = str(args.get("from_publication_date") or "").strip() or None
    sort = str(args.get("sort") or "publication_date:desc,cited_by_count:desc").strip()
    require_abstract = bool(args.get("require_abstract", True))
    require_oa = bool(args.get("require_oa", False))
    require_pdf = bool(args.get("require_pdf", False))

    try:
        out = openalex_search_works(
            query,
            max_results=max_results,
            from_publication_date=from_date,
            sort=sort,
            require_abstract=require_abstract,
            require_oa=require_oa,
            require_pdf=require_pdf,
        )
    except Exception as exc:
        ctx.memory.log_tool_call({"type": "web_search_papers", "ok": False, "query": query, "error": str(exc)})
        return {"ok": False, "error": str(exc)}

    artifacts_dir = (ctx.cfg.run_dir / "artifacts" / "web").resolve()
    ensure_dir(artifacts_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact = artifacts_dir / f"openalex_{ts}.json"
    artifact.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ctx.memory.log_tool_call(
        {
            "type": "web_search_papers",
            "ok": True,
            "query": query,
            "count": len(out.get("results") or []),
            "artifact": str(artifact),
            "url": out.get("url"),
            "status": out.get("status"),
            "truncated": out.get("truncated"),
        }
    )
    return {"ok": True, "query": query, "count": len(out.get("results") or []), "artifact": str(artifact), "results": out.get("results") or []}


def _web_fetch(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if not ctx.cfg.allow_web:
        return {"ok": False, "error": "web tools disabled (pass --allow-web)"}

    url = str(args.get("url") or "").strip()
    if not url:
        return {"ok": False, "error": "url required"}
    timeout_s = float(args.get("timeout_s") or 20)
    max_bytes = int(args.get("max_bytes") or 2_000_000)
    extract_text = bool(args.get("extract_text", True))
    max_chars = int(args.get("max_chars") or 8000)

    ok_url, reason = validate_http_url(url)
    if not ok_url:
        ctx.memory.log_tool_call({"type": "web_fetch", "ok": False, "url": url, "error": reason})
        return {"ok": False, "error": reason}

    try:
        res = http_get(url, timeout_s=timeout_s, max_bytes=max_bytes)
    except Exception as exc:
        ctx.memory.log_tool_call({"type": "web_fetch", "ok": False, "url": url, "error": str(exc)})
        return {"ok": False, "error": str(exc)}

    ctype = (res.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    ext = "bin"
    if "html" in ctype:
        ext = "html"
    elif "json" in ctype:
        ext = "json"
    elif "xml" in ctype:
        ext = "xml"
    elif ctype.startswith("text/"):
        ext = "txt"
    elif "pdf" in ctype or url.lower().endswith(".pdf"):
        ext = "pdf"

    artifacts_dir = (ctx.cfg.run_dir / "artifacts" / "web").resolve()
    ensure_dir(artifacts_dir)
    sha = sha256_bytes(res.body)[:16]
    raw_path = artifacts_dir / f"fetch_{sha}.{ext}"
    raw_path.write_bytes(res.body)

    out: dict[str, Any] = {
        "ok": True,
        "url": url,
        "status": res.status,
        "content_type": ctype,
        "bytes": len(res.body),
        "truncated": res.truncated,
        "artifact": str(raw_path),
    }

    text_preview = ""
    text_artifact = None
    if extract_text and (ctype.startswith("text/") or "html" in ctype or "json" in ctype or "xml" in ctype):
        # best-effort decode
        charset = None
        ct_full = res.headers.get("Content-Type") or ""
        m = re.search(r"charset=([A-Za-z0-9_\\-]+)", ct_full)
        if m:
            charset = m.group(1)
        decoded = res.body.decode(charset or "utf-8", errors="replace")
        if "html" in ctype:
            decoded = html_to_text(decoded)
        text_preview = decoded[:max_chars]
        text_artifact = artifacts_dir / f"fetch_{sha}.txt"
        text_artifact.write_text(decoded, encoding="utf-8")
        out["text_preview"] = text_preview
        out["text_artifact"] = str(text_artifact)

    ctx.memory.log_tool_call(
        {
            "type": "web_fetch",
            "ok": True,
            "url": url,
            "status": res.status,
            "content_type": ctype,
            "bytes": len(res.body),
            "truncated": res.truncated,
            "artifact": str(raw_path),
            "text_artifact": str(text_artifact) if text_artifact else "",
        }
    )
    return out


def _write_text(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under_repo(str(args.get("path")), ctx.cfg.repo_root)
    content = str(args.get("content") or "")
    ok, reason = _check_write_path(path, ctx.cfg)
    if not ok:
        ctx.memory.log_tool_call({"type": "write_text", "path": str(path), "ok": False, "error": reason})
        return {"ok": False, "error": reason}

    old = ""
    existed = path.exists()
    if existed and path.is_file():
        old = path.read_text(encoding="utf-8", errors="replace")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    ctx.touched_paths.add(path)

    rel = os.path.relpath(path, ctx.cfg.repo_root)
    diff = _diff_text(old, content, rel)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    diff_path = _write_artifact_text(ctx.cfg.run_dir, f"diff_{ts}_{Path(rel).name}.patch", diff)

    ctx.memory.log_tool_call(
        {
            "type": "write_text",
            "path": rel,
            "ok": True,
            "existed": existed,
            "bytes_before": len(old.encode("utf-8", errors="ignore")),
            "bytes_after": len(content.encode("utf-8", errors="ignore")),
            "diff_artifact": str(diff_path),
        }
    )
    return {"ok": True, "path": rel, "diff_artifact": str(diff_path)}


def _append_text(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under_repo(str(args.get("path")), ctx.cfg.repo_root)
    content = str(args.get("content") or "")
    ok, reason = _check_write_path(path, ctx.cfg)
    if not ok:
        ctx.memory.log_tool_call({"type": "append_text", "path": str(path), "ok": False, "error": reason})
        return {"ok": False, "error": reason}

    old = ""
    if path.exists() and path.is_file():
        old = path.read_text(encoding="utf-8", errors="replace")

    new = old + content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new, encoding="utf-8")
    ctx.touched_paths.add(path)

    rel = os.path.relpath(path, ctx.cfg.repo_root)
    diff = _diff_text(old, new, rel)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    diff_path = _write_artifact_text(ctx.cfg.run_dir, f"diff_{ts}_{Path(rel).name}.patch", diff)
    ctx.memory.log_tool_call(
        {
            "type": "append_text",
            "path": rel,
            "ok": True,
            "bytes_before": len(old.encode("utf-8", errors="ignore")),
            "bytes_after": len(new.encode("utf-8", errors="ignore")),
            "diff_artifact": str(diff_path),
        }
    )
    return {"ok": True, "path": rel, "diff_artifact": str(diff_path)}


def _apply_unified_diff(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    patch = str(args.get("patch") or "")
    ok, reason, touched = _validate_git_apply_paths(patch, ctx.cfg)
    if not ok:
        ctx.memory.log_tool_call({"type": "apply_unified_diff", "ok": False, "error": reason})
        return {"ok": False, "error": reason}

    patch_path = _write_artifact_text(ctx.cfg.run_dir, f"apply_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.patch", patch)
    check = run_shell(f"git apply --check {shlex.quote(str(patch_path))}", cwd=ctx.cfg.repo_root, timeout_s=60)
    if check.returncode != 0:
        ctx.memory.log_tool_call(
            {
                "type": "apply_unified_diff",
                "ok": False,
                "error": "git apply --check failed",
                "stderr_tail": tail(check.stderr, 2000),
                "patch_artifact": str(patch_path),
                "touched": touched,
            }
        )
        return {"ok": False, "error": "git apply --check failed", "stderr_tail": tail(check.stderr, 2000)}

    apply_res = run_shell(f"git apply {shlex.quote(str(patch_path))}", cwd=ctx.cfg.repo_root, timeout_s=60)
    ok2 = apply_res.returncode == 0
    if ok2:
        for p_str in touched:
            p = _resolve_under_repo(p_str, ctx.cfg.repo_root)
            if _is_within(p, ctx.cfg.repo_root):
                ctx.touched_paths.add(p)
    ctx.memory.log_tool_call(
        {
            "type": "apply_unified_diff",
            "ok": ok2,
            "returncode": apply_res.returncode,
            "stderr_tail": tail(apply_res.stderr, 2000),
            "patch_artifact": str(patch_path),
            "touched": touched,
        }
    )
    return {"ok": ok2, "touched": touched, "patch_artifact": str(patch_path), "stderr_tail": tail(apply_res.stderr, 1000)}


def _make_patch(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    out_path = str(args.get("out_path") or "").strip()
    if out_path:
        path = _resolve_under_repo(out_path, ctx.cfg.repo_root)
        ok, reason = _check_write_path(path, ctx.cfg)
        if not ok:
            return {"ok": False, "error": reason}
    else:
        path = (ctx.cfg.run_dir / "artifacts" / "changes.patch").resolve()

    diff_parts: list[str] = []
    paths = args.get("paths") or []
    if not isinstance(paths, list):
        paths = []
    path_filters = [str(p) for p in paths if str(p)]

    diff_cmd = ["git", "diff"]
    if path_filters:
        diff_cmd += ["--"] + path_filters
    diff = subprocess.run(diff_cmd, text=True, capture_output=True, cwd=str(ctx.cfg.repo_root))
    diff_parts.append(diff.stdout or "")

    # Include untracked files via --no-index (best-effort).
    # Use ls-files to get full untracked file list (git status may collapse dirs).
    untracked_out = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        text=True,
        capture_output=True,
        cwd=str(ctx.cfg.repo_root),
    )
    untracked = []
    for p in (untracked_out.stdout or "").splitlines():
        p = p.strip()
        if not p:
            continue
        if p == ".env" or p.endswith("/.env"):
            continue
        if path_filters:
            if not any(p == f or p.startswith(f.rstrip("/") + "/") for f in path_filters):
                continue
        untracked.append(p)
    for p in untracked:
        proc = subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", p],
            text=True,
            capture_output=True,
            cwd=str(ctx.cfg.repo_root),
        )
        if proc.stdout:
            diff_parts.append(proc.stdout)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(diff_parts), encoding="utf-8")

    ctx.memory.log_tool_call(
        {
            "type": "make_patch",
            "ok": True,
            "path": str(path),
            "bytes": path.stat().st_size,
            "untracked_included": len(untracked),
            "paths": path_filters,
        }
    )
    return {"ok": True, "path": str(path), "untracked_included": len(untracked), "paths": path_filters}


def _git_branch_and_commit(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    branch = str(args.get("branch") or "").strip()
    message = str(args.get("message") or "").strip()
    paths = args.get("paths") or []
    if not branch or not message:
        return {"ok": False, "error": "branch and message required"}

    if not isinstance(paths, list):
        return {"ok": False, "error": "paths must be a list"}

    if not paths:
        paths = ["docs"]

    resolved = []
    for p_str in paths:
        p = _resolve_under_repo(str(p_str), ctx.cfg.repo_root)
        ok, reason = _check_write_path(p, ctx.cfg)
        if not ok:
            return {"ok": False, "error": reason}
        resolved.append(os.path.relpath(p, ctx.cfg.repo_root))

    cmds = [
        f"git checkout -b {shlex.quote(branch)}",
        "git status --porcelain",
        "git add " + " ".join(shlex.quote(p) for p in resolved),
        f"git commit -m {shlex.quote(message)}",
        "git rev-parse HEAD",
    ]
    out = []
    sha = ""
    for c in cmds:
        ok_cmd, reason = validate_cmd(c, ctx.cfg)
        if not ok_cmd:
            return {"ok": False, "error": reason}
        res = run_shell(c, cwd=ctx.cfg.repo_root, timeout_s=60)
        out.append({"cmd": c, "returncode": res.returncode, "stdout_tail": tail(res.stdout, 1200), "stderr_tail": tail(res.stderr, 1200)})
        if res.returncode != 0:
            ctx.memory.log_tool_call({"type": "git_branch_and_commit", "ok": False, "branch": branch, "message": message, "error": f"cmd failed: {c}", "detail": out})
            return {"ok": False, "error": f"cmd failed: {c}", "detail": out}
        if c == "git rev-parse HEAD":
            sha = (res.stdout or "").strip()

    ctx.memory.log_tool_call({"type": "git_branch_and_commit", "ok": True, "branch": branch, "message": message, "sha": sha, "paths": resolved})
    return {"ok": True, "sha": sha, "branch": branch}


# -- Daemon bridge -------------------------------------------------------


def make_agent_config_from_daemon(
    daemon_config: Any,
    run_dir: Path,
    run_id: str,
) -> "AgentConfig":
    """Create an AgentConfig from a DaemonConfig so existing tool handlers work."""
    from .config import AgentConfig, budget_profile

    return AgentConfig(
        repo_root=daemon_config.repo_root,
        run_dir=run_dir,
        run_id=run_id,
        goal=daemon_config.topic,
        data_dir=daemon_config.data_dir,
        model=daemon_config.inference_model,
        forecast_model=daemon_config.inference_model,
        budget=budget_profile("standard"),
        approval_mode="auto",
        git_policy="none",
        allow_web=daemon_config.allow_web,
        allow_code_edit=daemon_config.allow_code_edit,
        paper_path=daemon_config.data_dir / "papers",
        timeout_s=120,
    )
