#!/usr/bin/env python3
"""
Agentic baseline runner (Codex CLI or Claude Code CLI).

Creates per-sample work dirs containing payload.json and asks the agent to
produce a JSON forecast. Tools/web are allowed by the agent CLI config.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "forecast": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "lead_hours": {"type": "number"},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "wind": {"type": "number"},
                },
                "required": ["lead_hours", "lat", "lon", "wind"],
            },
            "minItems": 3,
        },
        "reasoning": {"type": "string"},
    },
    "required": ["forecast", "reasoning"],
}

JSON_SCHEMA_BATCH = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer"},
                    "forecast": JSON_SCHEMA["properties"]["forecast"],
                    "reasoning": {"type": "string"},
                },
                "required": ["index", "forecast", "reasoning"],
            },
        }
    },
    "required": ["outputs"],
}


def load_payloads(path: Path, start: int, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    # direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # try to extract the first JSON object block
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
    return None


def build_prompt() -> str:
    return (
        "Read payload.json in the current directory.\n"
        "If you need to read the file, use shell commands (e.g., `cat payload.json`).\n"
        "Do not use MCP file tools (they may be unavailable).\n"
        "You may use tools and web search if helpful.\n"
        "Task: Predict center lat/lon and max wind at 24h, 48h, 72h.\n"
        "Return json only in this exact schema:\n"
        "{\n"
        "  \"forecast\": [\n"
        "    {\"lead_hours\": 24, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>},\n"
        "    {\"lead_hours\": 48, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>},\n"
        "    {\"lead_hours\": 72, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>}\n"
        "  ],\n"
        "  \"reasoning\": \"<brief>\"\n"
        "}\n"
    )

def build_batch_prompt(payload_files: list[str]) -> str:
    file_list = "\n".join(f"- {name}" for name in payload_files)
    return (
        "Read the following payload files in the current directory:\n"
        f"{file_list}\n"
        "If you need to read files, use shell commands (e.g., `cat payload_000.json`).\n"
        "Do not use MCP file tools (they may be unavailable).\n"
        "You may use tools and web search if helpful.\n"
        "Task: For each payload, predict center lat/lon and max wind at 24h, 48h, 72h.\n"
        "Return json only with this exact schema:\n"
        "{\n"
        "  \"outputs\": [\n"
        "    {\n"
        "      \"index\": <int>,\n"
        "      \"forecast\": [\n"
        "        {\"lead_hours\": 24, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>},\n"
        "        {\"lead_hours\": 48, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>},\n"
        "        {\"lead_hours\": 72, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>}\n"
        "      ],\n"
        "      \"reasoning\": \"<brief>\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Use index=0 for the first file in the list, 1 for the second, etc.\n"
    )


def normalize_reasoning(value: str) -> str:
    allowed = {"low", "medium", "high"}
    if value not in allowed:
        return "high"
    return value


def codex_exec_help_text() -> str:
    try:
        return subprocess.check_output(["codex", "exec", "--help"], text=True, timeout=10)
    except Exception:
        return ""


def codex_supports(help_text: str, flag: str) -> bool:
    return flag in help_text


def run_codex(sample_dir: Path, model: str, reasoning: str, timeout: int, help_text: str) -> str:
    reasoning = normalize_reasoning(reasoning)
    cmd = ["codex", "exec"]
    if codex_supports(help_text, "--full-auto"):
        cmd.append("--full-auto")
    elif codex_supports(help_text, "--approval-mode"):
        cmd += ["--approval-mode", "full-auto"]
    if codex_supports(help_text, "--no-project-doc"):
        cmd.append("--no-project-doc")
    if codex_supports(help_text, "--writable-root"):
        cmd += ["--writable-root", str(sample_dir)]
    if codex_supports(help_text, "-m") or codex_supports(help_text, "--model"):
        cmd += ["-m", model]
    if codex_supports(help_text, "--reasoning"):
        cmd += ["--reasoning", reasoning]
    if codex_supports(help_text, "--output-schema"):
        schema_path = (sample_dir / "schema.json").resolve()
        schema_path.write_text(json.dumps(JSON_SCHEMA, indent=2), encoding="utf-8")
        cmd += ["--output-schema", str(schema_path)]
    cmd.append(build_prompt())
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
        cwd=sample_dir,
    )
    return proc.stdout


def run_claude(sample_dir: Path, model: str, timeout: int, schema: dict, prompt: str) -> str:
    schema_str = json.dumps(schema)
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        model,
        "--add-dir",
        str(sample_dir),
        "--json-schema",
        schema_str,
    ]
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
        cwd=sample_dir,
    )
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agentic baselines via Codex/Claude CLIs.")
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["codex", "claude"], required=True)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--reasoning", type=str, default="xhigh")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for Claude (>=1).")
    parser.add_argument("--work-dir", type=Path, default=Path("sim_outputs_agent/work"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    payloads = load_payloads(args.payloads, args.start, args.limit)
    if not payloads:
        raise SystemExit("No payloads found.")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = args.work_dir
    work_root.mkdir(parents=True, exist_ok=True)

    if args.backend == "codex":
        model = args.model or "gpt-5.2-codex"
        reasoning_used = normalize_reasoning(args.reasoning)
        codex_help = codex_exec_help_text()
    else:
        model = args.model or "claude-opus-4-5-20251101"
        reasoning_used = None
        codex_help = ""

    with out_path.open("w", encoding="utf-8") as f_out:
        if args.backend == "claude" and args.batch_size > 1:
            batch_size = max(1, args.batch_size)
            for bstart in range(0, len(payloads), batch_size):
                batch = payloads[bstart : bstart + batch_size]
                sample_dir = work_root / f"batch_{args.start + bstart:06d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                payload_files = []
                for j, payload in enumerate(batch):
                    name = f"payload_{j:03d}.json"
                    payload_files.append(name)
                    (sample_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

                try:
                    raw = run_claude(
                        sample_dir,
                        model,
                        args.timeout,
                        JSON_SCHEMA_BATCH,
                        build_batch_prompt(payload_files),
                    )
                    parsed_outputs = {}
                    try:
                        meta = json.loads(raw)
                        structured = meta.get("structured_output") if isinstance(meta, dict) else None
                        if isinstance(structured, dict):
                            for item in structured.get("outputs", []):
                                if isinstance(item, dict) and "index" in item:
                                    parsed_outputs[int(item["index"])] = item
                    except json.JSONDecodeError:
                        pass
                except subprocess.TimeoutExpired:
                    raw = ""
                    parsed_outputs = {}
                except subprocess.CalledProcessError as exc:
                    raw = exc.output or ""
                    parsed_outputs = {}

                for j in range(len(batch)):
                    item = parsed_outputs.get(j)
                    ok = isinstance(item, dict) and "forecast" in item
                    record = {
                        "backend": args.backend,
                        "model": model,
                        "reasoning": reasoning_used,
                        "content": raw.strip(),
                        "parsed": item if ok else None,
                        "valid_json": ok,
                    }
                    f_out.write(json.dumps(record) + "\n")

                if not args.keep_work:
                    try:
                        for child in sample_dir.iterdir():
                            child.unlink()
                        sample_dir.rmdir()
                    except Exception:
                        pass

                done = min(bstart + batch_size, len(payloads))
                print(f"{args.backend}: {done}/{len(payloads)}")
        else:
            for i, payload in enumerate(payloads):
                sample_dir = work_root / f"sample_{args.start + i:06d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                payload_path = sample_dir / "payload.json"
                payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                try:
                    if args.backend == "codex":
                        raw = run_codex(sample_dir, model, args.reasoning, args.timeout, codex_help)
                        parsed = extract_json(raw)
                        ok = parsed is not None
                    else:
                        raw = run_claude(sample_dir, model, args.timeout, JSON_SCHEMA, build_prompt())
                        parsed = None
                        ok = False
                        try:
                            meta = json.loads(raw)
                            if isinstance(meta, dict):
                                if "structured_output" in meta and isinstance(meta["structured_output"], dict):
                                    parsed = meta["structured_output"]
                                    ok = True
                                elif isinstance(meta.get("result"), str) and meta["result"].strip():
                                    parsed = extract_json(meta["result"])
                                    ok = parsed is not None
                        except json.JSONDecodeError:
                            parsed = extract_json(raw)
                            ok = parsed is not None
                except subprocess.TimeoutExpired:
                    raw = ""
                    parsed = None
                    ok = False
                except subprocess.CalledProcessError as exc:
                    raw = exc.output or ""
                    parsed = None
                    ok = False

                record = {
                    "backend": args.backend,
                    "model": model,
                    "reasoning": reasoning_used,
                    "content": raw.strip(),
                    "parsed": parsed,
                    "valid_json": ok,
                }
                f_out.write(json.dumps(record) + "\n")

                if not args.keep_work:
                    try:
                        for child in sample_dir.iterdir():
                            child.unlink()
                        sample_dir.rmdir()
                    except Exception:
                        pass

                if (i + 1) % 10 == 0 or i == len(payloads) - 1:
                    print(f"{args.backend}: {i + 1}/{len(payloads)}")


if __name__ == "__main__":
    main()
