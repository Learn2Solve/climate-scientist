#!/usr/bin/env python3
"""
Run the LLM forecaster over a JSONL payload file and write JSONL outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forecaster import SPECIALE_BASE_URL, build_prompt, call_model, demo_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch LLM forecaster over JSONL payloads.")
    parser.add_argument("--payloads", type=Path, required=True, help="JSONL payloads (one per line).")
    parser.add_argument("--out", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--model", type=str, default="deepseek-reasoner")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--speciale", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=None, help="Per-request timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Request server-side JSON mode if supported.")
    parser.add_argument("--compact-prompt", action="store_true", help="Use a shorter prompt (useful for reasoner).")
    parser.add_argument("--tiny-prompt", action="store_true", help="Use an ultra-short prompt for smoke tests.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    if args.speciale:
        args.base_url = SPECIALE_BASE_URL
    if args.model == "deepseek-reasoner":
        if not args.json:
            args.json = True
        if args.max_tokens < 4096:
            args.max_tokens = 4096

    system_message = (
        "You are a concise, numerically precise hurricane forecaster. "
        "Always return the json exactly per the requested schema."
    )

    model_kwargs = {}
    if args.json and not args.speciale:
        model_kwargs["response_format"] = {"type": "json_object"}

    total = 0
    written = 0

    with args.payloads.open("r", encoding="utf-8") as f_in, args.out.open("w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            if idx < args.start:
                continue
            if args.limit and written >= args.limit:
                break
            total += 1
            payload = json.loads(line)
            storm = payload.get("storm", {})
            env = payload.get("environment", {})
            ls = payload.get("large_scale", {})
            analogs = payload.get("analogs", []) or []
            guidance = payload.get("guidance", []) or []

            if args.tiny_prompt:
                prompt = (
                    "Return json only.\n"
                    "Storm:\n"
                    f"- ID: {storm.get('id')}\n"
                    f"- Basin: {storm.get('basin')}\n"
                    f"- Valid time: {storm.get('time')}\n"
                    f"- Center: {storm.get('lat')}, {storm.get('lon')}\n"
                    f"- Intensity: {storm.get('wind')} kt, {storm.get('pressure')} hPa\n"
                    f"- Motion: {storm.get('motion')}\n\n"
                    "Task: Predict lat/lon/wind at 24h, 48h, 72h. Keep reasoning brief.\n"
                    "Respond in json:\n"
                    "{\n"
                    "  \"forecast\": [\n"
                    "    {\"lead_hours\": 24, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>},\n"
                    "    {\"lead_hours\": 48, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>},\n"
                    "    {\"lead_hours\": 72, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>}\n"
                    "  ],\n"
                    "  \"reasoning\": \"<concise text>\"\n"
                    "}\n"
                )
            elif args.compact_prompt:
                env_lines = "\n".join(f"- {k}: {v}" for k, v in env.items())
                prompt = (
                    "Return json only.\n"
                    "Storm state:\n"
                    f"- ID: {storm.get('id')}\n"
                    f"- Basin: {storm.get('basin')}\n"
                    f"- Valid time: {storm.get('time')}\n"
                    f"- Center: {storm.get('lat')}, {storm.get('lon')}\n"
                    f"- Intensity: {storm.get('wind')} kt, {storm.get('pressure')} hPa\n"
                    f"- Motion: {storm.get('motion')}\n\n"
                    "Environment:\n"
                    f"{env_lines}\n\n"
                    "Task: Predict lat/lon/wind at 24h, 48h, 72h.\n"
                    "Respond in json:\n"
                    "{\n"
                    "  \"forecast\": [\n"
                    "    {\"lead_hours\": 24, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>},\n"
                    "    {\"lead_hours\": 48, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>},\n"
                    "    {\"lead_hours\": 72, \"lat\": <float>, \"lon\": <float>, \"wind\": <float>}\n"
                    "  ],\n"
                    "  \"reasoning\": \"<concise text>\"\n"
                    "}\n"
                )
            else:
                prompt = build_prompt(storm, env, ls, analogs, guidance)
            try:
                resp = call_model(
                    prompt,
                    base_url=args.base_url,
                    model=args.model,
                    system_message=system_message,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    model_kwargs=model_kwargs,
                )
                choice = resp.choices[0]
                content = (choice.message.content or "").strip()
                reasoning = (getattr(choice.message, "reasoning_content", "") or "").strip()
            except SystemExit as exc:
                record = {"error": str(exc), "valid_json": False}
                f_out.write(json.dumps(record) + "\n")
                written += 1
                continue
            except Exception as exc:
                record = {"error": f"call_model failed: {exc}", "valid_json": False}
                f_out.write(json.dumps(record) + "\n")
                written += 1
                continue

            record = {"content": content}
            if reasoning:
                record["reasoning"] = reasoning

            # Try to parse JSON if possible
            try:
                record["parsed"] = json.loads(content)
                record["valid_json"] = True
            except json.JSONDecodeError:
                if reasoning:
                    try:
                        record["parsed"] = json.loads(reasoning)
                        record["valid_json"] = True
                    except json.JSONDecodeError:
                        record["valid_json"] = False
                else:
                    record["valid_json"] = False

            f_out.write(json.dumps(record) + "\n")
            written += 1

    print(f"Wrote {written} predictions to {args.out} (processed {total} rows)")


if __name__ == "__main__":
    main()
