#!/usr/bin/env python3
"""
Quick MVP to call DeepSeek-V3.2 as an autonomous hurricane forecaster.

Goal: zero-shot / few-shot LLM forecasts (track + intensity + reasoning) with structured
text inputs. No finetuning; just prompt engineering plus curated context.
"""

from __future__ import annotations

import argparse
import json
import os
from textwrap import dedent
from typing import Iterable, Mapping, Sequence
from pathlib import Path

from openai import OpenAI


# Default to official DeepSeek API with the reasoning model.
DEFAULT_BASE_URL = "https://api.deepseek.com"
# Temporary endpoint for DeepSeek-V3.2-Speciale (thinking-only, no tool calls).
# Per DeepSeek docs, the published endpoint expired on 2025-12-15; override via env if needed.
SPECIALE_BASE_URL = os.environ.get(
    "DEEPSEEK_SPECIALE_BASE_URL",
    "https://api.deepseek.com/v3.2_speciale_expires_on_20251215",
)
DEFAULT_MODEL = "deepseek-reasoner"


def load_env_key() -> str | None:
    """Load DEEPSEEK_API_KEY from env or a local .env file."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key

    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "DEEPSEEK_API_KEY" and v.strip():
                    key = v.strip()
                    os.environ["DEEPSEEK_API_KEY"] = key  # set for OpenAI client
                    return key
    return None


def format_lat_lon(lat: object, lon: object) -> str:
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return f"{lat}, {lon}"
    hemi_lat = "N" if lat_f >= 0 else "S"
    hemi_lon = "E" if lon_f >= 0 else "W"
    return f"{abs(lat_f):.2f}{hemi_lat}, {abs(lon_f):.2f}{hemi_lon}"


def build_prompt(
    storm: Mapping[str, str | float],
    environment: Mapping[str, str | float],
    large_scale: Mapping[str, str | float],
    analogs: Sequence[Mapping[str, str | float]],
    guidance: Iterable[str] | None = None,
) -> str:
    """Turn structured scalars into a forecaster-style prompt."""
    env_lines = "\n".join(f"- {k}: {v}" for k, v in environment.items())
    ls_lines = "\n".join(f"- {k}: {v}" for k, v in large_scale.items())
    analog_lines = "\n".join(
        f"- {a.get('name', 'Analog')}: {a.get('summary', '')}"
        for a in analogs
    )
    guidance_block = ""
    if guidance:
        guidance_block = "External guidance:\n" + "\n".join(f"- {g}" for g in guidance) + "\n\n"

    prompt = dedent(
        f"""
        You are an expert hurricane forecaster. Produce structured 24h/48h/72h predictions
        and a concise reasoning paragraph.

        Storm state:
        - ID: {storm.get('id')}
        - Basin: {storm.get('basin')}
        - Valid time: {storm.get('time')}
        - Center: {format_lat_lon(storm.get('lat'), storm.get('lon'))}
        - Intensity: {storm.get('wind')} kt, {storm.get('pressure')} hPa
        - Motion: {storm.get('motion')}

        Environment (local):
        {env_lines}

        Large-scale pattern:
        {ls_lines}

        Historical analogs (similar environment + position):
        {analog_lines}

        {guidance_block}Task:
        1) Predict center lat/lon (deg) and max wind (kt) at 24h, 48h, 72h.
        2) Provide one paragraph of reasoning referencing steering, shear, thermodynamics, and analog behavior.

        Respond in json (a single JSON object):
        {{
          "forecast": [
            {{"lead_hours": 24, "lat": <float>, "lon": <float>, "wind": <float>}},
            {{"lead_hours": 48, "lat": <float>, "lon": <float>, "wind": <float>}},
            {{"lead_hours": 72, "lat": <float>, "lon": <float>, "wind": <float>}}
          ],
          "reasoning": "<concise text>"
        }}
        """
    ).strip()
    return prompt


def call_model(
    prompt: str,
    *,
    base_url: str,
    model: str,
    system_message: str,
    temperature: float = 0.4,
    max_tokens: int = 800,
    timeout: float | None = None,
    model_kwargs: dict | None = None,
) -> object:
    """Call DeepSeek and return the raw response object."""
    api_key = load_env_key()
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in your environment.")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    extra = model_kwargs or {}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
    except Exception as exc:  # broad to surface server-side issues clearly
        # Speciale endpoint sometimes expects a trailing /v1; retry once if missing.
        alt_url = None
        if not base_url.rstrip("/").endswith("v1"):
            alt_url = base_url.rstrip("/") + "/v1"
        if alt_url:
            client_alt = OpenAI(api_key=api_key, base_url=alt_url, timeout=timeout)
            try:
                resp = client_alt.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )
                return resp
            except Exception:
                pass
        raise SystemExit(f"API call failed: {exc}\n"
                         f"(base_url tried: {base_url}"
                         f"{' and ' + alt_url if alt_url else ''}, model={model})") from exc
    return resp


def demo_payload() -> tuple[Mapping[str, str | float], Mapping[str, str | float], Mapping[str, str | float], list[Mapping[str, str | float]], list[str]]:
    """Toy inputs to exercise the prompt; replace with real data hookups."""
    storm = {
        "id": "AL09",
        "basin": "Atlantic",
        "time": "2021-09-10 12Z",
        "lat": 24.3,
        "lon": -68.2,
        "wind": 95,
        "pressure": 965,
        "motion": "305 deg at 9 kt",
    }
    environment = {
        "SST": "29.5 C (high OHC)",
        "Vertical wind shear (200-850 hPa)": "8 kt from WSW (low)",
        "Mid-level RH (600-800 hPa)": "70%",
        "MPI": "155 kt",
        "Upper-level divergence": "favorable",
    }
    large_scale = {
        "Subtropical ridge": "axis SW-NE, centered NE of storm",
        "Trough": "digging along US East Coast, ~1500 km away, strong amplitude",
        "Steering flow (850-500 hPa mean)": "WNW at 10-15 kt",
    }
    analogs = [
        {"name": "Analog 1: Hurricane Fran (1996)", "summary": "Similar position with approaching trough; recurved north of Bahamas then into Carolinas; maintained cat 3 before landfall."},
        {"name": "Analog 2: Hurricane Floyd (1999)", "summary": "Strong ridge then trough-induced recurve; rapid intensification over 29.5C SST, shear low."},
    ]
    guidance = [
        "Persistence 24h: 25.5N 70.0W, 95 kt",
        "Climatology track bend after 36h toward NNE given trough timing",
    ]
    return storm, environment, large_scale, analogs, guidance


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V3.2 hurricane forecaster MVP (no finetune).")
    parser.add_argument("--preview", action="store_true", help="Only print the prompt instead of calling the API.")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for the response. For deepseek-chat the valid range is [1, 8192]; deepseek-reasoner supports larger values.",
    )
    parser.add_argument("--base-url", type=str, default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
                        help="Override DeepSeek base URL (env DEEPSEEK_BASE_URL also supported).")
    parser.add_argument("--model", type=str, default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
                        help="Override model name (env DEEPSEEK_MODEL also supported).")
    parser.add_argument(
        "--speciale",
        action="store_true",
        help=(
            "Use DeepSeek-V3.2-Speciale endpoint "
            f"({SPECIALE_BASE_URL}; thinking-only, no tool calls; per docs the published endpoint expired on 2025-12-15). "
            "Ignores --base-url (use DEEPSEEK_SPECIALE_BASE_URL to override)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Request JSON mode (response_format=json_object).")
    parser.add_argument("--reasoning-limit", type=int, default=120,
                        help="Soft limit for reasoning tokens; included in system prompt.")
    parser.add_argument(
        "--payload-json",
        type=Path,
        help="Path to JSON file with keys: storm, environment, large_scale, analogs, guidance.",
    )
    args = parser.parse_args()

    if args.speciale and args.model == "deepseek-chat":
        parser.error("DeepSeek-V3.2-Speciale is thinking-only; use deepseek-reasoner model.")
    if args.speciale:
        args.base_url = SPECIALE_BASE_URL

    if args.payload_json:
        with args.payload_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        storm = payload.get("storm", {})
        env = payload.get("environment", {})
        ls = payload.get("large_scale", {})
        analogs = payload.get("analogs", []) or []
        guidance = payload.get("guidance", []) or []
    else:
        storm, env, ls, analogs, guidance = demo_payload()
    prompt = build_prompt(storm, env, ls, analogs, guidance)

    system_message = (
        "You are a concise, numerically precise hurricane forecaster. "
        f"Limit chain-of-thought to <= {args.reasoning_limit} tokens. "
        "Always return the JSON exactly per the requested schema."
    )
    model_kwargs = {}
    if args.json:
        if args.speciale:
            # Speciale endpoint currently does not support server-side JSON mode.
            print(
                "[warn] --json requested with --speciale, but the V3.2-Speciale "
                "endpoint does not support response_format=json_object; "
                "will only enforce JSON via the prompt.",
                flush=True,
            )
        else:
            model_kwargs["response_format"] = {"type": "json_object"}

    if args.preview:
        print("----- PROMPT -----")
        print(prompt)
        return

    # Clamp max_tokens for chat model to API limits.
    effective_max_tokens = args.max_tokens
    if args.model == "deepseek-chat" and effective_max_tokens > 8192:
        effective_max_tokens = 8192

    resp = call_model(
        prompt,
        base_url=args.base_url,
        model=args.model,
        system_message=system_message,
        temperature=args.temperature,
        max_tokens=effective_max_tokens,
        model_kwargs=model_kwargs,
    )
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    reasoning = (getattr(choice.message, "reasoning_content", "") or "").strip()
    finish_reason = choice.finish_reason

    print("----- MODEL OUTPUT -----")
    if content:
        print(content)
    else:
        print("[empty content from model]")

    if reasoning:
        print("\n----- MODEL REASONING -----")
        print(reasoning)

    usage = getattr(resp, "usage", None)
    if usage:
        print("\n----- TOKEN USAGE -----")
        print(json.dumps(usage.model_dump(), indent=2))

    # Optional: try to parse JSON block if the model obeys.
    try:
        parsed = json.loads(content)
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        if finish_reason == "length":
            print("\n[warn] finish_reason=length: increase --max-tokens (e.g., 8192) or simplify prompt.")
        else:
            print("\n[warn] could not parse JSON; check content above.")


if __name__ == "__main__":
    main()
