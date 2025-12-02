#!/usr/bin/env python3
"""
Quick MVP to call DeepSeek-V3.2-Speciale as an autonomous hurricane forecaster.

Goal: zero-shot / few-shot LLM forecasts (track + intensity + reasoning) with structured
text inputs. No finetuning; just prompt engineering plus curated context.
"""

from __future__ import annotations

import argparse
import json
import os
from textwrap import dedent
from typing import Iterable, Mapping, Sequence

from openai import OpenAI


# DeepSeek-V3.2-Speciale (thinking-only). Expires 2025-12-15 15:59 UTC per docs.
DEFAULT_BASE_URL = "https://api.deepseek.com/v3.2_speciale_expires_on_20251215"
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
        - Center: {storm.get('lat')}N, {storm.get('lon')}W
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

        Respond in JSON:
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
    temperature: float = 0.4,
    max_tokens: int = 800,
) -> str:
    """Call DeepSeek-V3.2-Speciale and return the raw content string."""
    api_key = load_env_key()
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in your environment.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise, numerically precise hurricane forecaster."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # broad to surface server-side issues clearly
        # Speciale endpoint sometimes expects a trailing /v1; retry once if missing.
        alt_url = None
        if not base_url.rstrip("/").endswith("v1"):
            alt_url = base_url.rstrip("/") + "/v1"
        if alt_url:
            client_alt = OpenAI(api_key=api_key, base_url=alt_url)
            try:
                resp = client_alt.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a concise, numerically precise hurricane forecaster."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception:
                pass
        raise SystemExit(f"API call failed: {exc}\n"
                         f"(base_url tried: {base_url}"
                         f"{' and ' + alt_url if alt_url else ''}, model={model})") from exc
    return resp.choices[0].message.content


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
    parser = argparse.ArgumentParser(description="DeepSeek-V3.2-Speciale hurricane forecaster MVP (no finetune).")
    parser.add_argument("--preview", action="store_true", help="Only print the prompt instead of calling the API.")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--base-url", type=str, default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
                        help="Override DeepSeek base URL (env DEEPSEEK_BASE_URL also supported).")
    parser.add_argument("--model", type=str, default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
                        help="Override model name (env DEEPSEEK_MODEL also supported).")
    args = parser.parse_args()

    storm, env, ls, analogs, guidance = demo_payload()
    prompt = build_prompt(storm, env, ls, analogs, guidance)

    if args.preview:
        print("----- PROMPT -----")
        print(prompt)
        return

    output = call_model(
        prompt,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print("----- MODEL OUTPUT -----")
    print(output)

    # Optional: try to parse JSON block if the model obeys.
    try:
        parsed = json.loads(output)
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        pass


if __name__ == "__main__":
    main()
