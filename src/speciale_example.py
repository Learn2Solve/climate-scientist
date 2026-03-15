#!/usr/bin/env python3
"""
Minimal scripts to call DeepSeek-V3.2-Speciale.

This file shows:
  1) A generic \"hello world\" Speciale call.
  2) A hurricane-forecast style call reusing the prompt from forecaster.py.

Requirements:
  - pip install -U openai
  - export DEEPSEEK_API_KEY=sk-...
"""

from __future__ import annotations

import argparse
import json
import os
from textwrap import dedent

from openai import OpenAI

from forecaster import (
    SPECIALE_BASE_URL,
    build_prompt as hurricane_build_prompt,
    demo_payload as hurricane_demo_payload,
    load_env_key,
)


SPECIAL_BASE_URL = SPECIALE_BASE_URL
SPECIAL_MODEL = "deepseek-reasoner"


def client_from_env(base_url: str) -> OpenAI:
    api_key = load_env_key()
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in your environment or .env file.")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_speciale_simple(question: str) -> None:
    """
    Simple demo: send a question to Speciale and print the answer + reasoning.
    """
    client = client_from_env(SPECIAL_BASE_URL)
    system_prompt = (
        "You are DeepSeek-V3.2-Speciale, a strong reasoning model. "
        "Think step by step, then give a concise final answer."
    )

    resp = client.chat.completions.create(
        model=SPECIAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        # NOTE: Speciale endpoint does not support JSON mode (response_format).
        max_tokens=4096,
    )

    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    reasoning = (getattr(choice.message, "reasoning_content", "") or "").strip()

    print("----- ANSWER -----")
    print(content or "[empty content]")

    if reasoning:
        print("\n----- REASONING -----")
        print(reasoning)


def call_speciale_hurricane(use_demo_payload: bool = True) -> None:
    """
    Hurricane-style Speciale call reusing the existing hurricane prompt builder.
    Uses the same structure as forecaster.py but hits the
    Speciale endpoint and does not request JSON mode.
    """
    client = client_from_env(SPECIAL_BASE_URL)

    if use_demo_payload:
        storm, env, ls, analogs, guidance = hurricane_demo_payload()
    else:
        raise SystemExit("Non-demo payload wiring not implemented in this mini script.")

    prompt = hurricane_build_prompt(storm, env, ls, analogs, guidance)

    system_message = (
        "You are a concise, numerically precise hurricane forecaster. "
        "Think carefully, but keep the final answer short."
    )

    resp = client.chat.completions.create(
        model=SPECIAL_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        max_tokens=12000,
    )

    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    reasoning = (getattr(choice.message, "reasoning_content", "") or "").strip()

    print("----- MODEL OUTPUT -----")
    print(content or "[empty content]")

    if reasoning:
        print("\n----- MODEL REASONING -----")
        print(reasoning)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-V3.2-Speciale example scripts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_simple = sub.add_parser("simple", help="Simple Speciale QA demo.")
    p_simple.add_argument("question", help="Question to send to Speciale.")

    p_hurr = sub.add_parser("hurricane-demo", help="Hurricane-style forecast using the demo payload.")

    args = parser.parse_args()

    if args.cmd == "simple":
        call_speciale_simple(args.question)
    elif args.cmd == "hurricane-demo":
        call_speciale_hurricane(use_demo_payload=True)
    else:
        parser.error(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
