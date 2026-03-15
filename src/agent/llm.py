from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


def load_openai_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    env_path = ".env"
    if os.path.exists(env_path):
        for line in open(env_path, "r", encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "OPENAI_API_KEY" and v.strip():
                key = v.strip()
                os.environ["OPENAI_API_KEY"] = key
                return key
    return None


@dataclass(frozen=True)
class LlmConfig:
    model: str = "gpt-5.2"
    temperature: float = 0.2
    max_tokens: int = 2000


def make_client() -> OpenAI:
    # OpenAI client picks up OPENAI_API_KEY / OPENAI_BASE_URL from env.
    _ = load_openai_key()
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(base_url=base_url)
    return OpenAI()


def safe_tool_result(obj: Any) -> str:
    # Tool messages must be strings; keep them compact.
    import json

    try:
        text = json.dumps(obj, ensure_ascii=False)
    except Exception:
        text = str(obj)
    if len(text) > 6000:
        return text[:6000] + "...(truncated)"
    return text

