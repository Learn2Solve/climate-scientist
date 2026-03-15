"""
Inference client - routes chat completions to OpenAI or Anthropic backends.

Port of automaton's TypeScript inference client, adapted for Python with
native SDK usage instead of raw HTTP.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class InferenceResponse:
    id: str
    model: str
    content: str
    tool_calls: list[dict] | None
    usage: dict
    finish_reason: str


class InferenceClient:
    def __init__(
        self,
        default_model: str,
        max_tokens: int,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        low_compute_model: str = "gpt-4.1-mini",
        cli_proxy_url: str | None = None,
        cli_proxy_api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._default_model = default_model
        self._default_max_tokens = max_tokens
        self._current_model = default_model
        self._current_max_tokens = max_tokens
        self._openai_api_key = openai_api_key
        self._anthropic_api_key = anthropic_api_key
        self._low_compute_model = low_compute_model
        self._cli_proxy_url = cli_proxy_url
        self._cli_proxy_api_key = cli_proxy_api_key
        self._reasoning_effort = reasoning_effort

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> InferenceResponse:
        model = model or self._current_model
        token_limit = max_tokens or self._current_max_tokens
        backend = self._resolve_backend(model)

        if backend == "cliproxy":
            return self._chat_cliproxy(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=token_limit,
                temperature=temperature,
            )
        elif backend == "anthropic":
            return self._chat_anthropic(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=token_limit,
                temperature=temperature,
            )
        elif backend == "openai":
            return self._chat_openai(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=token_limit,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def set_low_compute_mode(self, enabled: bool) -> None:
        if enabled:
            self._current_model = self._low_compute_model
            self._current_max_tokens = 4096
        else:
            self._current_model = self._default_model
            self._current_max_tokens = self._default_max_tokens

    def get_default_model(self) -> str:
        return self._current_model

    # -- Backend resolution --------------------------------------------------

    @property
    def using_proxy(self) -> bool:
        return bool(self._cli_proxy_url)

    def _resolve_backend(self, model: str) -> str:
        # CLIProxy handles all models - both GPT and Claude via OpenAI-compatible endpoint
        if self._cli_proxy_url:
            return "cliproxy"
        if self._anthropic_api_key and re.match(r"^claude", model, re.IGNORECASE):
            return "anthropic"
        if self._openai_api_key and re.match(
            r"^(gpt|o[1-9]|o3|o4|chatgpt)", model, re.IGNORECASE
        ):
            return "openai"
        raise ValueError(
            f"No API key available for model '{model}'. "
            "Provide openai_api_key or anthropic_api_key."
        )

    # -- OpenAI backend ------------------------------------------------------

    def _chat_openai(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float | None,
    ) -> InferenceResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self._openai_api_key)

        # Newer models use max_completion_tokens instead of max_tokens
        uses_completion_tokens = bool(re.match(r"^(o[1-9]|gpt-5|gpt-4\.1)", model))

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if uses_completion_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        if temperature is not None:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message

        tool_calls: list[dict] | None = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage_data = response.usage
        usage = {
            "prompt_tokens": usage_data.prompt_tokens if usage_data else 0,
            "completion_tokens": usage_data.completion_tokens if usage_data else 0,
            "total_tokens": usage_data.total_tokens if usage_data else 0,
        }

        return InferenceResponse(
            id=response.id or "",
            model=response.model or model,
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
        )

    # -- CLIProxy backend (native OpenAI-compatible tool calling) ---------------

    def _chat_cliproxy(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float | None,
    ) -> InferenceResponse:
        """Route through CLIProxyAPI's OpenAI-compatible /v1/chat/completions.

        CLIProxyAPI (router-for-me/CLIProxyAPI) supports native function calling
        via the standard OpenAI tools interface - no text-mode workarounds needed.
        """
        import httpx
        from openai import OpenAI

        client = OpenAI(
            api_key=self._cli_proxy_api_key or "automaton-local-key",
            base_url=f"{self._cli_proxy_url.rstrip('/')}/v1",
            timeout=httpx.Timeout(600.0, connect=10.0),
            max_retries=0,
        )

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        extra_body: dict = {}
        if self._reasoning_effort:
            extra_body["reasoning"] = {"effort": self._reasoning_effort}

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs, extra_body=extra_body or None)

        choice = response.choices[0]
        message = choice.message

        tool_calls: list[dict] | None = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage_data = response.usage
        usage = {
            "prompt_tokens": usage_data.prompt_tokens if usage_data else 0,
            "completion_tokens": usage_data.completion_tokens if usage_data else 0,
            "total_tokens": usage_data.total_tokens if usage_data else 0,
        }

        return InferenceResponse(
            id=response.id or "",
            model=response.model or model,
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
        )

    # -- Anthropic backend ---------------------------------------------------

    def _chat_anthropic(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float | None,
    ) -> InferenceResponse:
        import anthropic

        client = anthropic.Anthropic(api_key=self._anthropic_api_key)

        transformed = _transform_messages_for_anthropic(messages)

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": transformed["messages"]
            if transformed["messages"]
            else [{"role": "user", "content": "Continue."}],
        }

        if transformed["system"]:
            kwargs["system"] = transformed["system"]

        if temperature is not None:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"].get("parameters", {}),
                }
                for tool in tools
            ]
            kwargs["tool_choice"] = {"type": "auto"}

        response = client.messages.create(**kwargs)

        content_blocks = response.content if isinstance(response.content, list) else []
        text_blocks = [b for b in content_blocks if getattr(b, "type", None) == "text"]
        tool_use_blocks = [
            b for b in content_blocks if getattr(b, "type", None) == "tool_use"
        ]

        text_content = "\n".join(
            getattr(b, "text", "") for b in text_blocks
        ).strip()

        tool_calls: list[dict] | None = None
        if tool_use_blocks:
            tool_calls = [
                {
                    "id": getattr(b, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(b, "name", ""),
                        "arguments": json.dumps(getattr(b, "input", {})),
                    },
                }
                for b in tool_use_blocks
            ]

        if not text_content and not tool_calls:
            raise ValueError("No completion content returned from Anthropic inference")

        input_tokens = getattr(response.usage, "input_tokens", 0) if response.usage else 0
        output_tokens = getattr(response.usage, "output_tokens", 0) if response.usage else 0
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        finish_reason = _normalize_anthropic_finish_reason(response.stop_reason)

        return InferenceResponse(
            id=response.id or "",
            model=response.model or model,
            content=text_content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )


# -- Anthropic message transformation ----------------------------------------


def _transform_messages_for_anthropic(
    messages: list[dict],
) -> dict:
    """Transform OpenAI-format messages to Anthropic message format.

    - System messages are extracted to a top-level system parameter.
    - Assistant messages with tool_calls become content block arrays.
    - Tool role messages become user messages with tool_result content blocks.
    """
    system_parts: list[str] = []
    transformed: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            content = msg.get("content", "")
            if content:
                system_parts.append(content)
            continue

        if role == "user":
            transformed.append({
                "role": "user",
                "content": msg.get("content", ""),
            })
            continue

        if role == "assistant":
            content_blocks: list[dict] = []
            text = msg.get("content", "")
            if text:
                content_blocks.append({"type": "text", "text": text})

            for tc in msg.get("tool_calls", []) or []:
                func = tc.get("function", {})
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": _parse_tool_arguments(func.get("arguments", "{}")),
                })

            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})

            transformed.append({
                "role": "assistant",
                "content": content_blocks,
            })
            continue

        if role == "tool":
            transformed.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", "unknown_tool_call"),
                        "content": msg.get("content", ""),
                    },
                ],
            })

    return {
        "system": "\n\n".join(system_parts) if system_parts else None,
        "messages": transformed,
    }


def _parse_tool_arguments(raw: str) -> dict:
    """Parse tool arguments JSON string, with fallback for malformed input."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}


def _normalize_anthropic_finish_reason(reason: str | None) -> str:
    """Map Anthropic stop_reason to OpenAI-compatible finish_reason."""
    if not isinstance(reason, str):
        return "stop"
    if reason == "tool_use":
        return "tool_calls"
    return reason


