"""Minimal authenticated client for the official Build Small MiniCPM API.

The hackathon endpoint is OpenAI-compatible. The API key is read only from
``OPENBMB_API_KEY`` and is never included in errors, artifacts, or logs.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import requests

MINICPM_MODEL_ID = "MiniCPM4.1-8B"
MINICPM_HF_REPO = "openbmb/MiniCPM4.1-8B"
# OpenBMB published this shared hackathon endpoint as HTTP-only. The bearer
# credential is the shared challenge token, not a personal Hugging Face token.
# Keep the URL configurable so an HTTPS endpoint can replace it without code
# changes if the sponsor provides one.
DEFAULT_BASE_URL = "http://35.203.155.71:8001"
DEFAULT_TIMEOUT_S = 120


def _base_url() -> str:
    return os.environ.get("OPENBMB_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _headers() -> dict[str, str]:
    token = os.environ.get("OPENBMB_API_KEY", "").strip()
    if not token:
        raise EnvironmentError(
            "MiniCPM requires the OPENBMB_API_KEY environment variable."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if detail:
            return str(detail)[:500]
    return str(payload)[:500]


def _redact_secret(value: str) -> str:
    token = os.environ.get("OPENBMB_API_KEY", "").strip()
    return value.replace(token, "[redacted]") if token else value


def chat(
    messages: Sequence[dict[str, str]],
    *,
    max_tokens: int = 220,
    temperature: float = 0.0,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Generate one MiniCPM chat completion."""
    payload = {
        "model": MINICPM_MODEL_ID,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(
        f"{_base_url()}/v1/chat/completions",
        headers=_headers(),
        json=payload,
        timeout=timeout_s,
    )
    if not 200 <= response.status_code < 300:
        raise RuntimeError(
            f"OpenBMB endpoint error ({response.status_code}): "
            f"{_redact_secret(_response_detail(response))}"
        )
    data = response.json()
    try:
        text = str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenBMB endpoint returned an invalid response.") from exc
    return {
        "text": text,
        "model": str(data.get("model") or MINICPM_MODEL_ID),
        "system_fingerprint": data.get("system_fingerprint"),
        "usage": data.get("usage"),
    }


def batch_chat(
    message_batches: Sequence[Sequence[dict[str, str]]],
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> list[str]:
    """Generate an ordered batch of MiniCPM chat completions."""
    if not message_batches:
        return []
    payload = {
        "model": MINICPM_MODEL_ID,
        "messages": [list(messages) for messages in message_batches],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(
        f"{_base_url()}/v1/chat/completions/batch",
        headers=_headers(),
        json=payload,
        timeout=timeout_s,
    )
    if not 200 <= response.status_code < 300:
        raise RuntimeError(
            f"OpenBMB batch endpoint error ({response.status_code}): "
            f"{_redact_secret(_response_detail(response))}"
        )
    data = response.json()
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != len(message_batches):
        raise RuntimeError("OpenBMB batch endpoint returned an invalid response.")
    try:
        indices = [int(choice["index"]) for choice in choices]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "OpenBMB batch endpoint returned invalid choice indices."
        ) from exc
    if sorted(indices) != list(range(len(message_batches))):
        raise RuntimeError(
            "OpenBMB batch endpoint returned duplicate or missing choice indices."
        )
    ordered = sorted(choices, key=lambda choice: int(choice["index"]))
    try:
        return [
            str(choice["message"]["content"]).strip()
            for choice in ordered
        ]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("OpenBMB batch endpoint returned an invalid choice.") from exc
