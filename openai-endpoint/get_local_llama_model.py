#!/usr/bin/env python3
"""Get model information from a local OpenAI-compatible LLaMA endpoint."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "http://10.10.1.91:11235"


def api_request(path: str, payload: dict | None = None) -> dict:
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}{path}"

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON in server response") from exc


def list_models() -> list[str]:
    payload = api_request("/v1/models")
    models = payload.get("data", [])
    ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
    if not ids:
        raise RuntimeError(f"No models found in response: {payload}")
    return ids


def detect_model_via_chat(model: str) -> str:
    payload = api_request(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one word: ok"}],
            "max_tokens": 4,
        },
    )
    used_model = payload.get("model")
    if not used_model:
        raise RuntimeError(f"Could not determine used model from response: {payload}")
    return used_model


def main() -> int:
    try:
        model_ids = list_models()
    except RuntimeError as exc:
        print(f"Failed to fetch models: {exc}", file=sys.stderr)
        return 1

    print("Available model(s):")
    for model_id in model_ids:
        print(f"- {model_id}")

    probe_model = os.getenv("LOCAL_LLAMA_MODEL", model_ids[0])
    try:
        used_model = detect_model_via_chat(probe_model)
    except RuntimeError as exc:
        print(f"\nCould not verify active response model via chat probe: {exc}")
        return 0

    print(f"\nModel reported by chat/completions: {used_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
