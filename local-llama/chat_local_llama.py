#!/usr/bin/env python3
"""Send user input to a local LLaMA server exposing an OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "http://10.10.1.91:11235"
DEFAULT_MODEL = os.getenv("LOCAL_LLAMA_MODEL", "llama3")


def send_prompt(prompt: str, model: str) -> tuple[str, str]:
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    # Older
    url = f"{base_url}/v1/chat/completions"
    # Newer
    # url = f"{base_url}/v1/responses"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "input": prompt,
    }
    data = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response: {response_body}") from exc

    # Typical Responses API shape.
    output_text = parsed.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return "output_text", output_text

    # Fallback for raw structured responses.
    output = parsed.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "output", "".join(chunks)

    # Compatibility fallback in case the server still returns chat shape.
    try:
        return "chat_shape_response", parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response format: {response_body}") from exc


def main() -> int:
    model = os.getenv("LOCAL_LLAMA_MODEL", DEFAULT_MODEL)
    prompt = input("Enter your prompt: ").strip()

    if not prompt:
        print("Prompt cannot be empty.", file=sys.stderr)
        return 1

    try:
        response_key, answer = send_prompt(prompt, model=model)
    except RuntimeError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nResponse key: {response_key}")
    print("LLaMA:")
    print(answer.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
