#!/usr/bin/env python3

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def fetch_models(base_url: str, api_key: str, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List models from an OpenAI-compatible v1 endpoint."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Base API URL (default: OPENAI_BASE_URL or https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (default: OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON response instead of model IDs only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("Error: API key is required. Use --api-key or set OPENAI_API_KEY.", file=sys.stderr)
        return 2

    try:
        payload = fetch_models(args.base_url, args.api_key, args.timeout)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP error {e.code}: {error_body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        return 1
    except TimeoutError:
        print("Request timed out.", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    data = payload.get("data", [])
    if not isinstance(data, list):
        print("Unexpected response format: 'data' is not a list.", file=sys.stderr)
        return 1

    for item in data:
        model_id = item.get("id")
        if model_id is not None:
            print(model_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
