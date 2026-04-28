#!/usr/bin/env python3

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket


class JsonHandler(BaseHTTPRequestHandler):
    server_version = "SimpleJsonServer/1.0"

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send_json(
            200,
            {
                "status": "ok",
                "message": "hello from python on " + socket.gethostname(),
                "path": self.path,
            },
        )

    def do_POST(self) -> None:
        self._send_json(
            200,
            {
                "status": "ok",
                "message": "post received",
                "path": self.path,
            },
        )

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a simple JSON HTTP server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=12345, help="Port to listen on.")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), JsonHandler)
    print(f"Serving JSON on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
