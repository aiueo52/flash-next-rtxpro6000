from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar


class MockSSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    token_count: ClassVar[int] = 6
    token_interval: ClassVar[float] = 0.025

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": "mock-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics":
            body = b"requests_total 1\nspec_accepted_tokens_total 12\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(content_length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for index in range(self.token_count):
            if index:
                time.sleep(self.token_interval)
            self._event(
                {
                    "choices": [
                        {"index": 0, "delta": {"content": chr(ord("a") + index)}}
                    ]
                }
            )
        self._event(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": self.token_count,
                    "total_tokens": 10 + self.token_count,
                },
                "timings": {
                    "prompt_ms": 4.0,
                    "predicted_ms": self.token_interval * self.token_count * 1000,
                    "predicted_per_second": 1 / self.token_interval,
                },
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _event(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value).encode()
        self.wfile.write(b"data: " + encoded + b"\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextmanager
def mock_sse_server() -> Iterator[tuple[str, type[MockSSEHandler]]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockSSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1", MockSSEHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
