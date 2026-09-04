#!/usr/bin/env python3
"""Small API-key reverse proxy for a loopback OpenAI-compatible server."""

from __future__ import annotations

import hmac
import http.client
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


HOST = os.environ.get("FRONTDOOR_HOST", "0.0.0.0")
PORT = int(os.environ.get("FRONTDOOR_PORT", "18080"))
BACKEND = urlsplit(
    os.environ.get("FRONTDOOR_BACKEND_URL", "http://127.0.0.1:18124")
)
KEY_FILE = os.environ.get("FRONTDOOR_API_KEY_FILE", "")
TIMEOUT = float(os.environ.get("FRONTDOOR_BACKEND_TIMEOUT_S", "7200"))

if BACKEND.scheme != "http" or not BACKEND.hostname or not BACKEND.port:
    raise SystemExit("FRONTDOOR_BACKEND_URL must be an http URL with a port")
if not KEY_FILE:
    raise SystemExit("FRONTDOOR_API_KEY_FILE is required")
with open(KEY_FILE, encoding="ascii") as handle:
    API_KEY = handle.readline().strip()
if not API_KEY:
    raise SystemExit("FRONTDOOR_API_KEY_FILE is empty")

HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "b70-openai-frontdoor/1"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stdout.write("frontdoor: " + (fmt % args) + "\n")
        sys.stdout.flush()

    def _authorized(self) -> bool:
        bearer = self.headers.get("Authorization", "")
        x_api_key = self.headers.get("X-API-Key", "")
        return hmac.compare_digest(bearer, "Bearer " + API_KEY) or hmac.compare_digest(
            x_api_key, API_KEY
        )

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("ascii")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy()

    def _proxy(self) -> None:
        public_path = self.path.split("?", 1)[0]
        if public_path not in {"/health", "/metrics"} and not self._authorized():
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Type", "application/json")
            body = b'{"error":{"message":"Unauthorized","type":"authentication_error"}}'
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_HEADERS
            and key.lower() not in {"host", "authorization", "x-api-key"}
        }
        headers["Host"] = f"{BACKEND.hostname}:{BACKEND.port}"
        headers["X-Forwarded-For"] = self.client_address[0]

        connection = http.client.HTTPConnection(
            BACKEND.hostname, BACKEND.port, timeout=TIMEOUT
        )
        response_started = False
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            has_length = False
            for key, value in response.getheaders():
                lower = key.lower()
                if lower in HOP_HEADERS:
                    continue
                if lower == "content-length":
                    has_length = True
                self.send_header(key, value)
            self.send_header("Access-Control-Allow-Origin", "*")
            if not has_length:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            response_started = True
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (ConnectionError, TimeoutError, http.client.HTTPException) as error:
            if not response_started and not self.wfile.closed:
                self._send_json(502, {"error": {"message": str(error), "type": "upstream_error"}})
            else:
                self.close_connection = True
        finally:
            connection.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


print(
    f"frontdoor: listening on {HOST}:{PORT} -> {BACKEND.hostname}:{BACKEND.port}",
    flush=True,
)
Server((HOST, PORT), Handler).serve_forever()
