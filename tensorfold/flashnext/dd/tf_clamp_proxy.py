#!/usr/bin/env python3
"""Sampling clamp between the hotschmoe-dd frontdoor and a greedy-only TensorFold server.

TensorFold's GGUF TP2 XPU engine decodes greedily only and refuses any request with temperature > 0
(server.zig resolveSampling). Strata accepted sampled requests, so clients that send temperature
would break at cutover. This proxy listens on 127.0.0.1:18124 (the frontdoor's backend) and forwards
to TensorFold on an owner-only Unix socket, rewriting generation requests:
  - temperature > 0            -> 0
  - top_p, top_k, min_p, typical_p, seed  -> dropped (ignored by greedy decoding)
  - presence/frequency/repetition penalties with non-default values -> dropped (engine has none)
Nothing about request or response content is logged. Counters only, at GET /_clamp/stats.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("CLAMP_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLAMP_PORT", "18124"))
SOCK = os.environ.get(
    "CLAMP_BACKEND_SOCKET",
    os.path.join(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"), "hotschmoe-dd-tf", "tf.sock"),
)
TIMEOUT = float(os.environ.get("CLAMP_BACKEND_TIMEOUT_S", "7200"))
GEN_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages",
             "/chat/completions", "/completions"}
DROP_ALWAYS = ("top_p", "top_k", "min_p", "typical_p", "seed")
PENALTY_DEFAULTS = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
       "transfer-encoding", "upgrade"}

STATS_LOCK = threading.Lock()
STATS: dict[str, object] = {"started": time.time(), "requests": 0, "generation_requests": 0,
                            "temperature_clamped": 0, "dropped": {}, "status": {}, "upstream_errors": 0,
                            "bad_json": 0}


def bump(key: str, sub: str | None = None) -> None:
    with STATS_LOCK:
        if sub is None:
            STATS[key] = int(STATS[key]) + 1
        else:
            d = STATS[key]
            d[sub] = d.get(sub, 0) + 1


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("127.0.0.1", 18124, timeout=timeout)
        self._path = path

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._path)
        self.sock = s


def is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def clamp(body: dict) -> bool:
    changed = False
    t = body.get("temperature")
    if is_number(t) and t > 0:
        body["temperature"] = 0
        bump("temperature_clamped")
        changed = True
    for k in DROP_ALWAYS:
        if k in body:
            del body[k]
            bump("dropped", k)
            changed = True
    for k, default in PENALTY_DEFAULTS.items():
        if k in body and not (is_number(body[k]) and float(body[k]) == default):
            del body[k]
            bump("dropped", k)
            changed = True
    return changed


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "b70-tf-clamp/1"

    def log_message(self, fmt: str, *args: object) -> None:  # no per-request logging
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/_clamp/stats":
            with STATS_LOCK:
                payload = json.dumps(STATS, separators=(",", ":")).encode("ascii")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._proxy()

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = lambda self: self._proxy()  # noqa: E731

    def _proxy(self) -> None:
        bump("requests")
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else None
        path = self.path.split("?", 1)[0]
        if body and self.command == "POST" and path in GEN_PATHS:
            bump("generation_requests")
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and clamp(parsed):
                    body = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            except (ValueError, UnicodeDecodeError):
                bump("bad_json")  # forward unchanged; the server answers with its own error
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "content-length"}
        headers["Host"] = "127.0.0.1:18124"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn = UnixHTTPConnection(SOCK, TIMEOUT)
        started = False
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            bump("status", str(resp.status))
            self.send_response(resp.status, resp.reason)
            has_length = False
            for k, v in resp.getheaders():
                if k.lower() in HOP:
                    continue
                if k.lower() == "content-length":
                    has_length = True
                self.send_header(k, v)
            if not has_length:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            started = True
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (ConnectionError, TimeoutError, OSError, http.client.HTTPException) as e:
            bump("upstream_errors")
            sys.stdout.write(f"tf-clamp: upstream error {type(e).__name__}\n")
            sys.stdout.flush()
            if not started:
                msg = json.dumps({"error": {"message": "backend unavailable", "type": "upstream_error"}}).encode()
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                except OSError:
                    pass
            self.close_connection = True
        finally:
            conn.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"tf-clamp: listening on {HOST}:{PORT} -> unix:{SOCK}", flush=True)
    Server((HOST, PORT), Handler).serve_forever()
