#!/usr/bin/env python3
"""Capture and validate Pi 0.84.3 Qwen request payloads without a GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from evals.terminalbench.harbor_pi import qwen_model_definition


class CaptureHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.requests.append(payload)
        body = (
            'data: {"id":"oracle","object":"chat.completion.chunk",'
            '"created":0,"model":"oracle","choices":[{"index":0,'
            '"delta":{"role":"assistant","content":"ok"},'
            '"finish_reason":null}]}\n\n'
            'data: {"id":"oracle","object":"chat.completion.chunk",'
            '"created":0,"model":"oracle","choices":[{"index":0,'
            '"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        ).encode("ascii")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


def run_pi(pi_binary: Path, level: str, base_url: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"b70-pi-{level}-") as temp:
        root = Path(temp)
        config = {
            "providers": {
                "harbor-endpoint": {
                    "baseUrl": base_url,
                    "apiKey": "$OPENAI_API_KEY",
                    "api": "openai-completions",
                    "compat": {
                        "supportsStore": False,
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                        "supportsUsageInStreaming": True,
                        "supportsFinishReason": True,
                        "maxTokensField": "max_tokens",
                        "thinkingFormat": "qwen-chat-template",
                        "supportsStrictMode": False,
                    },
                    "models": [
                        qwen_model_definition(
                            "oracle", context_window=4096, max_tokens=1024
                        )
                    ],
                }
            }
        }
        (root / "models.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        env = os.environ.copy()
        env.update(
            {
                "OPENAI_API_KEY": "oracle-key",
                "PI_CODING_AGENT_DIR": str(root),
                "PI_OFFLINE": "1",
                "PI_TELEMETRY": "0",
            }
        )
        before = len(CaptureHandler.requests)
        result = subprocess.run(
            [
                str(pi_binary),
                "--print",
                "--mode",
                "json",
                "--no-session",
                "--no-tools",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
                "--no-context-files",
                "--offline",
                "--provider",
                "harbor-endpoint",
                "--model",
                "oracle",
                "--thinking",
                level,
                "Reply with ok.",
            ],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Pi {level} oracle failed: {result.stdout[-2000:]}")
        captured = CaptureHandler.requests[before:]
        if len(captured) != 1:
            raise RuntimeError(f"Pi {level} emitted {len(captured)} requests")
        return captured[0]


def assert_payload(level: str, payload: dict[str, Any]) -> None:
    expected = level == "xhigh"
    kwargs = payload.get("chat_template_kwargs")
    if kwargs != {"enable_thinking": expected, "preserve_thinking": True}:
        raise AssertionError(f"{level}: unexpected chat_template_kwargs: {kwargs}")
    if "reasoning_effort" in payload:
        raise AssertionError(f"{level}: reasoning_effort must be absent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi", required=True, type=Path)
    parser.add_argument("--expected-version", default="0.84.3")
    args = parser.parse_args()
    if not args.pi.is_file():
        raise SystemExit(f"Pi binary not found: {args.pi}")
    version = subprocess.run(
        [str(args.pi), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=False,
    )
    if version.returncode != 0 or version.stdout.strip() != args.expected_version:
        raise SystemExit(
            f"Expected Pi {args.expected_version}, got {version.stdout.strip()!r}"
        )

    CaptureHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        results = {level: run_pi(args.pi, level, base_url) for level in ("off", "xhigh")}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    for level, payload in results.items():
        assert_payload(level, payload)
    compact = {
        level: {
            "chat_template_kwargs": payload["chat_template_kwargs"],
            "reasoning_effort_present": "reasoning_effort" in payload,
        }
        for level, payload in results.items()
    }
    print(
        json.dumps(
            {"pi_version": args.expected_version, "payloads": compact},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
