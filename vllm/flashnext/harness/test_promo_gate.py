#!/usr/bin/env python3
"""CPU-only HTTP failure injection for promotion and timing evidence."""

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import promo_gate

spec = importlib.util.spec_from_file_location(
    "lmx_vllm", Path(__file__).resolve().parents[1] / "localmaxxing/lmx_vllm.py")
lmx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lmx)


class GateHTTPTests(unittest.TestCase):
    def exercise(self, fault=None):
        class Handler(BaseHTTPRequestHandler):
            counts = {}
            identities = 0

            def log_message(self, *args):
                pass

            def reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                type(self).identities += 1
                first = "wrong-model" if fault == "identity" else "hotschmoe-dev"
                root = "changed" if fault == "identity_root" and self.identities > 1 else "fixed"
                self.reply({"data": [{"id": first, "created": self.identities, "root": root},
                                     {"id": "qwen3.6-35b-a3b-ud-iq3-xxs-mtp0"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    self.reply({"tokens": list(body["prompt"].encode())})
                    return
                if self.path == "/v1/chat/completions":
                    text = "667" if "23 * 29" in body["messages"][0]["content"] else "323"
                    self.reply({"choices": [{"message": {"content": text}}]})
                    return
                prompt = body["prompt"]
                if isinstance(prompt, list):
                    self.reply({"choices": [{"text": "violet cedar copper", "token_ids": [1, 2, 3]}],
                                "usage": {"prompt_tokens": len(prompt) + (1 if fault == "long_count" else 0)}})
                    return
                n = body["max_tokens"]
                count = self.counts.get(prompt, 0) + 1
                self.counts[prompt] = count
                ids = [100 + i for i in range(n)]
                if fault == "c2" and n == 96 and count == (4 if prompt == promo_gate.COHERENCE_PROMPT else 3):
                    ids[17] += 1
                if fault == "short":
                    ids.pop()
                choice = {"text": promo_gate.C4_PARIS if n == 16 else "deterministic output", "token_ids": ids}
                if fault == "missing_ids":
                    choice.pop("token_ids")
                self.reply({"choices": [choice]})

        with tempfile.TemporaryDirectory() as directory:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            args = argparse.Namespace(base_url=f"http://127.0.0.1:{server.server_port}",
                                      timeout=5, out=directory + "/result.json",
                                      profile="proxy35b", expect_alias=None)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = promo_gate.Gate(args).run()
                result = json.loads(Path(args.out).read_text())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
        return code, result

    def test_all_checks_over_http(self):
        code, result = self.exercise()
        self.assertEqual(code, 0)
        self.assertTrue(result["pass"])
        self.assertEqual(len(result["repeat_rounds"]), 4)
        self.assertEqual(len(result["checks"]["exact_repeat"]["comparisons"]), 16)

    def test_concurrent_only_token_change_fails(self):
        code, result = self.exercise("c2")
        self.assertEqual(code, 3)
        check = result["checks"]["exact_repeat"]
        self.assertFalse(check["pass"])
        self.assertTrue(any(row["first_mismatch"] == 17 for row in check["comparisons"]))

    def test_missing_or_short_forced_ids_fail(self):
        for fault in ("missing_ids", "short"):
            with self.subTest(fault=fault):
                code, result = self.exercise(fault)
                self.assertEqual(code, 3)
                self.assertFalse(result["checks"]["exact_repeat"]["pass"])

    def test_long_usage_count_must_match(self):
        code, result = self.exercise("long_count")
        self.assertEqual(code, 3)
        self.assertFalse(result["checks"]["long_8192"]["pass"])
        self.assertFalse(result["checks"]["long_32768"]["pass"])

    def test_wrong_identity_fails_before_inference(self):
        code, result = self.exercise("identity")
        self.assertEqual(code, 3)
        self.assertNotIn("repeat_rounds", result)

    def test_changed_model_root_fails(self):
        code, result = self.exercise("identity_root")
        self.assertEqual(code, 3)
        self.assertFalse(result["checks"]["identity_unchanged"]["pass"])


class TimingTests(unittest.TestCase):
    def stream(self, done=True):
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "three tokens in one chunk"}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
        ]
        lines = [b"data: " + json.dumps(e).encode() + b"\n" for e in events]
        if done:
            lines.append(b"data: [DONE]\n")
        return lines

    def test_usage_count_reasoning_ttft_and_multi_token_chunk(self):
        with patch.object(lmx.time, "perf_counter_ns", side_effect=[100, 200, 1000000200, 1100000200, 1200000200]):
            row = lmx.parse_stream(self.stream(), 0)
        self.assertEqual(row["ttft_ms"], 0.0002)
        self.assertEqual(row["content_chunks"], 2)
        self.assertEqual(row["tok_s_out"], 3)
        self.assertAlmostEqual(row["effective_itl_ms"], 1000 / 3)
        self.assertEqual(row["output"], "thinkthree tokens in one chunk")

    def test_truncated_stream_fails(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            lmx.parse_stream(self.stream(done=False), 0)

    def test_metrics_sum_dp_labels_and_detect_counter_reset(self):
        values = lmx.metric_values('vllm:generation_tokens_total{engine="0"} 96\n'
                                  'vllm:generation_tokens_total{engine="1"} 97\n'
                                  '# comment\nvllm:num_requests_running 0\n')
        self.assertEqual(values["vllm:generation_tokens_total"], 193)
        diff = lmx.deltas(values, {"vllm:generation_tokens_total": 1})
        self.assertEqual(diff["vllm:generation_tokens_total"], -192)


if __name__ == "__main__":
    unittest.main()
