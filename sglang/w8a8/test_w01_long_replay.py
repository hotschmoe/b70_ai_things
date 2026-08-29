#!/usr/bin/env python3

import io
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sglang.w8a8.capture_greedy_corpus import (
    PROMPTS as CORPUS_PROMPTS,
    validate_reference_contract,
)
from sglang.w8a8 import w01_long_replay
from sglang.w8a8.w01_long_replay import window_summary


class FakeResponse:
    def __init__(self, body=b"", lines=()):
        self.body = io.BytesIO(body)
        self.lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *args):
        return self.body.read(*args)

    def __iter__(self):
        return iter(self.lines)


class WindowSummaryTest(unittest.TestCase):
    def test_reference_contract_rejects_wrong_model(self):
        reference = {
            "model": "wrong",
            "max_tokens": 96,
            "repeat_exact": True,
            "samples": [
                {
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                }
                for prompt in CORPUS_PROMPTS
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
            validate_reference_contract(reference, "expected", 96)

    def test_flat_windows_pass(self):
        milestones = [
            {"completion_tokens": 1, "elapsed_s": 0.0},
            {"completion_tokens": 5001, "elapsed_s": 400.0},
            {"completion_tokens": 10001, "elapsed_s": 805.0},
            {"completion_tokens": 15000, "elapsed_s": 1210.0},
        ]
        result = window_summary(milestones, 0.80)
        self.assertTrue(result["passed"])
        self.assertGreater(result["final_over_first"], 0.98)

    def test_degraded_final_window_fails(self):
        milestones = [
            {"completion_tokens": 1, "elapsed_s": 0.0},
            {"completion_tokens": 5001, "elapsed_s": 400.0},
            {"completion_tokens": 10001, "elapsed_s": 800.0},
            {"completion_tokens": 15000, "elapsed_s": 1500.0},
        ]
        result = window_summary(milestones, 0.80)
        self.assertFalse(result["passed"])
        self.assertLess(result["final_over_first"], 0.80)

    def test_non_monotonic_milestones_fail(self):
        milestones = [
            {"completion_tokens": 1, "elapsed_s": 0.0},
            {"completion_tokens": 5001, "elapsed_s": 400.0},
            {"completion_tokens": 5001, "elapsed_s": 500.0},
        ]
        with self.assertRaisesRegex(RuntimeError, "strictly increasing"):
            window_summary(milestones, 0.80)

    def test_main_parses_exact_native_stream(self):
        model = "w01-test-model"
        models = json.dumps({"data": [{"id": model}]}).encode("ascii")
        events = []
        for count in (1, 11, 21, 30):
            event = {
                "text": "x" * count,
                "output_ids": list(range(count)),
                "meta_info": {
                    "completion_tokens": count,
                    "prompt_tokens": 7,
                    "finish_reason": {"type": "length"} if count == 30 else None,
                },
            }
            events.append(("data: " + json.dumps(event) + "\n").encode("ascii"))
        events.append(b"data: [DONE]\n")
        responses = [FakeResponse(models), FakeResponse(lines=events)]

        def fake_urlopen(_request, timeout=None):
            self.assertIsNotNone(timeout)
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = [
                "w01_long_replay.py",
                "--base",
                "http://test",
                "--model",
                model,
                "--json-out",
                str(output),
                "--output-tokens",
                "30",
                "--window-tokens",
                "10",
                "--stream-interval",
                "2",
            ]
            clock = iter((0.0, 0.1, 1.1, 2.1, 3.0, 3.1))
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    w01_long_replay.urllib.request,
                    "urlopen",
                    side_effect=fake_urlopen,
                ),
                mock.patch.object(
                    w01_long_replay.time, "perf_counter", side_effect=clock
                ),
            ):
                w01_long_replay.main()
            result = json.loads(output.read_text(encoding="ascii"))
            self.assertTrue(result["passed"])
            self.assertEqual(result["completion_tokens"], 30)
            self.assertEqual(result["finish_reason"], {"type": "length"})
            self.assertEqual(len(result["stability"]["windows"]), 3)


if __name__ == "__main__":
    unittest.main()
