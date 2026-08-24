#!/usr/bin/env python3
"""CPU-only fixtures for the queue-profiling isolation analyzer."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_qwen38_queue_profile_isolation.py"
SPEC = importlib.util.spec_from_file_location("queue_profile_isolation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


class QueueProfileIsolationAnalyzerTest(unittest.TestCase):
    def fixture(self, root: Path, on_outcome: str = "device_lost") -> None:
        image = "sha256:test"
        write_json(root / "manifest.json", {"image_id": image, "gpu_count": 2})
        write_json(root / "endpoint_down.json", {"passed": True})
        for label in ("pre", "after_off", "final"):
            (root / f"health_{label}.rc").write_text("0\n", encoding="ascii")
            (root / f"health_{label}.log").write_text(
                "xpu-health: HEALTHY (cards 0 1)\n", encoding="ascii"
            )
        (root / "code_sha256_check.rc").write_text("0\n", encoding="ascii")
        for arm, profile in (("queue_profile_off", "0"), ("queue_profile_on", "1")):
            directory = root / arm
            directory.mkdir()
            env = [
                "GPU_COUNT=2",
                "GGML_SYCL_QUANT_TIMING_SAMPLE=64",
                "GGML_SYCL_QUANT_TIMING_SKIP=18446744073709551615",
                f"GGML_SYCL_QUANT_TIMING_QUEUE_PROFILE={profile}",
            ]
            write_json(directory / "container_inspect.json", [{
                "Image": image,
                "Config": {"Env": env},
                "HostConfig": {"RestartPolicy": {"Name": "no"}},
                "RestartCount": 0,
            }])
            if arm == "queue_profile_off" or on_outcome == "clean":
                (directory / "start.rc").write_text("0\n", encoding="ascii")
                (directory / "server.log").write_text("server healthy\n", encoding="ascii")
            elif on_outcome == "device_lost":
                (directory / "start.rc").write_text("1\n", encoding="ascii")
                (directory / "server.log").write_text(
                    "UR_RESULT_ERROR_DEVICE_LOST\nError OP MUL_MAT\n", encoding="ascii"
                )
            else:
                (directory / "start.rc").write_text("1\n", encoding="ascii")
                (directory / "server.log").write_text("timed out\n", encoding="ascii")

    def test_device_lost_pair_is_decisive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            result = MODULE.analyze(root)
            self.assertTrue(result["passed"])
            self.assertEqual(result["classification"], "queue_property_root_cause")

    def test_clean_pair_is_decisive_refutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, "clean")
            result = MODULE.analyze(root)
            self.assertTrue(result["passed"])
            self.assertEqual(result["classification"], "queue_property_not_root_cause")

    def test_ambiguous_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, "ambiguous")
            result = MODULE.analyze(root)
            self.assertFalse(result["passed"])
            self.assertEqual(result["classification"], "ambiguous")

    def test_restart_or_timing_event_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            path = root / "queue_profile_on" / "container_inspect.json"
            payload = json.loads(path.read_text(encoding="ascii"))
            payload[0]["RestartCount"] = 1
            write_json(path, payload)
            with (root / "queue_profile_on" / "server.log").open("a", encoding="ascii") as handle:
                handle.write("[QUANT-TIMING] unexpected\n")
            result = MODULE.analyze(root)
            self.assertFalse(result["passed"])
            self.assertFalse(result["hard_gates"]["identity_and_no_barriers"])

    def test_final_health_or_code_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / "health_final.rc").write_text("1\n", encoding="ascii")
            (root / "code_sha256_check.rc").write_text("1\n", encoding="ascii")
            result = MODULE.analyze(root)
            self.assertFalse(result["passed"])
            self.assertFalse(result["hard_gates"]["gpu_health_pre_and_post"])
            self.assertFalse(result["hard_gates"]["code_hashes_stable"])


if __name__ == "__main__":
    unittest.main()
