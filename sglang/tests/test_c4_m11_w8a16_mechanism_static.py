#!/usr/bin/env python3
"""CPU-only contracts for experiment 06 M<=11 W8A16 mechanism."""

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SGLANG_DIR = Path(__file__).resolve().parents[1]
REPO = SGLANG_DIR.parent
sys.path.insert(0, str(SGLANG_DIR))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = load_module(
    "analyze_c4_m11_w8a16_mechanism",
    SGLANG_DIR / "analyze_c4_m11_w8a16_mechanism.py",
)


def add(events, name, dims, count):
    for _ in range(count):
        events.append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": name,
                "dur": 1,
                "args": {"External id": len(events) + 1, "Input Dims": dims},
            }
        )


class RouteContractTest(unittest.TestCase):
    def make_trace(self, path, gate_calls=320):
        events = []
        add(events, "_xpu_C::int8_gemm_w8a16", [[11, 5120], [5120, 17408], [17408], []], gate_calls)
        add(events, "_xpu_C::int8_gemm_w8a16", [[11, 8704], [8704, 5120], [5120], []], 320)
        add(events, "_xpu_C::int8_gemm_w8a16", [[11, 5120], [5120, 7168], [7168], []], 80)
        add(events, "_xpu_C::int8_gemm_w8a16", [[11, 3072], [3072, 5120], [5120], []], 80)
        with gzip.open(path, "wt", encoding="ascii") as handle:
            json.dump({"traceEvents": events}, handle)

    def test_exact_route_contract_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0-DECODE.trace.json.gz"
            self.make_trace(path)
            summary, checks = analyzer.route_summary(path)
            self.assertEqual(summary["w8a16_gate_up"], 320)
            self.assertTrue(all(checks.values()), checks)

    def test_one_missing_call_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0-DECODE.trace.json.gz"
            self.make_trace(path, gate_calls=319)
            _summary, checks = analyzer.route_summary(path)
            self.assertFalse(checks["w8a16_gate_up_exact"])
            self.assertFalse(checks["w8a16_total_160_per_step"])


class StaticHarnessTest(unittest.TestCase):
    def test_runner_is_external_lease_and_endpoint_down(self):
        source = (SGLANG_DIR / "06_c4_m11_w8a16_mechanism.sh").read_text(
            encoding="ascii"
        )
        for required in (
            "./bin/gpu-run bash sglang/06_c4_m11_w8a16_mechanism.sh",
            "W8A16_M_MAX=11 W8A16_ROUTE_DEBUG=1",
            "profile exactly five M11 decode iterations",
            "ensure_down",
            "xpu-health",
            "artifacts_after.sha256",
            "/get_server_info",
            "deterministic_1.json",
            "gate_concurrent_coherence.py",
        ):
            self.assertIn(required, source)
        self.assertNotIn("production_restore", source)

    def test_analyzer_requires_base_capacity(self):
        source = (
            SGLANG_DIR / "analyze_c4_m11_w8a16_mechanism.py"
        ).read_text(encoding="ascii")
        self.assertIn('server_info = load_json(root / "server_info.json")', source)
        self.assertIn('"capacity_ge_base"', source)
        self.assertIn(">= 143360", source)

    def test_shelf_propagates_inert_default_threshold_and_debug(self):
        shelf = (
            REPO / "rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
        ).read_text(encoding="ascii")
        self.assertIn('W8A16_M_MAX="${W8A16_M_MAX:-1}"', shelf)
        self.assertIn('W8A16_ROUTE_DEBUG="${W8A16_ROUTE_DEBUG:-0}"', shelf)
        self.assertIn('-e "B70_W8A16_M_MAX=$W8A16_M_MAX"', shelf)
        self.assertIn('-e "B70_W8A16_ROUTE_DEBUG=$W8A16_ROUTE_DEBUG"', shelf)

    def test_qkvz_lane_moved_after_06_and_07(self):
        plan = (SGLANG_DIR / "C4_GDN_QKVZ_ONLY_PLAN.md").read_text(
            encoding="ascii"
        )
        self.assertIn("08_c4_gdn_qkvz_int8_mechanism.sh", plan)
        self.assertIn("09_c4_gdn_qkvz_int8_abba.sh", plan)


if __name__ == "__main__":
    unittest.main()
