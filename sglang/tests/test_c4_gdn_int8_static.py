#!/usr/bin/env python3
"""CPU-only contract tests for the C4 GDN INT8 mechanism gate."""

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SGLANG_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SGLANG_DIR))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = load_module(
    "analyze_c4_gdn_int8_mechanism",
    SGLANG_DIR / "analyze_c4_gdn_int8_mechanism.py",
)
prepare = load_module(
    "prepare_c4_gdn_int8_candidate",
    SGLANG_DIR / "prepare_c4_gdn_int8_candidate.py",
)


def add_cpu_events(events, name, dims, count):
    for index in range(count):
        events.append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": name,
                "pid": 1,
                "tid": 1,
                "ts": index,
                "dur": 1,
                "args": {
                    "External id": len(events) + 1,
                    "Input Dims": dims,
                },
            }
        )


class RouteContractTest(unittest.TestCase):
    def make_trace(self, path, qkvz_calls=240):
        events = []
        add_cpu_events(
            events,
            "_xpu_C::int8_gemm_w8a8",
            [[11, 5120], [11, 1], [], [5120, 8192], [8192], [], [], []],
            qkvz_calls,
        )
        add_cpu_events(
            events,
            "_xpu_C::int8_gemm_w8a8",
            [[11, 3072], [11, 1], [], [3072, 5120], [5120], [], [], []],
            320,
        )
        add_cpu_events(events, "aten::mm", [[11, 3072], [3072, 5120]], 5)
        add_cpu_events(events, "aten::mm", [[11, 5120], [5120, 48]], 240)
        add_cpu_events(events, "aten::mm", [[11, 5120], [5120, 7168]], 5)
        add_cpu_events(
            events,
            "_xpu_C::dynamic_per_token_int8_quant",
            [[11, 5120], [], []],
            640,
        )
        add_cpu_events(
            events,
            "_xpu_C::dynamic_per_token_int8_quant",
            [[11, 3072], [], []],
            320,
        )
        with gzip.open(path, "wt", encoding="ascii") as handle:
            json.dump({"traceEvents": events}, handle)

    def test_exact_route_contract_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0-DECODE.trace.json.gz"
            self.make_trace(path)
            summary, checks = analyzer.route_summary(path)
            self.assertEqual(summary["w8a8_m11_qkvz"], 240)
            self.assertTrue(all(checks.values()), checks)

    def test_one_missing_gdn_call_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0-DECODE.trace.json.gz"
            self.make_trace(path, qkvz_calls=239)
            _summary, checks = analyzer.route_summary(path)
            self.assertFalse(checks["w8a8_m11_qkvz_48_per_step"])


class CheckpointContractTest(unittest.TestCase):
    def test_packed_ba_ignore_uses_both_checkpoint_leaves(self):
        quant_path = (
            SGLANG_DIR
            / "configs/qwen36_w8a8_sqgptq_gdnrtn_quantization.json"
        )
        quant = json.loads(quant_path.read_text(encoding="ascii"))
        ignores = set(quant["ignore"])
        self.assertIn(r"re:.*linear_attn\.in_proj_b$", ignores)
        self.assertIn(r"re:.*linear_attn\.in_proj_a$", ignores)
        self.assertNotIn(r"re:.*linear_attn\.in_proj_ba$", ignores)

    def test_host_candidate_audit(self):
        repo = SGLANG_DIR.parent
        base = repo / "models/files/qwen3.6-27b/w8a8-sqgptq"
        candidate = repo / "models/files/qwen3.6-27b/w8a8-sqgptq-gdnint8"
        quant = (
            SGLANG_DIR
            / "configs/qwen36_w8a8_sqgptq_gdnrtn_quantization.json"
        )
        if not base.is_dir() or not candidate.is_dir():
            self.skipTest("local model artifacts are not provisioned")
        _overlay, report = prepare.audit(base, candidate, quant)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["target_gdn_int8_weights"], 144)
        self.assertEqual(report["tp2_runtime_bytes_saved_per_rank"], 2766962688)


if __name__ == "__main__":
    unittest.main()
