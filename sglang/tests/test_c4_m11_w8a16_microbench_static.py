#!/usr/bin/env python3
"""CPU-only contracts for experiment 05 exact-M=11 microbenchmark."""

import importlib.util
import unittest
from pathlib import Path


SGLANG_DIR = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bench = load_module(
    "bench_c4_m11_w8a16", SGLANG_DIR / "bench_c4_m11_w8a16.py"
)


class AnalysisContractTest(unittest.TestCase):
    def make_shape(self, name, current, candidate, calls=48, cv=1.0):
        timing = lambda value: {
            "device_us_mean": value,
            "device_cv_pct": cv,
            "wall_us_mean": value + 5.0,
            "wall_cv_pct": cv,
            "repeat_blocks": [],
        }
        return {
            "name": name,
            "calls_per_step": calls,
            "timings": {
                "bf16": timing(current + 20.0),
                "current_w8a8": timing(current),
                "w8a16": timing(candidate),
            },
            "numerical": {
                "current_w8a8": {
                    "relative_l2": 0.02,
                    "max_abs": 0.1,
                    "finite": True,
                },
                "w8a16": {
                    "relative_l2": 0.01,
                    "max_abs": 0.05,
                    "finite": True,
                },
            },
            "w8a16_gain_pct": 100.0 * (current - candidate) / current,
        }

    def valid_results(self):
        return [
            self.make_shape("gdn_qkvz", 120.0, 100.0, 48),
            self.make_shape("gdn_and_attn_out", 80.0, 70.0, 64),
            self.make_shape("mlp_gate_up", 200.0, 180.0, 64),
            self.make_shape("mlp_down", 110.0, 100.0, 64),
            self.make_shape("attn_qkv", 90.0, 80.0, 16),
        ]

    def test_exact_real_shape_contract(self):
        self.assertEqual(bench.M, 11)
        self.assertEqual(
            {(name, k, n, calls) for name, k, n, calls in bench.SHAPES},
            {
                ("gdn_qkvz", 5120, 8192, 48),
                ("gdn_and_attn_out", 3072, 5120, 64),
                ("mlp_gate_up", 5120, 17408, 64),
                ("mlp_down", 8704, 5120, 64),
                ("attn_qkv", 5120, 7168, 16),
            },
        )

    def test_conservative_gate_passes(self):
        analysis = bench.analyze(self.valid_results(), 5.0, 5.0, 0.10)
        self.assertTrue(analysis["pass"], analysis["checks"])
        self.assertGreater(analysis["qkvz_out_weighted"]["gain_pct"], 5.0)

    def test_qkvz_below_five_percent_fails(self):
        results = self.valid_results()
        results[0] = self.make_shape("gdn_qkvz", 100.0, 96.0, 48)
        analysis = bench.analyze(results, 5.0, 5.0, 0.10)
        self.assertFalse(analysis["checks"]["qkvz_gain_ge_threshold"])
        self.assertFalse(analysis["pass"])

    def test_unstable_repeat_fails(self):
        results = self.valid_results()
        results[3] = self.make_shape("mlp_down", 110.0, 100.0, 64, cv=5.1)
        analysis = bench.analyze(results, 5.0, 5.0, 0.10)
        self.assertFalse(analysis["checks"]["all_current_and_candidate_cv_bounded"])

    def test_accuracy_regression_fails(self):
        results = self.valid_results()
        results[1]["numerical"]["w8a16"]["relative_l2"] = 0.03
        analysis = bench.analyze(results, 5.0, 5.0, 0.10)
        self.assertFalse(analysis["checks"]["w8a16_not_less_accurate_than_w8a8"])


if __name__ == "__main__":
    unittest.main()
