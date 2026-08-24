#!/usr/bin/env python3
"""CPU-only fixture tests for parse_quant_timing.py."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "parse_quant_timing.py"
SPEC = importlib.util.spec_from_file_location("parse_quant_timing", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParseQuantTimingTest(unittest.TestCase):
    def test_aggregates_and_projects_rows(self) -> None:
        lines = [
            "[QUANT-TIMING] version=1 scope=standard_mul_mat sample_period=128 skip=4 max_samples=65536 reserved=4 dropped=0",
            "[QUANT-TIMING] kind=device algo=MMVQ device=0 type=Q4_K reordered=1 split=1 width=1 K=5120 rows=2048 calls_seen=256 samples=2 device_ns=2000 mean_ns=1000 min_ns=900 max_ns=1100 barrier_ns=100 incomplete=0 invalid=0",
            "[QUANT-TIMING] kind=device algo=MMVQ device=1 type=Q4_K reordered=1 split=1 width=1 K=5120 rows=2048 calls_seen=256 samples=2 device_ns=2400 mean_ns=1200 min_ns=1000 max_ns=1400 barrier_ns=120 incomplete=0 invalid=0",
            "[QUANT-TIMING] end samples=4 device_ns=4400",
        ]
        parsed = MODULE.parse_lines(lines)
        self.assertEqual(parsed["summary"]["samples"], 4)
        self.assertEqual(parsed["summary"]["sampled_device_ns"], 4400)
        self.assertEqual(parsed["summary"]["projected_device_ns"], 563200)
        self.assertAlmostEqual(sum(row["projected_share"] for row in parsed["records"]), 1.0)
        self.assertEqual(parsed["declared_ends"], [{"samples": 4, "device_ns": 4400}])

    def test_rejects_incomplete_data_row(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.parse_lines(["[QUANT-TIMING] kind=device algo=MMVQ samples=1"])


if __name__ == "__main__":
    unittest.main()
