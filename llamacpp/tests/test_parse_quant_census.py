#!/usr/bin/env python3
"""CPU-only fixture tests for parse_quant_census.py."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "parse_quant_census.py"
SPEC = importlib.util.spec_from_file_location("parse_quant_census", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParseQuantCensusTest(unittest.TestCase):
    def test_aggregates_logical_and_per_device_actual_rows(self) -> None:
        lines = [
            "noise before census",
            "[QUANT-CENSUS] version=1 scope=standard_mul_mat",
            "[QUANT-CENSUS] kind=logical algo=MMVQ device=-1 type=Q4_K reordered=1 split=1 width=1 K=5120 N=4096 rows=-1 calls=2",
            "[QUANT-CENSUS] kind=actual algo=MMVQ device=0 type=Q4_K reordered=1 split=1 width=1 K=5120 N=-1 rows=2048 calls=2",
            "[QUANT-CENSUS] kind=actual algo=MMVQ device=1 type=Q4_K reordered=1 split=1 width=1 K=5120 N=-1 rows=2048 calls=2",
            "[QUANT-CENSUS] end logical_total=2 actual_total=4",
        ]
        parsed = MODULE.parse_lines(lines)
        self.assertEqual(parsed["versions"], ["1"])
        self.assertEqual(parsed["computed_totals"], {"logical_total": 2, "actual_total": 4})
        self.assertEqual(len(parsed["records"]), 3)
        actual_devices = {
            record["device"] for record in parsed["records"] if record["kind"] == "actual"
        }
        self.assertEqual(actual_devices, {0, 1})

    def test_rejects_incomplete_data_row(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.parse_lines(["[QUANT-CENSUS] kind=logical algo=MMVQ calls=1"])


if __name__ == "__main__":
    unittest.main()
