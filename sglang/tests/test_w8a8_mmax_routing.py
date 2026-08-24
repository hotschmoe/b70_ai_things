#!/usr/bin/env python3
"""CPU-only contracts for the default-off W8A16 M threshold."""

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path


SHIM_PATH = Path(__file__).resolve().parents[1] / "patches" / "w8a8_shim.py"


def load_shim():
    spec = importlib.util.spec_from_file_location("w8a8_shim_mmax_test", SHIM_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


shim = load_shim()


class W8A16MMaxTest(unittest.TestCase):
    def setUp(self):
        shim._ROUTE_COUNTS.update(w8a16=0, w8a8=0)
        shim._ROUTE_LOGGED.clear()

    def test_unset_preserves_m1_only_default(self):
        value, configured = shim._w8a16_m_max({})
        self.assertEqual(value, 1)
        self.assertFalse(configured)
        self.assertTrue(shim._use_w8a16(1, value))
        self.assertFalse(shim._use_w8a16(2, value))

    def test_exact_m11_threshold(self):
        value, configured = shim._w8a16_m_max({"B70_W8A16_M_MAX": "11"})
        self.assertEqual(value, 11)
        self.assertTrue(configured)
        self.assertFalse(shim._use_w8a16(0, value))
        self.assertTrue(all(shim._use_w8a16(m, value) for m in range(1, 12)))
        self.assertFalse(shim._use_w8a16(12, value))
        self.assertFalse(shim._use_w8a16(2048, value))

    def test_invalid_values_fail_closed(self):
        invalid = (
            "",
            "0",
            "12",
            "-1",
            "+11",
            " 11",
            "11 ",
            "1.0",
            "abc",
            "\u0661\u0661",
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                shim._w8a16_m_max({"B70_W8A16_M_MAX": raw})

    def test_route_log_proves_m11_and_above_threshold(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            shim._record_w8a8_route("w8a16", 11, 5120, 8192, 11)
            shim._record_w8a8_route("w8a16", 11, 5120, 8192, 11)
            shim._record_w8a8_route("w8a8", 12, 5120, 8192, 11)
            shim._record_w8a8_route("w8a8", 2048, 5120, 8192, 11)
        text = output.getvalue()
        self.assertEqual(text.count("route=w8a16"), 1)
        self.assertEqual(text.count("route=w8a8"), 1)
        self.assertIn("route=w8a16 M=11 K=5120 N=8192 m_max=11 relation=at_max", text)
        self.assertIn("route=w8a8 M=12 K=5120 N=8192 m_max=11 relation=above_max", text)
        self.assertEqual(shim._ROUTE_COUNTS, {"w8a16": 2, "w8a8": 2})

    def test_shim_uses_one_shared_weight_view_for_both_routes(self):
        source = SHIM_PATH.read_text(encoding="ascii")
        self.assertIn("if _use_w8a16(M, w8a16_m_max):", source)
        self.assertIn('route_debug = os.environ.get("B70_W8A16_ROUTE_DEBUG") == "1"', source)
        self.assertIn(
            "int8_gemm_w8a16(xf, layer.B_nt, layer.wscale_n, b)", source
        )
        self.assertIn(
            "xq, xs.contiguous(), None, layer.B_nt, layer.wscale_n", source
        )
        self.assertEqual(source.count("layer.B_nt = weight_NK.t()"), 1)


if __name__ == "__main__":
    unittest.main()
