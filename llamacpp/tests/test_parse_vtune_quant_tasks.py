#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llamacpp.parse_vtune_quant_tasks import classify_family, parse_report


class ClassifyFamilyTests(unittest.TestCase):
    def test_explicit_weight_wins_over_activation(self) -> None:
        task = "dequantize_mul_mat_vec_q5_K_q8_1_sycl"
        self.assertEqual(classify_family(task), "q5_K")

    def test_numeric_template_enum(self) -> None:
        task = "submit_mmvq_reorder<reorder_vec_dot_q_sycl<(ggml_type)12> >"
        self.assertEqual(classify_family(task), "q4_K")

    def test_mangled_numeric_template_enum(self) -> None:
        task = "HostKernelIZZL19submit_mmvq_reorderIL9ggml_type23EEv"
        self.assertEqual(classify_family(task), "iq4_xs")

    def test_non_quant_kernel(self) -> None:
        self.assertIsNone(classify_family("soft_max_f32_sycl"))


class ParseReportTests(unittest.TestCase):
    def write_report(self, text: str) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="ascii")
        with temporary:
            temporary.write(text)
        self.addCleanup(Path(temporary.name).unlink)
        return Path(temporary.name)

    def test_two_adapter_report(self) -> None:
        path = self.write_report(
            "vtune preamble\n"
            "GPU Adapter,Computing Task,Total Time\n"
            'Arc 0,"dequantize_mul_mat_vec_q5_K_q8_1_sycl",12ms\n'
            'Arc 1,"HostKernelIZZL19submit_mmvq_reorderIL9ggml_type12EEv",0.010\n'
            'Arc 0,"soft_max_f32_sycl",3us\n'
        )
        result = parse_report(path, {"q5_K", "q4_K"})
        self.assertTrue(result["passed"])
        self.assertEqual(result["adapters"], ["Arc 0", "Arc 1"])
        self.assertAlmostEqual(result["total_task_time_s"], 0.022003)
        self.assertEqual(len(result["unknown_tasks"]), 1)

    def test_missing_required_family_fails(self) -> None:
        path = self.write_report(
            "GPU Adapter,Computing Task,Total Time\n"
            "Arc 0,dequantize_row_q5_K_sycl,0.1\n"
            "Arc 1,dequantize_row_q5_K_sycl,0.1\n"
        )
        result = parse_report(path, {"q5_K", "iq4_xs"})
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_required_families"], ["iq4_xs"])

    def test_missing_adapter_column_fails(self) -> None:
        path = self.write_report(
            "Computing Task,Total Time\n"
            "dequantize_row_q5_K_sycl,0.1\n"
        )
        result = parse_report(path, {"q5_K"})
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["adapter_column_present"])


if __name__ == "__main__":
    unittest.main()
