#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qwen38_fp8_f02 import analyze


SERVED = "qwen3.8-f02-test"


def write_attempt(root: Path, index: int, arrays: list[list[int]]) -> None:
    attempt = root / f"attempt-{index}"
    attempt.mkdir()
    rows = [
        {
            "prompt_id": f"prompt-{row_index}",
            "prompt_class": "test",
            "token_ids": values,
            "sha256": f"text-{index}-{row_index}",
        }
        for row_index, values in enumerate(arrays)
    ]
    performance = {
        "realistic_final_gate": {"passed": True, "cached_tokens_all_zero": True},
        "fresh_response_validity": {"performance_gate_eligible": True},
        "summary": {
            "class_balanced_tok_s_1_100_intervals_after_ttft": {"median": 10.0 + index}
        },
        "rows": rows,
    }
    (attempt / "performance.json").write_text(json.dumps(performance), encoding="ascii")
    (attempt / "canaries.json").write_text(json.dumps({"pass_all": True}), encoding="ascii")
    (attempt / "models.json").write_text(
        json.dumps({"data": [{"id": SERVED}]}), encoding="ascii"
    )


class AnalyzerTests(unittest.TestCase):
    def test_exact_pair_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = [[index, index + 1] for index in range(12)]
            write_attempt(root, 1, arrays)
            write_attempt(root, 2, arrays)
            summary = analyze(root, 2, SERVED, [])
            self.assertEqual(summary["verdict"], "passed")
            self.assertEqual(summary["exact_prompts_minimum_pair"], 12)
            self.assertTrue(summary["performance_attribution_qualified"])
            self.assertTrue(summary["inductor_combo_kernels"])
            self.assertTrue(summary["inductor_benchmark_combo_kernel"])
            self.assertTrue(summary["inductor_max_autotune"])
            self.assertTrue(summary["inductor_coordinate_descent_tuning"])

    def test_combo_kernel_metadata_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = [[index, index + 1] for index in range(12)]
            write_attempt(root, 1, arrays)
            write_attempt(root, 2, arrays)
            summary = analyze(
                root,
                2,
                SERVED,
                [],
                inductor_combo_kernels=False,
                inductor_benchmark_combo_kernel=False,
                inductor_max_autotune=False,
                inductor_coordinate_descent_tuning=False,
            )
            self.assertFalse(summary["inductor_combo_kernels"])
            self.assertFalse(summary["inductor_benchmark_combo_kernel"])
            self.assertFalse(summary["inductor_max_autotune"])
            self.assertFalse(summary["inductor_coordinate_descent_tuning"])

    def test_mismatch_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = [[index, index + 1] for index in range(12)]
            right = [list(values) for values in left]
            right[3][1] = 999
            write_attempt(root, 1, left)
            write_attempt(root, 2, right)
            summary = analyze(root, 2, SERVED, [])
            comparison = summary["pair_comparisons"][0]
            mismatch = comparison["prompt_comparisons"][3]
            self.assertEqual(summary["verdict"], "failed_cross_server_token_exactness")
            self.assertEqual(comparison["exact_prompts"], 11)
            self.assertEqual(mismatch["first_mismatch_zero_based"], 1)
            self.assertFalse(summary["performance_attribution_qualified"])

    def test_required_reference_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = [[index, index + 1] for index in range(12)]
            write_attempt(root, 1, arrays)
            write_attempt(root, 2, arrays)
            reference = json.loads(
                (root / "attempt-1" / "performance.json").read_text(encoding="ascii")
            )
            reference["rows"][5]["token_ids"][0] = 999
            reference_path = root / "reference.json"
            reference_path.write_text(json.dumps(reference), encoding="ascii")
            summary = analyze(
                root,
                2,
                SERVED,
                [reference_path],
                require_reference_exact=True,
            )
            self.assertEqual(summary["verdict"], "failed_reference_token_exactness")
            self.assertTrue(summary["complete_token_arrays_exact"])
            self.assertFalse(summary["reference_token_arrays_exact"])
            self.assertFalse(summary["performance_attribution_qualified"])


if __name__ == "__main__":
    unittest.main()
