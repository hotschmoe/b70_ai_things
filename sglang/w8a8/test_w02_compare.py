#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from sglang.w8a8.w02_compare import ARMS, summarize


def result(rate: float, token_hash: str = "tokens") -> dict:
    return {
        "model": "qwen3.8-w8a8-w02",
        "prompt_sha256": "prompt",
        "prompt_tokens": 7,
        "completion_tokens": 6,
        "finish_reason": {"type": "length"},
        "sampling_contract": "native greedy temperature=0; seed unsupported",
        "stream_interval": 2,
        "passed": True,
        "output_ids": [1, 2, 3, 4, 5, 6],
        "output_ids_sha256": token_hash,
        "text_sha256": "text",
        "post_first_tok_s": rate,
        "ttft_ms": 100.0,
        "stability": {"final_over_first": 0.99},
    }


class CompareTest(unittest.TestCase):
    def write_fixture(self, root: Path, mismatch: bool = False) -> None:
        rates = {"eager": 10.0, "breakable": 12.0, "reclaim500": 11.8}
        for arm in ARMS:
            directory = root / arm
            directory.mkdir()
            for repeat in (1, 2, 3):
                token_hash = "changed" if mismatch and arm == "reclaim500" else "tokens"
                candidate = result(rates[arm], token_hash)
                if mismatch and arm == "reclaim500":
                    candidate["output_ids"][-1] = 7
                (directory / f"measured_{repeat}.json").write_text(
                    json.dumps(candidate), encoding="ascii"
                )

    def test_summary_attributes_graph_and_reclaim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            summary = summarize(root, 3)
            self.assertTrue(summary["cross_arm_exact"])
            self.assertAlmostEqual(
                summary["comparisons"]["breakable_over_eager"], 1.2
            )
            self.assertTrue(
                summary["comparisons"]["graph_gain_at_least_3_percent"]
            )
            self.assertTrue(summary["comparisons"]["reclaim_within_3_percent"])

    def test_output_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root, mismatch=True)
            summary = summarize(root, 3)
            self.assertFalse(summary["passed"])
            self.assertFalse(summary["cross_arm_exact"])
            self.assertEqual(summary["arms"]["reclaim500"]["first_mismatch_index"], 5)
            self.assertEqual(summary["arms"]["reclaim500"]["mismatch_count"], 1)
            self.assertFalse(
                summary["comparisons"]["performance_attribution_qualified"]
            )


if __name__ == "__main__":
    unittest.main()
