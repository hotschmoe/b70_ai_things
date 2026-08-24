#!/usr/bin/env python3
"""CPU-only contracts for the C4 target-GDN qkvz-only candidate."""

import importlib.util
import json
import sys
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


prepare = load_module(
    "prepare_c4_gdn_qkvz_candidate",
    SGLANG_DIR / "prepare_c4_gdn_qkvz_candidate.py",
)


class CandidateContractTest(unittest.TestCase):
    def test_exact_config_scope(self):
        path = (
            SGLANG_DIR
            / "configs/qwen36_w8a8_sqgptq_gdn_qkvz_rtn_quantization.json"
        )
        quant = json.loads(path.read_text(encoding="ascii"))
        ignores = set(quant["ignore"])
        self.assertIn(r"re:.*linear_attn\.out_proj$", ignores)
        self.assertIn(r"re:.*linear_attn\.in_proj_b$", ignores)
        self.assertIn(r"re:.*linear_attn\.in_proj_a$", ignores)
        self.assertNotIn(r"re:.*linear_attn\.in_proj_qkv$", ignores)
        self.assertNotIn(r"re:.*linear_attn\.in_proj_z$", ignores)
        self.assertEqual(quant["config_groups"]["group_0"]["targets"], ["Linear"])

    def test_exact_size_contracts(self):
        self.assertEqual(prepare.CHECKPOINT_BYTES_SAVED, 4_024_958_976)
        self.assertEqual(prepare.TP2_BYTES_SAVED_PER_RANK, 2_012_479_488)
        self.assertEqual(prepare.CANDIDATE_TOTAL_SIZE, 31_903_750_624)
        self.assertEqual(
            35_928_709_600 - prepare.CHECKPOINT_BYTES_SAVED,
            prepare.CANDIDATE_TOTAL_SIZE,
        )

    def test_checkpoint_leaf_selection(self):
        layers = [0, 2]
        root = "model.language_model.layers.2.linear_attn"
        self.assertEqual(
            prepare.target_projection(f"{root}.in_proj_qkv.weight", layers),
            "in_proj_qkv",
        )
        self.assertEqual(
            prepare.target_projection(f"{root}.in_proj_z.weight", layers),
            "in_proj_z",
        )
        self.assertIsNone(
            prepare.target_projection(f"{root}.out_proj.weight", layers)
        )
        self.assertIsNone(
            prepare.target_projection(f"{root}.in_proj_b.weight", layers)
        )

    def test_host_candidate_audit_if_materialized(self):
        repo = SGLANG_DIR.parent
        base = repo / "models/files/qwen3.6-27b/w8a8-sqgptq"
        combined = repo / "models/files/qwen3.6-27b/w8a8-sqgptq-gdnint8"
        candidate = repo / "models/files/qwen3.6-27b/w8a8-sqgptq-gdn-qkvz-int8"
        quant = (
            SGLANG_DIR
            / "configs/qwen36_w8a8_sqgptq_gdn_qkvz_rtn_quantization.json"
        )
        if not candidate.is_dir():
            self.skipTest("qkvz-only candidate is not materialized")
        _overlay, report = prepare.audit(base, combined, candidate, quant)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["target_gdn_int8_weights"], 96)
        self.assertEqual(
            report["tp2_runtime_bytes_saved_per_rank"], 2_012_479_488
        )


if __name__ == "__main__":
    unittest.main()
