#!/usr/bin/env python3
"""CPU-only contract tests for the C4 GDN out_proj-only candidate."""

import importlib.util
import json
import unittest
from pathlib import Path


SGLANG_DIR = Path(__file__).resolve().parents[1]
REPO = SGLANG_DIR.parent


def load_prepare():
    path = SGLANG_DIR / "prepare_c4_gdn_out_proj_candidate.py"
    spec = importlib.util.spec_from_file_location("c4_gdn_out_prepare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_prepare()


class OutProjCandidateTest(unittest.TestCase):
    def test_exact_quantization_metadata(self):
        path = (
            SGLANG_DIR
            / "configs/qwen36_w8a8_sqgptq_gdn_out_proj_rtn_quantization.json"
        )
        quant = json.loads(path.read_text(encoding="ascii"))
        self.assertEqual(
            set(quant["ignore"]),
            {
                "lm_head",
                r"re:.*linear_attn\.in_proj_qkv$",
                r"re:.*linear_attn\.in_proj_z$",
                r"re:.*linear_attn\.in_proj_b$",
                r"re:.*linear_attn\.in_proj_a$",
                r"re:.*visual.*",
                r"re:.*mtp.*",
            },
        )
        self.assertNotIn(r"re:.*linear_attn\.out_proj$", quant["ignore"])

    def test_materialized_candidate_audit(self):
        base = REPO / "models/files/qwen3.6-27b/w8a8-sqgptq"
        combined = REPO / "models/files/qwen3.6-27b/w8a8-sqgptq-gdnint8"
        candidate = (
            REPO
            / "models/files/qwen3.6-27b/w8a8-sqgptq-gdn-out-proj-int8"
        )
        quant = (
            SGLANG_DIR
            / "configs/qwen36_w8a8_sqgptq_gdn_out_proj_rtn_quantization.json"
        )
        if not candidate.is_dir():
            self.skipTest("local out_proj-only candidate is not provisioned")
        _overlay, report = prepare.audit(base, combined, candidate, quant)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["target_gdn_int8_weights"], 48)
        self.assertEqual(report["checkpoint_bytes_saved"], 1509457920)
        self.assertEqual(report["tp2_runtime_bytes_saved_per_rank"], 754483200)


if __name__ == "__main__":
    unittest.main()
