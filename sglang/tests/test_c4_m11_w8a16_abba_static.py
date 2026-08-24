#!/usr/bin/env python3
"""CPU-only contracts for experiment 07 M<=11 W8A16 A-B-B-A."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SGLANG_DIR = Path(__file__).resolve().parents[1]
REPO = SGLANG_DIR.parent


def load_analyzer():
    path = SGLANG_DIR / "analyze_c4_m11_w8a16_abba.py"
    spec = importlib.util.spec_from_file_location("c4_m11_w8a16_abba", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = load_analyzer()


def row(phase, soak, cv_pct=1.0):
    return {"phase_decode": phase, "soak_decode": soak, "phase_cv_pct": cv_pct}


class PerformanceGateTest(unittest.TestCase):
    def valid(self):
        rows = {
            "01_A1": row(20.0, 20.0),
            "02_B1": row(20.8, 20.5, 4.9),
            "03_B2": row(21.0, 20.4, 5.0),
            "04_A2": row(20.1, 20.0),
        }
        deltas = {
            "phase_decode_pct": 3.5,
            "soak_decode_pct": 2.1,
            "phase_ttft_ms_pct": -5.0,
            "perf_c1_ttft_ms_pct": 5.0,
            "perf_c4_ttft_ms_pct": 0.0,
            "prefill_c1_ttft_ms_pct": -4.9,
            "prefill_c4_ttft_ms_pct": 4.9,
        }
        return rows, deltas

    def test_predeclared_thresholds_pass(self):
        rows, deltas = self.valid()
        checks = analyzer.make_performance_checks(rows, deltas)
        self.assertTrue(all(checks.values()), checks)

    def test_primary_gates_fail_closed(self):
        base_rows, base_deltas = self.valid()
        cases = []
        rows = {key: dict(value) for key, value in base_rows.items()}
        rows["02_B1"]["phase_decode"] = 19.9
        cases.append((rows, dict(base_deltas), "phase_both_pairs_win"))
        cases.append((base_rows, dict(base_deltas, phase_decode_pct=2.99), "balanced_phase_gain_ge_3pct"))
        rows = {key: dict(value) for key, value in base_rows.items()}
        rows["03_B2"]["soak_decode"] = 19.9
        cases.append((rows, dict(base_deltas), "soak_both_pairs_nonregress"))
        cases.append((base_rows, dict(base_deltas, soak_decode_pct=1.99), "balanced_soak_gain_ge_2pct"))
        cases.append((base_rows, dict(base_deltas, prefill_c4_ttft_ms_pct=-5.01), "ttft_and_prefill_within_5pct"))
        rows = {key: dict(value) for key, value in base_rows.items()}
        rows["02_B1"]["phase_cv_pct"] = 5.01
        cases.append((rows, dict(base_deltas), "b_within_process_cv_le_5pct"))
        for rows, deltas, failed in cases:
            with self.subTest(failed=failed):
                self.assertFalse(analyzer.make_performance_checks(rows, deltas)[failed])


class RunnerContractTest(unittest.TestCase):
    def test_runner_is_exact_balanced_native_config_campaign(self):
        source = (SGLANG_DIR / "07_c4_m11_w8a16_abba.sh").read_text(encoding="ascii")
        for required in (
            "./bin/gpu-run bash sglang/07_c4_m11_w8a16_abba.sh",
            'MODEL_CONTAINER="/models/qwen3.6-27b/w8a8-sqgptq"',
            "run_arm 01_A1 A",
            "run_arm 02_B1 B",
            "run_arm 03_B2 B",
            "run_arm 04_A2 A",
            "W8A16_M_MAX=\"$threshold\" W8A16_ROUTE_DEBUG=0",
            "REPLICATE_MTP_EMBED=1 PUSH_AR=1 PUSH_AR_MIN_NUMEL=0",
            "DELAY_MLP_AR=0 FUSED_MLP_AR_NORM=0",
            "CTX=\"$CTX\" RADIX=0 MAXREQ=\"$MAXREQ\"",
            "SPEC_STEPS=10",
            "SPEC_DRAFT=11",
            "gate_concurrent_coherence.py",
            "soak_probe.py\" \"$PORT\" \"$served\" 6400 800",
            "endpoint_policy=down_between_arms_and_after_campaign_no_restore",
        ):
            self.assertIn(required, source)
        self.assertNotIn("PREPARE=", source)
        self.assertNotIn("candidate_config.json", source)
        self.assertNotIn("checkpoint_audit", source)

    def test_analyzer_has_all_requested_gates_and_acceptance_diagnostic(self):
        source = (SGLANG_DIR / "analyze_c4_m11_w8a16_abba.py").read_text(encoding="ascii")
        for required in (
            '"phase_both_pairs_win"',
            '"balanced_phase_gain_ge_3pct"',
            '"soak_both_pairs_nonregress"',
            '"balanced_soak_gain_ge_2pct"',
            '"ttft_and_prefill_within_5pct"',
            '"b_within_process_cv_le_5pct"',
            '"b_restart_phase_spread_le_5pct"',
            '"b_restart_soak_spread_le_5pct"',
            '"b_fixed_outputs_byte_identical"',
            '"b_deterministic_corpora_byte_identical"',
            '"all_96_mixed_coherent"',
            '"all_capacities_unchanged"',
            '"acceptance_diagnostic_only": True',
            '"accept_avg_pct"',
            '"accept_avg_abs"',
        ):
            self.assertIn(required, source)


class IdentityFixtureTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "image_id": "sha256:test",
            "model": "/models/base",
            "served_a": "base-m1",
            "served_b": "base-m11",
            "ctx": "131072",
            "maxreq": "4",
            "kdir": "/runtime/kernel",
            "pushdir": "/runtime/push",
            "push_ar_so": "/work/push_ar/libxpu_push_ar_graph.so",
            "push_ar_maxb": "536870912",
        }

    def inspect_item(self, candidate):
        served = self.manifest["served_b" if candidate else "served_a"]
        threshold = "11" if candidate else "1"
        mounts = [
            {"Destination": "/work/kernel", "Source": self.manifest["kdir"], "RW": False},
            {"Destination": "/work/push_ar", "Source": self.manifest["pushdir"], "RW": False},
        ]
        for name in ("w8a8_shim.py", "mtp_replicated_embedding.py", "push_ar_xpu.py"):
            mounts.append(
                {
                    "Destination": f"/opt/venv/lib/python3.12/site-packages/{name}",
                    "Source": str(REPO / f"sglang/patches/{name}"),
                    "RW": False,
                }
            )
        env = [
            "B70_XPU_W8A8=1",
            "B70_XPU_W8A8_FUSED=1",
            "B70_XPU_MTP=1",
            f"B70_W8A16_M_MAX={threshold}",
            "B70_W8A16_ROUTE_DEBUG=0",
            "B70_W8A8_QUANT_LMHEAD=0",
            "B70_XPU_REPLICATE_MTP_EMBED=1",
            "B70_XPU_DELAY_MLP_AR=0",
            "B70_XPU_FUSED_MLP_AR_NORM=0",
            "B70_XPU_PUSH_AR=1",
            "PUSH_AR_MIN_NUMEL=0",
            "PUSH_AR_MAXB=536870912",
            "PUSH_AR_GRAPH=0",
            "CCL_TOPO_P2P_ACCESS=0",
            "B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so",
            f"PUSH_AR_SO={self.manifest['push_ar_so']}",
        ]
        command = (
            f"exec server --model-path '{self.manifest['model']}' "
            f"--served-model-name '{served}' --tp 2 --context-length 131072 "
            "--max-running-requests 4 --disable-cuda-graph --disable-radix-cache "
            "--speculative-num-steps 10 --speculative-num-draft-tokens 11"
        )
        return {"Image": self.manifest["image_id"], "Config": {"Env": env, "Cmd": [command]}, "Mounts": mounts}

    def test_a_and_b_threshold_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inspect.json"
            for arm, candidate in (("01_A1", False), ("02_B1", True)):
                item = self.inspect_item(candidate)
                path.write_text(json.dumps([item]), encoding="ascii")
                self.assertTrue(analyzer.inspect_config(path, arm, self.manifest, REPO))
            item = self.inspect_item(True)
            item["Config"]["Env"].remove("B70_W8A16_ROUTE_DEBUG=0")
            path.write_text(json.dumps([item]), encoding="ascii")
            self.assertFalse(analyzer.inspect_config(path, "02_B1", self.manifest, REPO))

    def test_server_info_same_model_distinct_served_id(self):
        info = {
            "status": "ready",
            "model_path": self.manifest["model"],
            "served_model_name": self.manifest["served_b"],
            "context_length": 131072,
            "tp_size": 2,
            "pp_size": 1,
            "max_running_requests": 4,
            "disable_cuda_graph": True,
            "disable_radix_cache": True,
            "speculative_num_steps": 10,
            "speculative_num_draft_tokens": 11,
        }
        self.assertTrue(analyzer.server_info_exact(info, "02_B1", self.manifest))
        info["served_model_name"] = self.manifest["served_a"]
        self.assertFalse(analyzer.server_info_exact(info, "02_B1", self.manifest))


if __name__ == "__main__":
    unittest.main()
