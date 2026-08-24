#!/usr/bin/env python3
"""CPU-only contract tests for the target-GDN INT8 A-B-B-A campaign."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SGLANG_DIR = Path(__file__).resolve().parents[1]


def load_analyzer():
    path = SGLANG_DIR / "analyze_c4_gdn_int8_abba.py"
    spec = importlib.util.spec_from_file_location("c4_gdn_abba", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = load_analyzer()


def row(phase, soak, cv=1.0):
    return {
        "phase_decode": phase,
        "soak_decode": soak,
        "phase_cv_pct": cv,
    }


class PerformanceGateTest(unittest.TestCase):
    def test_predeclared_thresholds_pass(self):
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
        checks = analyzer.make_performance_checks(rows, deltas)
        self.assertTrue(all(checks.values()), checks)

    def test_each_primary_gate_is_fail_closed(self):
        base_rows = {
            "01_A1": row(20.0, 20.0),
            "02_B1": row(20.8, 20.5),
            "03_B2": row(21.0, 20.4),
            "04_A2": row(20.1, 20.0),
        }
        base_deltas = {
            "phase_decode_pct": 3.5,
            "soak_decode_pct": 2.1,
            "phase_ttft_ms_pct": 0.0,
            "perf_c1_ttft_ms_pct": 0.0,
            "perf_c4_ttft_ms_pct": 0.0,
            "prefill_c1_ttft_ms_pct": 0.0,
            "prefill_c4_ttft_ms_pct": 0.0,
        }
        cases = []
        rows = {key: dict(value) for key, value in base_rows.items()}
        rows["02_B1"]["phase_decode"] = 19.9
        cases.append((rows, dict(base_deltas), "phase_both_pairs_win"))
        deltas = dict(base_deltas, phase_decode_pct=2.99)
        cases.append((base_rows, deltas, "balanced_phase_gain_ge_3pct"))
        rows = {key: dict(value) for key, value in base_rows.items()}
        rows["03_B2"]["soak_decode"] = 19.9
        cases.append((rows, dict(base_deltas), "soak_both_pairs_nonregress"))
        deltas = dict(base_deltas, soak_decode_pct=1.99)
        cases.append((base_rows, deltas, "balanced_soak_gain_ge_2pct"))
        deltas = dict(base_deltas, prefill_c4_ttft_ms_pct=-5.01)
        cases.append((base_rows, deltas, "ttft_and_prefill_within_5pct"))
        rows = {key: dict(value) for key, value in base_rows.items()}
        rows["02_B1"]["phase_cv_pct"] = 5.01
        cases.append((rows, dict(base_deltas), "b_within_process_cv_le_5pct"))
        for rows, deltas, failed_key in cases:
            with self.subTest(failed_key=failed_key):
                checks = analyzer.make_performance_checks(rows, deltas)
                self.assertFalse(checks[failed_key], checks)


class RunnerContractTest(unittest.TestCase):
    def test_runner_preserves_promoted_and_disables_experimental_flags(self):
        script = (SGLANG_DIR / "02_c4_gdn_int8_abba.sh").read_text(
            encoding="ascii"
        )
        for required in (
            "REPLICATE_MTP_EMBED=1",
            "PUSH_AR=1 PUSH_AR_MIN_NUMEL=0",
            "DELAY_MLP_AR=0 FUSED_MLP_AR_NORM=0 LMHEAD_INT8=0",
            "run_arm 01_A1 A",
            "run_arm 02_B1 B",
            "run_arm 03_B2 B",
            "run_arm 04_A2 A",
            "require_external_dual_card_lease",
            "endpoint_policy=down_between_arms_and_after_campaign_no_restore",
        ):
            self.assertIn(required, script)


class IdentityFixtureTest(unittest.TestCase):
    def setUp(self):
        self.repo = SGLANG_DIR.parent
        self.manifest = {
            "image_id": "sha256:test",
            "model_a": "/models/base",
            "model_b": "/models/candidate",
            "served_a": "base-id",
            "served_b": "candidate-id",
            "overlay": "/results/candidate_config.json",
            "ctx": "131072",
            "maxreq": "4",
            "kdir": "/runtime/kernel",
            "pushdir": "/runtime/push",
            "push_ar_so": "/work/push_ar/libxpu_push_ar_graph.so",
        }

    def inspect_item(self, candidate):
        model = self.manifest["model_b" if candidate else "model_a"]
        served = self.manifest["served_b" if candidate else "served_a"]
        mounts = [
            {
                "Destination": "/work/kernel",
                "Source": self.manifest["kdir"],
                "RW": False,
            },
            {
                "Destination": "/work/push_ar",
                "Source": self.manifest["pushdir"],
                "RW": False,
            },
        ]
        for name in (
            "w8a8_shim.py",
            "mtp_replicated_embedding.py",
            "push_ar_xpu.py",
        ):
            mounts.append(
                {
                    "Destination": f"/opt/venv/lib/python3.12/site-packages/{name}",
                    "Source": str(self.repo / f"sglang/patches/{name}"),
                    "RW": False,
                }
            )
        if candidate:
            mounts.append(
                {
                    "Destination": f"{model}/config.json",
                    "Source": self.manifest["overlay"],
                    "RW": False,
                }
            )
        env = [
            "B70_XPU_W8A8=1",
            "B70_XPU_W8A8_FUSED=1",
            "B70_W8A8_QUANT_LMHEAD=0",
            "B70_XPU_REPLICATE_MTP_EMBED=1",
            "B70_XPU_DELAY_MLP_AR=0",
            "B70_XPU_FUSED_MLP_AR_NORM=0",
            "B70_XPU_PUSH_AR=1",
            "PUSH_AR_MIN_NUMEL=0",
            "PUSH_AR_GRAPH=0",
            "CCL_TOPO_P2P_ACCESS=0",
            "B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so",
            f"PUSH_AR_SO={self.manifest['push_ar_so']}",
        ]
        command = (
            f"exec server --model-path '{model}' --served-model-name '{served}' "
            "--tp 2 --context-length 131072 --max-running-requests 4 "
            "--disable-cuda-graph --disable-radix-cache "
            "--speculative-num-steps 10 --speculative-num-draft-tokens 11"
        )
        return {
            "Image": self.manifest["image_id"],
            "Config": {"Env": env, "Cmd": ["bash", "-c", command]},
            "Mounts": mounts,
        }

    def test_a_and_b_inspect_contracts_and_overlay_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inspect.json"
            for arm, candidate in (("01_A1", False), ("02_B1", True)):
                item = self.inspect_item(candidate)
                path.write_text(json.dumps([item]), encoding="ascii")
                self.assertTrue(
                    analyzer.inspect_config(path, arm, self.manifest, self.repo)
                )
            item = self.inspect_item(True)
            item["Mounts"][-1]["RW"] = True
            path.write_text(json.dumps([item]), encoding="ascii")
            self.assertFalse(
                analyzer.inspect_config(path, "02_B1", self.manifest, self.repo)
            )
            item = self.inspect_item(False)
            item["Mounts"].append(
                {
                    "Destination": "/models/base/config.json",
                    "Source": self.manifest["overlay"],
                    "RW": False,
                }
            )
            path.write_text(json.dumps([item]), encoding="ascii")
            self.assertFalse(
                analyzer.inspect_config(path, "01_A1", self.manifest, self.repo)
            )

    def test_server_info_identity_contract(self):
        info = {
            "status": "ready",
            "model_path": self.manifest["model_b"],
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
        info["model_path"] = self.manifest["model_a"]
        self.assertFalse(analyzer.server_info_exact(info, "02_B1", self.manifest))


if __name__ == "__main__":
    unittest.main()
