#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


FAMILIES = ["q3_K", "q4_K", "q5_K", "q6_K", "q8_0", "iq3_s", "iq4_nl", "iq4_xs"]
DOMINANT = ["q5_K", "q8_0", "iq4_xs", "q4_K"]
IMAGE = "sha256:test"
MODEL = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
MODEL_SHA = "a" * 64


class AnalyzeVtuneGateTests(unittest.TestCase):
    def make_fixture(self, trace_speed: float = 90.0, attach: bool = False) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        out = Path(temporary.name)
        reference = out / "reference"
        traced = out / "vtune"
        profile = traced / "profile"
        reference.mkdir()
        profile.mkdir(parents=True)

        def write_json(path: Path, payload: object) -> None:
            path.write_text(json.dumps(payload) + "\n", encoding="ascii")

        write_json(
            out / "manifest.json",
            {"image": {"id": IMAGE}, "model": {"file": MODEL, "sha256": MODEL_SHA},
             "config": {"served": "hotschmoe-dd", "collection_mode": (
                 "attach_after_load" if attach else "launch_under")}},
        )
        write_json(out / "endpoint_down.json", {"passed": True})
        reference_inspect = [{"Image": IMAGE, "HostConfig": {
            "RestartPolicy": {"Name": "no"}, "CapAdd": None,
            "SecurityOpt": None, "Privileged": False, "PidMode": ""}}]
        traced_inspect = [{"Image": IMAGE, "HostConfig": {
            "RestartPolicy": {"Name": "no"},
            "CapAdd": ["SYS_PTRACE"] if attach else None,
            "SecurityOpt": None, "Privileged": False, "PidMode": ""}}]
        models = {"data": [{"id": "hotschmoe-dd"}]}
        env = {
            "MODEL_FILE": MODEL, "MODEL_SHA256": MODEL_SHA, "GPU_COUNT": "2",
            "ENABLE_MTP": "0", "LAB_DOORS": "0", "CCL_TOPO_P2P_ACCESS": "0",
            "GGML_SYCL_QUANT_CENSUS": "1", "GGML_SYCL_QUANT_TIMING_SAMPLE": "0",
            "GGML_SYCL_PROFILE": "0", "GGML_SYCL_DEBUG": "0",
            "PROFILE_VERBOSE": "0", "PROFILE_STATS": "0",
        }
        for directory, vtune, inspect in (
            (reference, "0", reference_inspect),
            (traced, "0" if attach else "1", traced_inspect),
        ):
            write_json(directory / "container_inspect.json", inspect)
            write_json(directory / "models.json", models)
            (directory / "container_env.txt").write_text(
                "".join(
                    f"{key}={value}\n"
                    for key, value in {
                        **env, "VTUNE_GPU_OFFLOAD": vtune,
                        **({"VTUNE_ATTACH_MODE": "1" if directory == traced else "0"} if attach else {}),
                    }.items()
                ),
                encoding="ascii",
            )
            write_json(directory / "warmup.json", {"passed": True})
            (directory / "server.log").write_text("clean\n", encoding="ascii")
            (directory / "health_post.log").write_text("PASS\n", encoding="ascii")

        def measurement(speed: float) -> dict:
            return {"passed": True, "result": {"completion_tokens": 512,
                    "text_sha256": "b" * 64, "post_first_tok_s": speed, "ttft_s": 1.0}}

        write_json(reference / "measure.json", measurement(100.0))
        write_json(traced / "measure.json", measurement(trace_speed))
        write_json(traced / "stop_contract.json", {"passed": True})
        write_json(
            traced / "tasks.json",
            {"passed": True, "adapters": ["Arc 0", "Arc 1"], "families": FAMILIES,
             "total_task_time_s": 1.0, "classified_quant_task_time_s": 0.8,
             "unknown_tasks": [],
             "by_adapter_family": [
                 {"adapter": adapter, "family": family, "total_time_s": 0.01}
                 for adapter in ("Arc 0", "Arc 1") for family in DOMINANT
             ]},
        )
        write_json(
            traced / "census.json",
            {"records": [{"kind": "logical", "type": family, "calls": 1} for family in FAMILIES],
             "computed_totals": {"logical_total": 8, "actual_total": 16}},
        )
        for path in (traced / "tasks.csv", traced / "summary.csv"):
            path.write_text("header\n", encoding="ascii")
        (profile / "vtune_version.txt").write_text("VTune 2025.10\n", encoding="ascii")
        (out / "health_pre.log").write_text("PASS\n", encoding="ascii")
        if attach:
            (out / "health_final.log").write_text("PASS\n", encoding="ascii")
        return out

    def analyze(self, out: Path) -> tuple[int, dict]:
        script = Path(__file__).parents[1] / "analyze_qwen38_ud_q4k_xl_vtune.py"
        process = subprocess.run(
            ["python3", str(script), "--out-dir", str(out)],
            check=False, capture_output=True, text=True,
        )
        return process.returncode, json.loads(process.stdout)

    def test_complete_fixture_passes(self) -> None:
        rc, result = self.analyze(self.make_fixture())
        self.assertEqual(rc, 0)
        self.assertTrue(result["passed"])

    def test_profiler_perturbation_fails(self) -> None:
        rc, result = self.analyze(self.make_fixture(trace_speed=80.0))
        self.assertEqual(rc, 1)
        self.assertFalse(result["checks"]["trace_speed_at_least_85pct"])

    def test_scoped_attach_fixture_passes(self) -> None:
        rc, result = self.analyze(self.make_fixture(attach=True))
        self.assertEqual(rc, 0)
        self.assertTrue(result["checks"]["attach_capability_scoped"])
        self.assertTrue(result["checks"]["attach_security_scoped"])


if __name__ == "__main__":
    unittest.main()
