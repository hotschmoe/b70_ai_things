#!/usr/bin/env python3
"""CPU-only fixtures for the XL quant timing campaign analyzer."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_qwen38_ud_q4k_xl_quant_timing.py"
SPEC = importlib.util.spec_from_file_location("analyze_quant_timing_campaign", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def actual_cells() -> list[dict]:
    specs = [
        ("MMVQ", 0, "Q4_K", 1, 5120, 2048, 100),
        ("MMVQ", 1, "Q4_K", 1, 5120, 2048, 100),
        ("MMQ", 0, "Q5_K", 2, 2048, 1024, 80),
        ("MMQ", 1, "Q5_K", 2, 2048, 1024, 80),
    ]
    return [
        {
            "kind": "actual", "algo": algo, "device": device, "type": quant,
            "reordered": 0, "split": 1, "width": width, "K": k, "N": -1,
            "rows": rows, "calls": calls,
        }
        for algo, device, quant, width, k, rows, calls in specs
    ]


def census_payload() -> dict:
    records = actual_cells()
    total = sum(item["calls"] for item in records)
    return {
        "versions": ["1"],
        "records": records,
        "computed_totals": {"logical_total": total // 2, "actual_total": total},
        "declared_ends": [{"logical_total": total // 2, "actual_total": total}],
    }


def timing_payload(period: int) -> dict:
    means = [4000, 3000, 2500, 2000]
    records = []
    projected_total = 0
    for cell, mean in zip(actual_cells(), means, strict=True):
        projected_total += cell["calls"] * mean
    for cell, mean in zip(actual_cells(), means, strict=True):
        calls = cell["calls"]
        samples = 4
        projected = calls * mean
        records.append(
            {
                **{key: cell[key] for key in MODULE.KEY_FIELDS},
                "calls_seen": calls,
                "samples": samples,
                "device_ns": samples * mean,
                "barrier_ns": 20,
                "incomplete": 0,
                "invalid": 0,
                "mean_ns": mean,
                "min_ns": mean - 10,
                "max_ns": mean + 10,
                "projected_device_ns": projected,
                "projected_share": projected / projected_total,
            }
        )
    samples_total = sum(item["samples"] for item in records)
    device_ns = sum(item["device_ns"] for item in records)
    return {
        "headers": [{
            "version": "1", "scope": "standard_mul_mat", "sample_period": period,
            "skip": 4, "max_samples": 65536, "reserved": samples_total, "dropped": 0,
        }],
        "records": records,
        "summary": {
            "calls_seen": sum(item["calls_seen"] for item in records),
            "samples": samples_total,
            "sampled_device_ns": device_ns,
            "barrier_ns": 80,
            "incomplete": 0,
            "invalid": 0,
            "projected_device_ns": projected_total,
        },
        "declared_ends": [{"samples": samples_total, "device_ns": device_ns}],
    }


class AnalyzeQuantTimingCampaignTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> None:
        for arm in MODULE.ARMS:
            directory = root / arm
            directory.mkdir()
            image_id = "sha256:current" if arm == "current_off" else "sha256:candidate"
            write_json(directory / "identity.json", {
                "passed": True, "actual": {"image_id": image_id},
            })
            write_json(directory / "graceful_stop.json", {"passed": True})
            reps = 5 if arm in {"current_off", "candidate_off"} else 1
            speed = 100.0 if arm != "timing_128" else 98.0
            write_json(directory / "decode_profile.json", {
                "passed": True,
                "methodology": {
                    "temperature": 0, "ignore_eos": True, "gen_tokens": 256,
                    "warmup_tokens": 32,
                },
                "warmup": {"completion_tokens": 32},
                "runs": [
                    {"post_first_tok_s": speed, "completion_tokens": 256}
                    for _ in range(reps)
                ],
            })
        current_results = [
            {"text": f"answer-{index}", "coherent": index != 2}
            for index in range(7)
        ]
        write_json(root / "current_off" / "deterministic.json", {
            "passed": False, "results": current_results,
        })
        write_json(root / "candidate_off" / "deterministic.json", {
            "passed": True,
            "results": [
                {"text": item["text"], "coherent": item["coherent"], "exact_reference": True}
                for item in current_results
            ],
        })
        write_json(root / "counts_only" / "census.json", census_payload())
        for period in (64, 128, 256):
            directory = root / f"timing_{period}"
            write_json(directory / "census.json", census_payload())
            write_json(directory / "timing.json", timing_payload(period))
        write_json(root / "endpoint_down.json", {"passed": True})

    def test_complete_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root)
            result = MODULE.analyze(root)
            self.assertTrue(result["passed"])
            self.assertTrue(all(result["hard_gates"].values()))

    def test_dropped_sample_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root)
            path = root / "timing_128" / "timing.json"
            payload = json.loads(path.read_text(encoding="ascii"))
            payload["headers"][0]["dropped"] = 1
            write_json(path, payload)
            result = MODULE.analyze(root)
            self.assertFalse(result["passed"])
            self.assertFalse(result["hard_gates"]["timing_128_no_dropped_samples"])


if __name__ == "__main__":
    unittest.main()
