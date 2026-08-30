#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qwen38_fp8_compile_oracle import analyze


SERVED = "qwen38-compile-oracle-test"


def write_attempt(
    result_root: Path,
    cache_root: Path,
    attempt: int,
    config: dict[str, object],
) -> None:
    result = result_root / f"attempt-{attempt}"
    result.mkdir()
    (result / "models.json").write_text(
        json.dumps({"data": [{"id": SERVED}]}), encoding="ascii"
    )
    (result / "smoke.json").write_text(
        json.dumps({"choices": [{"text": "READY"}]}), encoding="ascii"
    )
    cache = (
        cache_root
        / f"attempt-{attempt}"
        / "torch_compile_cache"
        / "torch_aot_compile"
        / "aot-key"
        / "inductor_cache"
    )
    cache.mkdir(parents=True)
    (cache / "kernel.best_config").write_text(json.dumps(config), encoding="ascii")


class CompileOracleTests(unittest.TestCase):
    def test_nonsemantic_timing_difference_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results, caches = root / "results", root / "caches"
            results.mkdir()
            base = {"XBLOCK": 32, "num_warps": 1, "time_taken_ms": 10}
            write_attempt(results, caches, 1, base)
            write_attempt(results, caches, 2, {**base, "time_taken_ms": 20})
            summary = analyze(results, caches, 2, SERVED, "test.v1", "test")
            self.assertEqual(summary["verdict"], "passed")
            self.assertTrue(summary["semantic_config_exact"])

    def test_semantic_difference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results, caches = root / "results", root / "caches"
            results.mkdir()
            write_attempt(results, caches, 1, {"XBLOCK": 32, "num_warps": 1})
            write_attempt(results, caches, 2, {"XBLOCK": 512, "num_warps": 8})
            summary = analyze(results, caches, 2, SERVED, "test.v1", "test")
            self.assertEqual(summary["verdict"], "failed_compile_selection_exactness")
            self.assertFalse(summary["semantic_config_exact"])


if __name__ == "__main__":
    unittest.main()
