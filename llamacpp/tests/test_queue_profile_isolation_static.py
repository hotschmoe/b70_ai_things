#!/usr/bin/env python3
"""Static contracts for the queue-profiling isolation mechanism."""

import re
import unittest
from pathlib import Path


LLAMACPP = Path(__file__).resolve().parents[1]


class QueueProfileIsolationStaticTest(unittest.TestCase):
    def assert_patch_hunks_exact(self, relative_path: str) -> None:
        lines = (LLAMACPP / relative_path).read_text(encoding="ascii").splitlines()
        index = 0
        found = 0
        while index < len(lines):
            match = re.match(
                r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[index]
            )
            if match is None:
                index += 1
                continue
            found += 1
            cursor = index + 1
            old_count = 0
            new_count = 0
            while (
                cursor < len(lines)
                and not lines[cursor].startswith("@@ ")
                and not lines[cursor].startswith("diff --git ")
            ):
                marker = lines[cursor][:1]
                if marker == " ":
                    old_count += 1
                    new_count += 1
                elif marker == "-":
                    old_count += 1
                elif marker == "+":
                    new_count += 1
                elif marker != "\\":
                    self.fail(f"malformed patch line {cursor + 1}: {lines[cursor]!r}")
                cursor += 1
            declared_old = int(match.group(2) or 1)
            declared_new = int(match.group(4) or 1)
            self.assertEqual(
                (old_count, new_count),
                (declared_old, declared_new),
                f"bad hunk count at {relative_path}:{index + 1}",
            )
            index = cursor
        self.assertGreater(found, 0, f"no hunks found in {relative_path}")

    def test_instrumentation_patch_hunk_counts_are_exact(self) -> None:
        self.assert_patch_hunks_exact("qwen38-b70/patches/quant-census.patch")
        self.assert_patch_hunks_exact("qwen38-b70/patches/quant-timing.patch")

    def test_patch_decouples_queue_property_from_sampling(self) -> None:
        patch = (LLAMACPP / "qwen38-b70/patches/quant-timing.patch").read_text(
            encoding="ascii"
        )
        self.assertIn('std::getenv("GGML_SYCL_QUANT_TIMING_QUEUE_PROFILE")', patch)
        self.assertIn("q != nullptr", patch)
        self.assertIn("quant_queue_profile", patch)
        self.assertLess(
            patch.index("seen <= skip"),
            patch.index("sycl_quant_timing_token { key, stream->ext_oneapi_submit_barrier()"),
        )

    def test_launcher_passes_exact_profile_and_gpu_count(self) -> None:
        launcher = (LLAMACPP / "serve_qwen38_stock_q4km_tp2.sh").read_text(
            encoding="ascii"
        )
        for required in (
            'GPU_COUNT="${GPU_COUNT:-2}"',
            'GGML_SYCL_QUANT_TIMING_QUEUE_PROFILE="${GGML_SYCL_QUANT_TIMING_QUEUE_PROFILE:-auto}"',
            'runtime_env+=(-e "GGML_SYCL_QUANT_TIMING_QUEUE_PROFILE=',
            '-e GPU_COUNT="$GPU_COUNT"',
            '--restart "$RESTART_POLICY"',
            '[ "$GPU_COUNT" = "1" ] && health_args=(--card 0)',
        ):
            self.assertIn(required, launcher)

    def test_harness_is_two_arm_no_restart_and_no_barrier(self) -> None:
        runner = (LLAMACPP / "04_qwen38_ud_q4k_xl_queue_profile_isolation.sh").read_text(
            encoding="ascii"
        )
        for required in (
            "TIMING_SKIP=18446744073709551615",
            "run_arm off 0",
            "run_arm on 1",
            "RESTART_POLICY=no",
            'case "$GPU_COUNT" in 1|2)',
            "require_external_gpu_run",
            'ACTION="${1:-static}"',
            'ai.b70.quant_timing_patch_sha',
            "health_probe after_off",
            "health_probe final",
            "check_code_hashes",
            "apply --numstat",
        ):
            self.assertIn(required, runner)

    def test_build_labels_exact_timing_patch(self) -> None:
        build = (LLAMACPP / "qwen38-b70/build_image.sh").read_text(encoding="ascii")
        self.assertIn("QUANT_TIMING_PATCH_SHA=", build)
        self.assertIn('ai.b70.quant_timing_patch_sha=$QUANT_TIMING_PATCH_SHA', build)


if __name__ == "__main__":
    unittest.main()
