#!/usr/bin/env python3
"""Static contracts for the queue-profiling isolation mechanism."""

import unittest
from pathlib import Path


LLAMACPP = Path(__file__).resolve().parents[1]


class QueueProfileIsolationStaticTest(unittest.TestCase):
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
