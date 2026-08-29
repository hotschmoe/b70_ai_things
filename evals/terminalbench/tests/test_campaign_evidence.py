from __future__ import annotations

import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from evals.terminalbench.campaign_evidence import (
    build_lifecycle_record,
    parse_runtime_log,
    validate_identity,
)


class IdentityEvidenceTest(unittest.TestCase):
    def test_sglang_bf16_identity(self) -> None:
        model = "qwen-test-bf16kv"
        log = (
            "server_args=ServerArgs(model_path='/model', dtype='bfloat16', "
            "kv_cache_dtype='bfloat16', served_model_name='qwen-test-bf16kv')\n"
            "KV Cache is allocated. dtype: torch.bfloat16, #tokens: 100\n"
        )
        result = validate_identity(
            {"data": [{"id": model}]},
            log,
            expected_model=model,
            expected_target_dtype="bfloat16",
            expected_kv_dtype="bfloat16",
        )
        self.assertTrue(result["valid"])

    def test_vllm_auto_kv_without_observation_fails(self) -> None:
        model = "qwen-test-bf16kv"
        log = (
            "Initializing a V1 LLM engine (v0) with config: model='/model', "
            "dtype=torch.float16, kv_cache_dtype=auto, "
            "served_model_name=qwen-test-bf16kv, seed=0\n"
        )
        parsed = parse_runtime_log(log)
        self.assertEqual(parsed["target_dtype"], "float16")
        self.assertIsNone(parsed["observed_kv_dtype"])
        with self.assertRaisesRegex(ValueError, "target dtype"):
            validate_identity(
                {"data": [{"id": model}]},
                log,
                expected_model=model,
                expected_target_dtype="bfloat16",
                expected_kv_dtype="bfloat16",
            )

    def test_mislabeled_endpoint_fails(self) -> None:
        log = (
            "server_args=ServerArgs(dtype='bfloat16', kv_cache_dtype='bfloat16', "
            "served_model_name='actual')\n"
            "KV Cache is allocated. dtype: torch.bfloat16, #tokens: 100\n"
        )
        with self.assertRaisesRegex(ValueError, "/v1/models"):
            validate_identity(
                {"data": [{"id": "actual"}]},
                log,
                expected_model="claimed",
                expected_target_dtype="bfloat16",
                expected_kv_dtype="bfloat16",
            )


class LifecycleEvidenceTest(unittest.TestCase):
    def test_full_clock_includes_health_and_teardown(self) -> None:
        record = build_lifecycle_record(
            arm="mock",
            served_model="mock-model",
            exit_code=0,
            machine_start=100,
            prehealth_end=110,
            server_start=111,
            server_ready=120,
            harbor_end=150,
            preteardown_check=151,
            teardown_end=155,
            posthealth_end=170,
            endpoint_healthy_before_teardown=True,
            endpoint_down_after_teardown=True,
            pre_card_health=True,
            pre_collective_health=True,
            post_card_health=True,
            post_collective_health=True,
            fatal_server_markers=[],
        )
        self.assertEqual(record["full_machine_seconds"], 70)
        self.assertEqual(record["startup_seconds"], 9)
        self.assertEqual(record["harbor_seconds"], 30)
        self.assertTrue(record["health_contract_passed"])

    def test_mock_endpoint_lifecycle(self) -> None:
        class HealthyHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/health":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok\n")

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), HealthyHandler)
        thread = Thread(target=server.serve_forever)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/health"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                self.assertEqual(response.status, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        with self.assertRaises(urllib.error.URLError):
            urllib.request.urlopen(url, timeout=0.2)

        record = build_lifecycle_record(
            arm="mock-endpoint",
            served_model="mock-model",
            exit_code=0,
            machine_start=100,
            prehealth_end=101,
            server_start=102,
            server_ready=103,
            harbor_end=105,
            preteardown_check=106,
            teardown_end=107,
            posthealth_end=109,
            endpoint_healthy_before_teardown=True,
            endpoint_down_after_teardown=True,
            pre_card_health=True,
            pre_collective_health=True,
            post_card_health=True,
            post_collective_health=True,
            fatal_server_markers=[],
        )
        self.assertEqual(record["full_machine_seconds"], 9)
        self.assertTrue(record["health_contract_passed"])

    def test_nonmonotonic_clock_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "not monotonic"):
            build_lifecycle_record(
                arm="mock",
                served_model="mock-model",
                exit_code=1,
                machine_start=100,
                prehealth_end=99,
                server_start=None,
                server_ready=None,
                harbor_end=None,
                preteardown_check=None,
                teardown_end=None,
                posthealth_end=110,
                endpoint_healthy_before_teardown=None,
                endpoint_down_after_teardown=None,
                pre_card_health=False,
                pre_collective_health=False,
                post_card_health=True,
                post_collective_health=True,
                fatal_server_markers=[],
            )


if __name__ == "__main__":
    unittest.main()
