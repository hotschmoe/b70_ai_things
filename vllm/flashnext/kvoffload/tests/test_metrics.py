"""CPU checks for the Prometheus scrape used by the eviction test."""

from __future__ import annotations

import unittest

from kvoffload.metrics import counter_delta, scrape_counter


SAMPLE = """
# HELP vllm:prefix_cache_hits Prefix cache hits.
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{engine="0",model_name="hotschmoe-dev"} 100
vllm:prefix_cache_hits_created{engine="0",model_name="hotschmoe-dev"} 10
vllm:external_prefix_cache_hits_total{engine="0",model_name="hotschmoe-dev"} 0
vllm:prompt_tokens_cached_by_source_total{engine="0",model_name="hotschmoe-dev",source="device"} 100
vllm:prompt_tokens_cached_by_source_total{engine="0",model_name="hotschmoe-dev",source="host"} 0
"""

AFTER = SAMPLE.replace(
    'vllm:external_prefix_cache_hits_total{engine="0",model_name="hotschmoe-dev"} 0',
    'vllm:external_prefix_cache_hits_total{engine="0",model_name="hotschmoe-dev"} 29376',
).replace(
    'source="host"} 0',
    'source="host"} 29376',
).replace(
    'vllm:prefix_cache_hits_total{engine="0",model_name="hotschmoe-dev"} 100',
    'vllm:prefix_cache_hits_total{engine="0",model_name="hotschmoe-dev"} 100',
)


class MetricsTest(unittest.TestCase):
    def test_ignores_created_and_comments(self) -> None:
        self.assertEqual(scrape_counter(SAMPLE, "vllm:prefix_cache_hits"), 100)
        self.assertEqual(
            scrape_counter(SAMPLE, "vllm:prompt_tokens_cached_by_source", 'source="host"'),
            0,
        )

    def test_delta(self) -> None:
        self.assertEqual(
            counter_delta(SAMPLE, AFTER, "vllm:external_prefix_cache_hits"),
            29376,
        )
        self.assertEqual(
            counter_delta(
                SAMPLE,
                AFTER,
                "vllm:prompt_tokens_cached_by_source",
                'source="host"',
            ),
            29376,
        )
        self.assertEqual(
            counter_delta(SAMPLE, AFTER, "vllm:prefix_cache_hits"),
            0,
        )

    def test_missing_host_series_is_zero(self) -> None:
        # Prometheus omits source="host" until the first external hit.
        self.assertEqual(
            counter_delta("", AFTER, "vllm:prompt_tokens_cached_by_source", 'source="host"'),
            29376,
        )
        self.assertEqual(
            counter_delta(SAMPLE, SAMPLE, "vllm:prompt_tokens_cached_by_source", 'source="disk"'),
            0,
        )


if __name__ == "__main__":
    unittest.main()
