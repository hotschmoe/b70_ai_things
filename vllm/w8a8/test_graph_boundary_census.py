#!/usr/bin/env python3
"""Unit tests for the graph boundary census."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_boundary_census as census


def make_trace(path: Path, *, submissions: int = 2) -> None:
    events = []
    for index in range(5):
        start = 1000.0 * index
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": f"execute_context_{index}",
                "pid": 7,
                "tid": 9,
                "ts": start,
                "dur": 900.0,
            }
        )
        names = (
            ["zeFenceReset"]
            + ["zeEventHostSynchronize"] * 2
            + ["zeCommandQueueExecuteCommandLists"] * submissions
        )
        for offset, name in enumerate(names, start=1):
            events.append(
                {
                    "ph": "X",
                    "cat": "xpu_driver",
                    "name": name,
                    "pid": 7,
                    "tid": 9,
                    "ts": start + offset,
                    "dur": 0.5,
                }
            )
    path.write_text(json.dumps({"traceEvents": events}), encoding="ascii")


def make_graph(path: Path, *, calls: int = 2) -> None:
    lines = [
        f'    value_{index}: "bf16[s18, 5120]" = '
        "torch.ops.vllm.all_reduce(input_value, group_name='tp:0')"
        for index in range(calls)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class CensusTests(unittest.TestCase):
    def test_rank_agreement_and_overhead(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = {}
            graphs = {}
            for rank in (0, 1):
                trace = root / f"rank-{rank}.json"
                graph = root / f"rank-{rank}.py"
                make_trace(trace)
                make_graph(graph)
                traces[rank] = trace
                graphs[rank] = graph
            report = census.build_report(
                traces,
                graphs,
                skip_iterations=2,
                decode_rows=1,
                control_value=100.0,
                profiled_value=90.0,
                minimum_profiled_ratio=0.75,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(
                report["traces"]["0"]["per_target_token"],
                {
                    "graph_pieces": 1,
                    "fences": 1,
                    "host_waits": 2,
                    "submissions": 2,
                },
            )
            graph = report["compiled_graphs"]["0"]
            self.assertEqual(graph["collective_calls_per_target_token"], 2)
            self.assertEqual(graph["collectives"][0]["shape"], [1, 5120])

    def test_rank_structure_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = {}
            graphs = {}
            for rank, submissions in ((0, 2), (1, 3)):
                trace = root / f"rank-{rank}.json"
                graph = root / f"rank-{rank}.py"
                make_trace(trace, submissions=submissions)
                make_graph(graph)
                traces[rank] = trace
                graphs[rank] = graph
            report = census.build_report(
                traces,
                graphs,
                skip_iterations=2,
                decode_rows=1,
                control_value=None,
                profiled_value=None,
                minimum_profiled_ratio=0.75,
            )
            self.assertFalse(report["passed"])
            self.assertFalse(report["rank_structural_agreement"])

    def test_rank_collective_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = {}
            graphs = {}
            for rank, calls in ((0, 2), (1, 1)):
                trace = root / f"rank-{rank}.json"
                graph = root / f"rank-{rank}.py"
                make_trace(trace)
                make_graph(graph, calls=calls)
                traces[rank] = trace
                graphs[rank] = graph
            report = census.build_report(
                traces,
                graphs,
                skip_iterations=2,
                decode_rows=1,
                control_value=None,
                profiled_value=None,
                minimum_profiled_ratio=0.75,
            )
            self.assertFalse(report["passed"])
            self.assertFalse(report["rank_collective_agreement"])

    def test_unstable_iteration_signature_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "rank-0.json"
            make_trace(trace)
            payload = json.loads(trace.read_text(encoding="ascii"))
            payload["traceEvents"].append(
                {
                    "ph": "X",
                    "cat": "xpu_driver",
                    "name": "zeFenceReset",
                    "pid": 7,
                    "tid": 9,
                    "ts": 3006.0,
                    "dur": 0.5,
                }
            )
            trace.write_text(json.dumps(payload), encoding="ascii")
            report = census.trace_census(trace, skip_iterations=2)
            self.assertFalse(report["stable_signature"])


if __name__ == "__main__":
    unittest.main()
