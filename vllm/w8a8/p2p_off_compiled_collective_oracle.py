#!/usr/bin/env python3
"""M02 two-rank P2P-off direct, compiled, and XPUGraph collective oracle."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol


ALL_REDUCE_SHAPES = ((1, 5120), (4, 5120))
ALL_GATHER_SHAPES = ((4, 2560),)
CONSUMER_MULTIPLIER = 2
CONSUMER_OFFSET = 1


@torch.library.custom_op("b70_m02::all_gather", mutates_args=())
def opaque_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Keep direct oneCCL all-gather opaque to Inductor during graph capture."""
    output = torch.empty(
        (tensor.shape[0] * dist.get_world_size(), *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(output, tensor.contiguous())
    return output


@opaque_all_gather.register_fake
def opaque_all_gather_fake(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (tensor.shape[0] * 2, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )


class RankEventLog:
    """Line-buffered evidence for tested collective entry and return events."""

    def __init__(self, path: Path, rank: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.rank = rank
        self.handle = path.open("w", encoding="ascii", buffering=1)
        self.next_call_id = 1
        self.open_calls: set[int] = set()
        self.signatures: list[dict[str, Any]] = []

    def enter(
        self,
        *,
        collective: str,
        shape: tuple[int, ...],
        mode: str,
        iteration: int,
    ) -> int:
        call_id = self.next_call_id
        self.next_call_id += 1
        signature = {
            "call_id": call_id,
            "collective": collective,
            "input_shape": list(shape),
            "mode": mode,
            "iteration": iteration,
        }
        self.open_calls.add(call_id)
        self.signatures.append(signature)
        self._write({**signature, "event": "entry"})
        return call_id

    def returned(self, call_id: int, *, equal: bool) -> None:
        if call_id not in self.open_calls:
            raise RuntimeError(f"return for unknown call_id={call_id}")
        self.open_calls.remove(call_id)
        self._write({"call_id": call_id, "event": "return", "equal": equal})

    def _write(self, event: dict[str, Any]) -> None:
        record = {
            "monotonic_ns": time.monotonic_ns(),
            "rank": self.rank,
            **event,
        }
        print(json.dumps(record, sort_keys=True), file=self.handle, flush=True)

    def close(self) -> None:
        self.handle.close()


def make_base(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return (
        torch.arange(elements, dtype=torch.int32, device=device)
        .remainder_(23)
        .to(torch.bfloat16)
        .reshape(shape)
    )


def make_input(base: torch.Tensor, rank: int, iteration: int) -> torch.Tensor:
    # These small integer-derived values and every expected result are exactly
    # representable in BF16. No tolerance can hide a completion race.
    return base + rank * 3 + (iteration % 7)


def expected_collective(
    *,
    collective: str,
    base: torch.Tensor,
    world_size: int,
    iteration: int,
) -> torch.Tensor:
    if collective == "all_reduce":
        rank_offset_sum = 3 * world_size * (world_size - 1) // 2
        return base * world_size + rank_offset_sum + world_size * (iteration % 7)
    if collective == "all_gather":
        return torch.cat(
            [make_input(base, rank, iteration) for rank in range(world_size)], dim=0
        )
    raise ValueError(f"unknown collective: {collective}")


def dependent_consumer(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * CONSUMER_MULTIPLIER + CONSUMER_OFFSET


def expected_consumer(
    *,
    collective: str,
    base: torch.Tensor,
    world_size: int,
    iteration: int,
) -> torch.Tensor:
    return dependent_consumer(
        expected_collective(
            collective=collective,
            base=base,
            world_size=world_size,
            iteration=iteration,
        )
    )


def eager_direct(collective: str, tensor: torch.Tensor) -> torch.Tensor:
    if collective == "all_reduce":
        dist.all_reduce(tensor)
        collective_output = tensor
    elif collective == "all_gather":
        collective_output = torch.empty(
            (tensor.shape[0] * dist.get_world_size(), *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(collective_output, tensor)
    else:
        raise ValueError(f"unknown collective: {collective}")
    return dependent_consumer(collective_output)


def functional_collective(collective: str, tensor: torch.Tensor) -> torch.Tensor:
    if collective == "all_reduce":
        pending = funcol.all_reduce(tensor, "sum", dist.group.WORLD)
    elif collective == "all_gather":
        pending = funcol.all_gather_tensor(tensor, 0, dist.group.WORLD)
    else:
        raise ValueError(f"unknown collective: {collective}")
    completed = funcol.wait_tensor(pending)
    return dependent_consumer(completed)


def equality_detail(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[bool, dict[str, Any] | None]:
    if torch.equal(actual, expected):
        return True, None
    difference = (actual.float() - expected.float()).abs().reshape(-1)
    indexes = difference.nonzero().reshape(-1)
    first_index = int(indexes[0].item()) if indexes.numel() else -1
    return False, {
        "flat_index": first_index,
        "actual": (
            float(actual.reshape(-1)[first_index].item()) if first_index >= 0 else None
        ),
        "expected": (
            float(expected.reshape(-1)[first_index].item())
            if first_index >= 0
            else None
        ),
        "max_abs_diff": float(difference.max().item()),
        "mismatch_elements": int(indexes.numel()),
    }


def validate_calls(
    *,
    collective: str,
    shape: tuple[int, ...],
    mode: str,
    iterations: int,
    rank: int,
    world_size: int,
    base: torch.Tensor,
    call: Callable[[torch.Tensor], torch.Tensor],
    logger: RankEventLog,
) -> dict[str, Any]:
    mismatches = 0
    first_mismatch = None
    start_call_id = logger.next_call_id
    start = time.perf_counter()
    for iteration in range(iterations):
        source = make_input(base, rank, iteration)
        call_id = logger.enter(
            collective=collective,
            shape=shape,
            mode=mode,
            iteration=iteration,
        )
        output = call(source)
        torch.xpu.synchronize()
        expected = expected_consumer(
            collective=collective,
            base=base,
            world_size=world_size,
            iteration=iteration,
        )
        equal, detail = equality_detail(output, expected)
        logger.returned(call_id, equal=equal)
        if not equal:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {"iteration": iteration, **(detail or {})}
    return {
        "mode": mode,
        "iterations": iterations,
        "call_id_first": start_call_id,
        "call_id_last": logger.next_call_id - 1,
        "mismatch_iterations": mismatches,
        "first_mismatch": first_mismatch,
        "elapsed_seconds": time.perf_counter() - start,
    }


def validate_compiled_graph(
    *,
    collective: str,
    shape: tuple[int, ...],
    iterations: int,
    rank: int,
    world_size: int,
    base: torch.Tensor,
    compiled_call: Callable[[torch.Tensor], torch.Tensor],
    route: str,
    logger: RankEventLog,
) -> dict[str, Any]:
    if not hasattr(torch.xpu, "XPUGraph") or not hasattr(torch.xpu, "graph"):
        return {
            "mode": "compiled_xpu_graph",
            "supported": False,
            "reason": "torch.xpu XPUGraph API is absent",
            "route": route,
            "iterations": 0,
            "mismatch_iterations": 0,
        }

    static_input = make_input(base, rank, 0)
    warmup_id = logger.enter(
        collective=collective,
        shape=shape,
        mode="compiled_xpu_graph_warmup",
        iteration=0,
    )
    warmup_output = compiled_call(static_input)
    torch.xpu.synchronize()
    warmup_expected = expected_consumer(
        collective=collective,
        base=base,
        world_size=world_size,
        iteration=0,
    )
    warmup_equal, warmup_detail = equality_detail(warmup_output, warmup_expected)
    logger.returned(warmup_id, equal=warmup_equal)
    if not warmup_equal:
        return {
            "mode": "compiled_xpu_graph",
            "supported": True,
            "route": route,
            "iterations": 0,
            "mismatch_iterations": 1,
            "first_mismatch": {"stage": "warmup", **(warmup_detail or {})},
        }

    dist.barrier()
    graph = torch.xpu.XPUGraph()
    capture_id = logger.enter(
        collective=collective,
        shape=shape,
        mode="compiled_xpu_graph_capture",
        iteration=0,
    )
    with torch.xpu.graph(graph):
        static_output = compiled_call(static_input)
    torch.xpu.synchronize()
    logger.returned(capture_id, equal=True)
    dist.barrier()

    mismatches = 0
    first_mismatch = None
    start_call_id = logger.next_call_id
    start = time.perf_counter()
    for iteration in range(iterations):
        static_input.copy_(make_input(base, rank, iteration))
        torch.xpu.synchronize()
        call_id = logger.enter(
            collective=collective,
            shape=shape,
            mode="compiled_xpu_graph_replay",
            iteration=iteration,
        )
        graph.replay()
        torch.xpu.synchronize()
        expected = expected_consumer(
            collective=collective,
            base=base,
            world_size=world_size,
            iteration=iteration,
        )
        equal, detail = equality_detail(static_output, expected)
        logger.returned(call_id, equal=equal)
        if not equal:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {"iteration": iteration, **(detail or {})}
    return {
        "mode": "compiled_xpu_graph",
        "supported": True,
        "route": route,
        "iterations": iterations,
        "call_id_first": start_call_id,
        "call_id_last": logger.next_call_id - 1,
        "mismatch_iterations": mismatches,
        "first_mismatch": first_mismatch,
        "elapsed_seconds": time.perf_counter() - start,
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lifetime", type=int, required=True)
    parser.add_argument("--eager-iterations", type=int, default=8)
    parser.add_argument("--compiled-iterations", type=int, default=8)
    parser.add_argument("--graph-iterations", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    for name in (
        "lifetime",
        "eager_iterations",
        "compiled_iterations",
        "graph_iterations",
        "timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"M02 requires world_size=2, got {world_size}")
    if os.environ.get("CCL_TOPO_P2P_ACCESS") != "0":
        raise RuntimeError("M02 requires CCL_TOPO_P2P_ACCESS=0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.output_dir / f"lifetime-{args.lifetime}.rank-{rank}.jsonl"
    rank_path = args.output_dir / f"lifetime-{args.lifetime}.rank-{rank}.json"
    combined_path = args.output_dir / f"lifetime-{args.lifetime}.json"
    logger = RankEventLog(event_path, rank)
    local_result: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "lifetime": args.lifetime,
        "event_log": str(event_path),
        "cases": [],
        "exception": None,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "B70_ORACLE_IMAGE",
                "CCL_ATL_TRANSPORT",
                "CCL_TOPO_P2P_ACCESS",
                "CCL_ZE_IPC_EXCHANGE",
                "FI_TCP_IFACE",
                "CCL_KVS_IFACE",
                "ONEAPI_DEVICE_SELECTOR",
                "ZE_AFFINITY_MASK",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
            )
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
        },
        "contract": {
            "dtype": "bfloat16",
            "all_reduce_shapes": [list(shape) for shape in ALL_REDUCE_SHAPES],
            "all_gather_input_shapes": [list(shape) for shape in ALL_GATHER_SHAPES],
            "functional_completion": "wait_tensor",
            "dependent_consumer": (
                f"collective_output * {CONSUMER_MULTIPLIER} + {CONSUMER_OFFSET}"
            ),
        },
    }

    initialized = False
    try:
        torch.xpu.set_device(local_rank)
        dist.init_process_group(
            backend="xccl", timeout=timedelta(seconds=args.timeout)
        )
        initialized = True
        device = torch.device(f"xpu:{local_rank}")
        cases = [
            ("all_reduce", shape) for shape in ALL_REDUCE_SHAPES
        ] + [("all_gather", shape) for shape in ALL_GATHER_SHAPES]

        for collective, shape in cases:
            base = make_base(shape, device)
            eager_result = validate_calls(
                collective=collective,
                shape=shape,
                mode="eager_direct",
                iterations=args.eager_iterations,
                rank=rank,
                world_size=world_size,
                base=base,
                call=lambda tensor, kind=collective: eager_direct(kind, tensor),
                logger=logger,
            )
            dist.barrier()

            def functional(tensor: torch.Tensor, kind: str = collective) -> torch.Tensor:
                return functional_collective(kind, tensor)

            compiled_call = torch.compile(
                functional, fullgraph=True, dynamic=False
            )
            compiled_result = validate_calls(
                collective=collective,
                shape=shape,
                mode="compiled_functional_wait_tensor",
                iterations=args.compiled_iterations,
                rank=rank,
                world_size=world_size,
                base=base,
                call=compiled_call,
                logger=logger,
            )
            dist.barrier()
            graph_result = validate_compiled_graph(
                collective=collective,
                shape=shape,
                iterations=args.graph_iterations,
                rank=rank,
                world_size=world_size,
                base=base,
                compiled_call=(
                    torch.compile(
                        lambda tensor: dependent_consumer(opaque_all_gather(tensor)),
                        fullgraph=True,
                        dynamic=False,
                    )
                    if collective == "all_gather"
                    else compiled_call
                ),
                route=(
                    "opaque_direct_all_gather_custom_op"
                    if collective == "all_gather"
                    else "functional_wait_tensor"
                ),
                logger=logger,
            )
            dist.barrier()
            local_result["cases"].append(
                {
                    "collective": collective,
                    "input_shape": list(shape),
                    "modes": [eager_result, compiled_result, graph_result],
                }
            )

        local_result["call_signatures"] = logger.signatures
        local_result["unreturned_call_ids"] = sorted(logger.open_calls)
        local_result["passed"] = not logger.open_calls and all(
            mode["mismatch_iterations"] == 0 and mode.get("supported", True)
            for case in local_result["cases"]
            for mode in case["modes"]
        )
        write_json(rank_path, local_result)

        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_result)
        signatures_match = all(
            result is not None
            and result["call_signatures"] == gathered[0]["call_signatures"]
            for result in gathered
        )
        passed = signatures_match and all(
            result is not None and result["passed"] for result in gathered
        )
        if rank == 0:
            combined = {
                "passed": passed,
                "lifetime": args.lifetime,
                "world_size": world_size,
                "backend": "xccl",
                "p2p": 0,
                "matched_rank_call_signatures": signatures_match,
                "ranks": gathered,
            }
            write_json(combined_path, combined)
            print(json.dumps(combined, sort_keys=True), flush=True)
        return 0 if passed else 1
    except BaseException as error:
        local_result["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        local_result["call_signatures"] = logger.signatures
        local_result["unreturned_call_ids"] = sorted(logger.open_calls)
        local_result["passed"] = False
        write_json(rank_path, local_result)
        print(
            f"M02_FAIL rank={rank} type={type(error).__name__} message={error}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        logger.close()
        if initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
