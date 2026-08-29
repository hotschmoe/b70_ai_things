#!/usr/bin/env python3
"""M03 P2P-off blocking versus explicit Work.wait completion oracle."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any

import torch
import torch.distributed as dist


SHAPES = ((1, 5120), (4, 5120))
MODES = ("blocking", "async_wait")
CONSUMER_MULTIPLIER = 2
CONSUMER_OFFSET = 1
EXPECTED_PHASES = {
    "blocking": ("entry", "collective_return", "consumer_return", "validation"),
    "async_wait": (
        "entry",
        "work_created",
        "wait_return",
        "consumer_return",
        "validation",
    ),
}


class RankEventLog:
    """Flushed per-rank event evidence with ordered, monotonic call IDs."""

    def __init__(self, path: Path, rank: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.rank = rank
        self.handle = path.open("w", encoding="ascii", buffering=1)
        self.next_call_id = 1
        self.open_calls: dict[int, dict[str, Any]] = {}
        self.signatures: list[dict[str, Any]] = []

    def enter(
        self,
        *,
        mode: str,
        shape: tuple[int, ...],
        stage: str,
        round_index: int,
        order_index: int,
    ) -> int:
        call_id = self.next_call_id
        self.next_call_id += 1
        signature = {
            "call_id": call_id,
            "collective": "all_reduce",
            "input_shape": list(shape),
            "mode": mode,
            "stage": stage,
            "round": round_index,
            "order_index": order_index,
        }
        self.signatures.append(signature)
        self.open_calls[call_id] = {"mode": mode, "phases": []}
        self.event(call_id, "entry", signature=signature)
        return call_id

    def event(self, call_id: int, event: str, **fields: Any) -> None:
        state = self.open_calls.get(call_id)
        if state is None:
            raise RuntimeError(f"event={event} for unknown call_id={call_id}")
        state["phases"].append(event)
        record = {
            "monotonic_ns": time.monotonic_ns(),
            "rank": self.rank,
            "call_id": call_id,
            "event": event,
            **fields,
        }
        print(json.dumps(record, sort_keys=True), file=self.handle, flush=True)
        if event == "validation":
            expected = EXPECTED_PHASES[state["mode"]]
            observed = tuple(state["phases"])
            if observed != expected:
                raise RuntimeError(
                    f"call_id={call_id} phase order {observed} != {expected}"
                )
            del self.open_calls[call_id]

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


def make_input(base: torch.Tensor, rank: int, round_index: int) -> torch.Tensor:
    return base + rank * 3 + (round_index % 7)


def expected_consumer(
    base: torch.Tensor, world_size: int, round_index: int
) -> torch.Tensor:
    rank_offset_sum = 3 * world_size * (world_size - 1) // 2
    reduced = (
        base * world_size
        + rank_offset_sum
        + world_size * (round_index % 7)
    )
    return reduced * CONSUMER_MULTIPLIER + CONSUMER_OFFSET


def dependent_consumer(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * CONSUMER_MULTIPLIER + CONSUMER_OFFSET


def tensor_sha256(tensor: torch.Tensor) -> str:
    # This runs after the validation synchronize and is excluded from latency.
    raw = bytes(
        tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()
    )
    return hashlib.sha256(raw).hexdigest()


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


def mode_order(lifetime: int, stage_index: int, round_index: int) -> tuple[str, str]:
    if (lifetime + stage_index + round_index) % 2 == 0:
        return MODES
    return tuple(reversed(MODES))


def run_one(
    *,
    mode: str,
    shape: tuple[int, ...],
    stage: str,
    stage_index: int,
    round_index: int,
    order_index: int,
    rank: int,
    world_size: int,
    base: torch.Tensor,
    logger: RankEventLog,
) -> dict[str, Any]:
    source = make_input(base, rank, round_index + stage_index * 97)
    expected = expected_consumer(
        base, world_size, round_index + stage_index * 97
    )
    # Input and reference preparation are outside the exploratory operator
    # timing. This is the only pre-call synchronize; there is deliberately no
    # synchronize between collective completion and the dependent consumer.
    torch.xpu.synchronize()

    call_id = logger.enter(
        mode=mode,
        shape=shape,
        stage=stage,
        round_index=round_index,
        order_index=order_index,
    )
    entry_ns = time.perf_counter_ns()
    timing: dict[str, float] = {}
    if mode == "blocking":
        dist.all_reduce(source)
        collective_return_ns = time.perf_counter_ns()
        timing["collective_call_return_us"] = (
            collective_return_ns - entry_ns
        ) / 1000.0
        logger.event(call_id, "collective_return")
        consumer_start_ns = collective_return_ns
    elif mode == "async_wait":
        work = dist.all_reduce(source, async_op=True)
        work_created_ns = time.perf_counter_ns()
        timing["async_launch_us"] = (work_created_ns - entry_ns) / 1000.0
        logger.event(call_id, "work_created")
        work.wait()
        wait_return_ns = time.perf_counter_ns()
        timing["wait_us"] = (wait_return_ns - work_created_ns) / 1000.0
        timing["launch_plus_wait_us"] = (wait_return_ns - entry_ns) / 1000.0
        logger.event(call_id, "wait_return")
        consumer_start_ns = wait_return_ns
    else:
        raise ValueError(f"unknown mode: {mode}")

    output = dependent_consumer(source)
    consumer_return_ns = time.perf_counter_ns()
    timing["consumer_launch_us"] = (
        consumer_return_ns - consumer_start_ns
    ) / 1000.0
    timing["entry_through_consumer_return_us"] = (
        consumer_return_ns - entry_ns
    ) / 1000.0
    logger.event(call_id, "consumer_return")

    # Synchronize only after the dependent consumer has been submitted. Exact
    # validation therefore detects a missing completion edge before consumer.
    torch.xpu.synchronize()
    equal, detail = equality_detail(output, expected)
    fingerprint = tensor_sha256(output)
    logger.event(
        call_id,
        "validation",
        equal=equal,
        output_sha256=fingerprint,
    )
    return {
        "call_id": call_id,
        "mode": mode,
        "input_shape": list(shape),
        "stage": stage,
        "round": round_index,
        "order_index": order_index,
        "equal": equal,
        "mismatch": detail,
        "output_sha256": fingerprint,
        "exploratory_host_latency_us": timing,
    }


def latency_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for shape in SHAPES:
        for mode in MODES:
            selected = [
                result
                for result in results
                if result["stage"] == "measurement"
                and result["input_shape"] == list(shape)
                and result["mode"] == mode
            ]
            metric_names = sorted(
                {
                    name
                    for result in selected
                    for name in result["exploratory_host_latency_us"]
                }
            )
            metrics = {}
            for name in metric_names:
                values = [
                    result["exploratory_host_latency_us"][name]
                    for result in selected
                ]
                metrics[name] = {
                    "minimum": min(values),
                    "median": statistics.median(values),
                    "maximum": max(values),
                }
            summaries.append(
                {
                    "input_shape": list(shape),
                    "mode": mode,
                    "calls": len(selected),
                    "metrics": metrics,
                }
            )
    return summaries


def cross_mode_exact(results: list[dict[str, Any]]) -> bool:
    indexed = {
        (
            tuple(result["input_shape"]),
            result["stage"],
            result["round"],
            result["mode"],
        ): result["output_sha256"]
        for result in results
    }
    for shape in SHAPES:
        for stage in ("warmup", "measurement"):
            rounds = {
                result["round"]
                for result in results
                if result["input_shape"] == list(shape) and result["stage"] == stage
            }
            for round_index in rounds:
                if indexed[(shape, stage, round_index, "blocking")] != indexed[
                    (shape, stage, round_index, "async_wait")
                ]:
                    return False
    return True


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lifetime", type=int, required=True)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measurement-rounds", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    for name in ("lifetime", "warmup_rounds", "measurement_rounds", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"M03 requires world_size=2, got {world_size}")
    if os.environ.get("CCL_TOPO_P2P_ACCESS") != "0":
        raise RuntimeError("M03 requires CCL_TOPO_P2P_ACCESS=0")

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
        "calls": [],
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
            )
        },
        "software": {"python": sys.version, "torch": torch.__version__},
        "contract": {
            "dtype": "bfloat16",
            "shapes": [list(shape) for shape in SHAPES],
            "modes": list(MODES),
            "warmup_rounds": args.warmup_rounds,
            "measurement_rounds": args.measurement_rounds,
            "consumer_before_validation_sync": True,
            "dependent_consumer": (
                f"collective_output * {CONSUMER_MULTIPLIER} + {CONSUMER_OFFSET}"
            ),
            "latency_status": "exploratory operator timing only",
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
        for shape in SHAPES:
            base = make_base(shape, device)
            for stage_index, (stage, rounds) in enumerate(
                (
                    ("warmup", args.warmup_rounds),
                    ("measurement", args.measurement_rounds),
                )
            ):
                for round_index in range(rounds):
                    order = mode_order(args.lifetime, stage_index, round_index)
                    for order_index, mode in enumerate(order):
                        local_result["calls"].append(
                            run_one(
                                mode=mode,
                                shape=shape,
                                stage=stage,
                                stage_index=stage_index,
                                round_index=round_index,
                                order_index=order_index,
                                rank=rank,
                                world_size=world_size,
                                base=base,
                                logger=logger,
                            )
                        )
                    dist.barrier()
            dist.barrier()

        local_result["call_signatures"] = logger.signatures
        local_result["unreturned_call_ids"] = sorted(logger.open_calls)
        local_result["cross_mode_exact"] = cross_mode_exact(local_result["calls"])
        local_result["exploratory_latency_summary_us"] = latency_summary(
            local_result["calls"]
        )
        local_result["passed"] = (
            not logger.open_calls
            and local_result["cross_mode_exact"]
            and all(call["equal"] for call in local_result["calls"])
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
                "latency_status": "exploratory operator timing only",
                "ranks": gathered,
            }
            write_json(combined_path, combined)
            print(
                "M03_LIFETIME_OK "
                f"lifetime={args.lifetime} ranks={world_size} "
                f"calls_per_rank={len(local_result['call_signatures'])} "
                "exact=true matched_rank_events=true",
                flush=True,
            )
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
            f"M03_FAIL rank={rank} type={type(error).__name__} message={error}",
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
