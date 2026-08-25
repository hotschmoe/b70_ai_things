#!/usr/bin/env python3
"""Validate the vLLM custom-op layer above the proven raw oneCCL oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
import torch.distributed as dist


def make_input(base: torch.Tensor, rank: int, iteration: int) -> torch.Tensor:
    return base + rank * 3 + (iteration % 17)


def expected_output(
    base: torch.Tensor, world_size: int, iteration: int
) -> torch.Tensor:
    rank_sum = 3 * world_size * (world_size - 1) // 2
    return base * world_size + rank_sum + world_size * (iteration % 17)


def validate_calls(
    *,
    name: str,
    call: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    static_input: torch.Tensor,
    rank: int,
    world_size: int,
    iterations: int,
    use_graph: bool,
) -> dict[str, Any]:
    graph: torch.xpu.XPUGraph | None = None
    static_output: torch.Tensor | None = None
    if use_graph:
        static_input.copy_(make_input(base, rank, 0))
        torch.xpu.synchronize()
        dist.barrier()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            static_output = call(static_input)
        torch.xpu.synchronize()
        dist.barrier()

    mismatches = 0
    input_mutations = 0
    output_aliases = 0
    first_mismatch: dict[str, Any] | None = None
    start = time.perf_counter()
    for iteration in range(iterations):
        source = make_input(base, rank, iteration)
        static_input.copy_(source)
        torch.xpu.synchronize()
        if graph is None:
            output = call(static_input)
        else:
            graph.replay()
            assert static_output is not None
            output = static_output
        torch.xpu.synchronize()
        expected = expected_output(base, world_size, iteration)
        output_ok = torch.equal(output, expected)
        input_ok = torch.equal(static_input, source)
        if output.data_ptr() == static_input.data_ptr():
            output_aliases += 1
        if not input_ok:
            input_mutations += 1
        if not output_ok:
            mismatches += 1
            if first_mismatch is None:
                diff = (output.float() - expected.float()).abs().reshape(-1)
                indexes = diff.nonzero().reshape(-1)
                index = int(indexes[0].item()) if indexes.numel() else -1
                first_mismatch = {
                    "iteration": iteration,
                    "flat_index": index,
                    "max_abs_diff": float(diff.max().item()),
                }

    torch.xpu.synchronize()
    return {
        "name": name,
        "iterations": iterations,
        "mismatch_iterations": mismatches,
        "input_mutation_iterations": input_mutations,
        "output_alias_iterations": output_aliases,
        "first_mismatch": first_mismatch,
        "avg_iteration_ms_including_sync_and_validation": (
            (time.perf_counter() - start) * 1000.0 / iterations
        ),
    }


def validate_many_collectives(
    *,
    call: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    static_input: torch.Tensor,
    rank: int,
    world_size: int,
    collective_count: int,
    iterations: int,
) -> list[dict[str, Any]]:
    def many(input_: torch.Tensor) -> torch.Tensor:
        # Match the Qwen backbone's sequential collective execution without
        # retaining every 32 MiB profile-run output at once.
        result = torch.zeros_like(input_)
        for offset in range(collective_count):
            result = result + call(input_ + offset)
        return result

    def expected_many(input_base: torch.Tensor, iteration: int) -> torch.Tensor:
        result = torch.zeros_like(input_base)
        for offset in range(collective_count):
            collective_output = expected_output(input_base, world_size, iteration)
            collective_output = collective_output + offset * world_size
            result = result + collective_output
        return result

    compiled_many = torch.compile(many, fullgraph=True)
    for warmup in range(2):
        static_input.copy_(make_input(base, rank, warmup))
        compiled_many(static_input)
    torch.xpu.synchronize()
    dist.barrier()

    # vLLM's failing profile run executes 81 compiled [8192,2048] BF16
    # collectives before graph capture. Exercise that exact shape and count.
    profile_rows = 8192
    profile_base = (
        torch.arange(
            profile_rows * base.shape[1], dtype=torch.int32, device=base.device
        )
        .remainder_(31)
        .to(torch.bfloat16)
        .reshape(profile_rows, base.shape[1])
    )
    profile_input = make_input(profile_base, rank, 0)
    profile_source = profile_input.clone()
    start = time.perf_counter()
    profile_output = compiled_many(profile_input)
    torch.xpu.synchronize()
    profile_elapsed_ms = (time.perf_counter() - start) * 1000.0
    profile_expected = expected_many(profile_base, 0)
    profile_ok = torch.equal(profile_output, profile_expected)
    profile_input_ok = torch.equal(profile_input, profile_source)
    profile_alias = profile_output.data_ptr() == profile_input.data_ptr()
    profile_result = {
        "name": "compiled_direct_81_collective_profile",
        "shape": [profile_rows, int(base.shape[1])],
        "collectives": collective_count,
        "iterations": 1,
        "mismatch_iterations": 0 if profile_ok else 1,
        "input_mutation_iterations": 0 if profile_input_ok else 1,
        "output_alias_iterations": 1 if profile_alias else 0,
        "elapsed_ms_including_compile_sync_and_validation": profile_elapsed_ms,
    }
    del profile_output, profile_expected, profile_source, profile_input, profile_base
    torch.xpu.empty_cache()
    dist.barrier()

    static_input.copy_(make_input(base, rank, 0))
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        static_output = compiled_many(static_input)
    torch.xpu.synchronize()
    dist.barrier()

    mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    start = time.perf_counter()
    for iteration in range(iterations):
        source = make_input(base, rank, iteration)
        static_input.copy_(source)
        torch.xpu.synchronize()
        graph.replay()
        torch.xpu.synchronize()
        expected = expected_many(base, iteration)
        if not torch.equal(static_output, expected):
            mismatches += 1
            if first_mismatch is None:
                diff = (static_output.float() - expected.float()).abs()
                first_mismatch = {
                    "iteration": iteration,
                    "max_abs_diff": float(diff.max().item()),
                }

    graph_result = {
        "name": "compiled_graph_many",
        "collectives_per_replay": collective_count,
        "iterations": iterations,
        "mismatch_iterations": mismatches,
        "first_mismatch": first_mismatch,
        "avg_replay_ms_including_sync_and_validation": (
            (time.perf_counter() - start) * 1000.0 / iterations
        ),
    }
    return [profile_result, graph_result]


def write_partial(path: Path | None, rank: int, document: dict[str, Any]) -> None:
    if path is None:
        return
    partial = path.with_name(f"{path.stem}.rank{rank}.partial.json")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--eager-iterations", type=int, default=64)
    parser.add_argument("--compiled-iterations", type=int, default=64)
    parser.add_argument("--large-shape-iterations", type=int, default=4)
    parser.add_argument("--graph-iterations", type=int, default=256)
    parser.add_argument("--collective-count", type=int, default=81)
    parser.add_argument("--many-iterations", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"this oracle requires world_size=2, got {world_size}")

    torch.xpu.set_device(local_rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    local_result: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "stages": [],
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CCL_TOPO_P2P_ACCESS",
                "CCL_ZE_IPC_EXCHANGE",
                "VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES",
                "VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP",
                "VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT",
                "VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT",
            )
        },
    }

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(backend="xccl")
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
            backend="xccl",
        )
        warmup = torch.ones(1, dtype=torch.float32, device=f"xpu:{local_rank}")
        dist.all_reduce(warmup)
        torch.xpu.synchronize()
        if warmup.item() != float(world_size):
            raise RuntimeError(f"one-element worker warmup mismatch: {warmup.item()}")
        local_result["worker_warmup"] = {
            "shape": [1],
            "dtype": "float32",
            "value": float(warmup.item()),
        }
        group = get_tp_group()
        local_result["group"] = {
            "unique_name": group.unique_name,
            "use_custom_op_call": group.use_custom_op_call,
            "device_communicator": type(group.device_communicator).__name__,
        }
        if not group.use_custom_op_call:
            raise RuntimeError("XPU outer custom-op collective route is not active")

        device = torch.device(f"xpu:{local_rank}")
        base = (
            torch.arange(
                args.rows * args.hidden_size, dtype=torch.int32, device=device
            )
            .remainder_(31)
            .to(torch.bfloat16)
            .reshape(args.rows, args.hidden_size)
        )
        static_input = torch.empty_like(base)

        export = torch._dynamo.export(tensor_model_parallel_all_reduce, static_input)
        exported_graph = str(export.graph_module.graph)
        local_result["exported_graph"] = exported_graph
        if "vllm.s2b_all_reduce_clone" not in exported_graph:
            raise RuntimeError("compiled route does not contain s2b_all_reduce_clone")
        if "torch.ops.vllm.all_reduce" in exported_graph:
            raise RuntimeError("compiled route still contains stock vllm.all_reduce")

        eager = validate_calls(
            name="group_custom_op_eager",
            call=tensor_model_parallel_all_reduce,
            base=base,
            static_input=static_input,
            rank=rank,
            world_size=world_size,
            iterations=args.eager_iterations,
            use_graph=False,
        )
        local_result["stages"].append(eager)
        write_partial(args.output, rank, local_result)
        print(f"rank={rank} stage={eager}", flush=True)

        compiled = torch.compile(tensor_model_parallel_all_reduce, fullgraph=True)
        for warmup in range(2):
            static_input.copy_(make_input(base, rank, warmup))
            compiled(static_input)
        torch.xpu.synchronize()
        dist.barrier()

        compiled_direct = validate_calls(
            name="group_custom_op_compiled",
            call=compiled,
            base=base,
            static_input=static_input,
            rank=rank,
            world_size=world_size,
            iterations=args.compiled_iterations,
            use_graph=False,
        )
        local_result["stages"].append(compiled_direct)
        write_partial(args.output, rank, local_result)
        print(f"rank={rank} stage={compiled_direct}", flush=True)

        for rows in (4, 8192):
            shape_base = (
                torch.arange(
                    rows * args.hidden_size, dtype=torch.int32, device=device
                )
                .remainder_(31)
                .to(torch.bfloat16)
                .reshape(rows, args.hidden_size)
            )
            shape_input = torch.empty_like(shape_base)
            shape_stage = validate_calls(
                name=f"group_custom_op_compiled_{rows}x{args.hidden_size}",
                call=compiled,
                base=shape_base,
                static_input=shape_input,
                rank=rank,
                world_size=world_size,
                iterations=args.large_shape_iterations,
                use_graph=False,
            )
            local_result["stages"].append(shape_stage)
            write_partial(args.output, rank, local_result)
            print(f"rank={rank} stage={shape_stage}", flush=True)

        compiled_graph = validate_calls(
            name="group_custom_op_compiled_graph",
            call=compiled,
            base=base,
            static_input=static_input,
            rank=rank,
            world_size=world_size,
            iterations=args.graph_iterations,
            use_graph=True,
        )
        local_result["stages"].append(compiled_graph)
        write_partial(args.output, rank, local_result)
        print(f"rank={rank} stage={compiled_graph}", flush=True)

        many_stages = validate_many_collectives(
            call=tensor_model_parallel_all_reduce,
            base=base,
            static_input=static_input,
            rank=rank,
            world_size=world_size,
            collective_count=args.collective_count,
            iterations=args.many_iterations,
        )
        local_result["stages"].extend(many_stages)
        write_partial(args.output, rank, local_result)
        for stage in many_stages:
            print(f"rank={rank} stage={stage}", flush=True)

        all_results: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(all_results, local_result)
        passed = all(
            result is not None
            and all(
                stage["mismatch_iterations"] == 0
                and stage.get("input_mutation_iterations", 0) == 0
                and stage.get("output_alias_iterations", 0) == 0
                for stage in result["stages"]
            )
            for result in all_results
        )
        if rank == 0:
            document = {
                "passed": passed,
                "backend": "xccl",
                "world_size": world_size,
                "ranks": all_results,
            }
            rendered = json.dumps(document, indent=2, sort_keys=True)
            print(rendered)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n")

        destroy_model_parallel()
        destroy_distributed_environment()
        return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
