#!/usr/bin/env python3
"""Correctness, stress, and latency gate for TP=2 fused push-AR Gemma RMSNorm.

The candidate shared library must expose the existing 118 setup/current-AR ABI
and the new fixed-shape fused ABI::

    int ar_allreduce_residual_gemma_rmsnorm_bf16(
        uint64_t current_queue_addr,
        uint64_t inout_local_partial,
        uint64_t residual,
        uint64_t raw_gemma_weight,
        int32_t rows,
        int32_t hidden,
        float eps);

The gold path is the same library's ar_allreduce_ptr_dt followed by the exact
image sgl_kernel.gemma_fused_add_rmsnorm implementation. No tolerance is hidden:
residual and cross-rank results are required bit-exact, while any candidate/gold
output mismatch is reported as a BF16 ULP distance.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import socket
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch

try:
    import intel_extension_for_pytorch  # noqa: F401
except Exception:
    pass
try:
    import oneccl_bindings_for_pytorch  # noqa: F401
except Exception:
    pass

import sgl_kernel
import torch.distributed as dist
import torch.multiprocessing as mp


DT_BF16 = 1


@dataclass
class CaseTensors:
    local: torch.Tensor
    residual: torch.Tensor
    weight: torch.Tensor


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[pos]


def _tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _bf16_ordered_key(tensor: torch.Tensor) -> torch.Tensor:
    bits = tensor.detach().cpu().contiguous().view(torch.int16).to(torch.int32)
    bits = torch.bitwise_and(bits, 0xFFFF)
    magnitude = torch.bitwise_and(bits, 0x7FFF)
    return torch.where(
        torch.bitwise_and(bits, 0x8000) != 0,
        0x8000 - magnitude,
        0x8000 + magnitude,
    )


def _bf16_compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().contiguous()
    expected_cpu = expected.detach().cpu().contiguous()
    if actual_cpu.shape != expected_cpu.shape:
        raise AssertionError(f"shape mismatch: {actual_cpu.shape} != {expected_cpu.shape}")
    equal = actual_cpu.view(torch.int16) == expected_cpu.view(torch.int16)
    mismatch = int((~equal).sum().item())
    if mismatch == 0:
        return {"bit_exact": True, "mismatch_elements": 0, "max_bf16_ulp": 0}

    finite = torch.isfinite(actual_cpu.float()) & torch.isfinite(expected_cpu.float())
    comparable = (~equal) & finite
    if bool(comparable.any().item()):
        ulp = torch.abs(
            _bf16_ordered_key(actual_cpu) - _bf16_ordered_key(expected_cpu)
        )
        max_ulp = int(ulp[comparable].max().item())
    else:
        max_ulp = None
    return {
        "bit_exact": False,
        "mismatch_elements": mismatch,
        "max_bf16_ulp": max_ulp,
        "nonfinite_mismatch_elements": int(((~equal) & (~finite)).sum().item()),
    }


def _reference(
    rank0: torch.Tensor,
    rank1: torch.Tensor,
    residual: torch.Tensor,
    raw_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar contract with both required BF16 rounding points."""
    ar_bf16 = (rank0.float() + rank1.float()).to(torch.bfloat16)
    residual_out = (ar_bf16.float() + residual.float()).to(torch.bfloat16)
    variance = residual_out.float().pow(2).mean(dim=-1, keepdim=True)
    output = residual_out.float() * torch.rsqrt(variance + eps)
    output = output * (1.0 + raw_weight.float())
    return output.to(torch.bfloat16), residual_out


def _make_case(
    name: str,
    rows: int,
    hidden: int,
    rank: int,
    seed: int,
) -> CaseTensors:
    if name == "random":
        gen_local = torch.Generator().manual_seed(seed + 101 * rank)
        gen_shared = torch.Generator().manual_seed(seed + 10007)
        local = torch.randn((rows, hidden), generator=gen_local) * 0.35
        residual = torch.randn((rows, hidden), generator=gen_shared) * 0.5
        weight = torch.randn((hidden,), generator=gen_shared) * 0.1
    elif name == "cancellation":
        gen = torch.Generator().manual_seed(seed)
        base = torch.randn((rows, hidden), generator=gen) * 8.0
        delta = torch.randn((rows, hidden), generator=gen) * (2.0**-7)
        local = base if rank == 0 else -base + delta
        residual = torch.randn((rows, hidden), generator=gen) * (2.0**-5)
        weight = torch.linspace(-0.25, 0.25, hidden)
    elif name == "half_ulp":
        index = torch.arange(rows * hidden, dtype=torch.int64).reshape(rows, hidden)
        exponent = (index % 7).to(torch.float32) - 3.0
        base = torch.pow(torch.tensor(2.0), exponent)
        sign = torch.where((index % 2) == 0, 1.0, -1.0)
        if rank == 0:
            local = sign * base
        else:
            # base/256 is exactly half one BF16 ULP at base.
            local = sign * base / 256.0
        residual = torch.where((index % 3) == 0, base / 256.0, -base / 256.0)
        weight = ((torch.arange(hidden) % 17).float() - 8.0) / 64.0
    elif name == "log_scale":
        index = torch.arange(rows * hidden, dtype=torch.int64).reshape(rows, hidden)
        exponent = ((index % 33).to(torch.float32) - 16.0) / 2.0
        value = torch.pow(torch.tensor(2.0), exponent)
        sign = torch.where(((index // 3 + rank) % 2) == 0, 1.0, -1.0)
        local = sign * value * (1.0 if rank == 0 else 0.75)
        residual = torch.flip(value, dims=(-1,)) * 0.125
        weight = torch.sin(torch.arange(hidden).float() * 0.03125) * 0.2
    else:
        raise ValueError(name)

    return CaseTensors(
        local=local.to(torch.bfloat16).contiguous(),
        residual=residual.to(torch.bfloat16).contiguous(),
        weight=weight.to(torch.bfloat16).contiguous(),
    )


class FusedLibrary:
    def __init__(
        self,
        path: str,
        rank: int,
        queue_addr: int,
        max_bytes: int,
        sock_path: str,
    ) -> None:
        self.lib = ctypes.CDLL(path)
        self.lib.ar_setup_torch.restype = ctypes.c_int
        self.lib.ar_setup_torch.argtypes = [
            ctypes.c_int,
            ctypes.c_ulonglong,
            ctypes.c_long,
        ]
        self.lib.ar_exchange.restype = ctypes.c_int
        self.lib.ar_exchange.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.ar_allreduce_ptr_dt.restype = None
        self.lib.ar_allreduce_ptr_dt.argtypes = [
            ctypes.c_ulonglong,
            ctypes.c_long,
            ctypes.c_int,
        ]
        self.lib.ar_allreduce_residual_gemma_rmsnorm_bf16.restype = ctypes.c_int
        self.lib.ar_allreduce_residual_gemma_rmsnorm_bf16.argtypes = [
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
        ]
        if hasattr(self.lib, "ar_teardown"):
            self.lib.ar_teardown.argtypes = []

        rc = self.lib.ar_setup_torch(rank, ctypes.c_ulonglong(queue_addr), max_bytes)
        if rc == 0:
            rc = self.lib.ar_exchange(rank, sock_path.encode())
        if rc != 0:
            raise RuntimeError(f"candidate setup failed on rank {rank}: rc={rc}")

    def gold(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        self.lib.ar_allreduce_ptr_dt(
            ctypes.c_ulonglong(x.data_ptr()),
            x.numel() * x.element_size(),
            DT_BF16,
        )
        sgl_kernel.gemma_fused_add_rmsnorm(x, residual, weight, eps)

    def candidate(
        self,
        queue_addr: int,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        rows: int,
        hidden: int,
        eps: float,
    ) -> bool:
        x_ptr = x.data_ptr()
        residual_ptr = residual.data_ptr()
        if x_ptr == residual_ptr:
            raise AssertionError("input and residual unexpectedly alias each other")
        rc = self.lib.ar_allreduce_residual_gemma_rmsnorm_bf16(
            ctypes.c_ulonglong(queue_addr),
            ctypes.c_ulonglong(x_ptr),
            ctypes.c_ulonglong(residual_ptr),
            ctypes.c_ulonglong(weight.data_ptr()),
            rows,
            hidden,
            ctypes.c_float(eps),
        )
        if rc == 38:
            return False
        if rc != 0:
            raise RuntimeError(f"candidate call failed: rc={rc}, rows={rows}")
        if x.data_ptr() != x_ptr or residual.data_ptr() != residual_ptr:
            raise AssertionError("candidate changed an in-place tensor data_ptr")
        return True

    def teardown(self) -> None:
        if hasattr(self.lib, "ar_teardown"):
            self.lib.ar_teardown()


def _gather_objects(value: Any, group) -> list[Any]:
    gathered = [None, None]
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _run_exact_cases(rank: int, args, library: FusedLibrary, queue_addr: int, gloo_group):
    device = torch.device(f"xpu:{rank}")
    reports = []
    local_hashes = []
    for rows in args.rows:
        for case_index, case_name in enumerate(args.cases):
            seed = args.seed + rows * 1009 + case_index * 65537
            cpu_case = _make_case(case_name, rows, args.hidden, rank, seed)
            local = cpu_case.local.to(device)
            residual = cpu_case.residual.to(device)
            weight = cpu_case.weight.to(device)

            dist.barrier()
            gold_x = local.clone()
            gold_residual = residual.clone()
            library.gold(gold_x, gold_residual, weight, args.eps)
            torch.xpu.synchronize()

            dist.barrier()
            candidate_x = local.clone()
            candidate_residual = residual.clone()
            x_ptr = candidate_x.data_ptr()
            residual_ptr = candidate_residual.data_ptr()
            engaged = library.candidate(
                queue_addr,
                candidate_x,
                candidate_residual,
                weight,
                rows,
                args.hidden,
                args.eps,
            )
            if not engaged:
                raise AssertionError(f"eligible M={rows} unexpectedly returned fallback")
            torch.xpu.synchronize()
            if candidate_x.data_ptr() != x_ptr or candidate_residual.data_ptr() != residual_ptr:
                raise AssertionError("candidate in-place alias contract failed after synchronize")

            residual_cmp = _bf16_compare(candidate_residual, gold_residual)
            if not residual_cmp["bit_exact"]:
                raise AssertionError(
                    f"rank {rank} {case_name} M={rows}: residual is not bit-exact: {residual_cmp}"
                )
            output_cmp = _bf16_compare(candidate_x, gold_x)
            hashes = {
                "case": case_name,
                "rows": rows,
                "gold_output": _tensor_hash(gold_x),
                "gold_residual": _tensor_hash(gold_residual),
                "candidate_output": _tensor_hash(candidate_x),
                "candidate_residual": _tensor_hash(candidate_residual),
            }
            local_hashes.append(hashes)
            reports.append(
                {
                    "case": case_name,
                    "rows": rows,
                    "residual": residual_cmp,
                    "output": output_cmp,
                    "pointer_alias_preserved": True,
                }
            )

    rank_hashes = _gather_objects(local_hashes, gloo_group)
    if rank_hashes[0] != rank_hashes[1]:
        for left, right in zip(rank_hashes[0], rank_hashes[1]):
            if left != right:
                raise AssertionError(f"cross-rank exact-case mismatch: rank0={left}, rank1={right}")
        raise AssertionError("cross-rank exact-case hash list length mismatch")
    return reports


def _run_fallback_cases(rank: int, args, library: FusedLibrary, queue_addr: int):
    device = torch.device(f"xpu:{rank}")
    reports = []
    for rows in args.fallback_rows:
        cpu_case = _make_case("random", rows, args.hidden, rank, args.seed + rows)
        x = cpu_case.local.to(device)
        residual = cpu_case.residual.to(device)
        weight = cpu_case.weight.to(device)
        before_x = _tensor_hash(x)
        before_residual = _tensor_hash(residual)
        engaged = library.candidate(
            queue_addr, x, residual, weight, rows, args.hidden, args.eps
        )
        torch.xpu.synchronize()
        if engaged:
            raise AssertionError(f"M={rows} should return fallback status 38")
        if _tensor_hash(x) != before_x or _tensor_hash(residual) != before_residual:
            raise AssertionError(f"M={rows} fallback mutated an input tensor")
        reports.append({"rows": rows, "status": 38, "inputs_unchanged": True})
    return reports


def _run_stress(rank: int, args, library: FusedLibrary, queue_addr: int, gloo_group):
    device = torch.device(f"xpu:{rank}")
    delay_rng = random.Random(args.seed + 9001 + rank)
    local_hashes = []
    max_output_ulp = 0
    output_mismatches = 0
    completed = 0

    for chunk_start in range(0, args.stress_calls, args.stress_chunk):
        pending = []
        chunk_end = min(args.stress_calls, chunk_start + args.stress_chunk)
        for call in range(chunk_start, chunk_end):
            rows = args.rows[call % len(args.rows)]
            case_name = args.cases[(call // len(args.rows)) % len(args.cases)]
            seed = args.seed + 1000003 + call * 8191
            cpu_local = _make_case(case_name, rows, args.hidden, rank, seed)
            cpu_rank0 = _make_case(case_name, rows, args.hidden, 0, seed)
            cpu_rank1 = _make_case(case_name, rows, args.hidden, 1, seed)
            expected_x, expected_residual = _reference(
                cpu_rank0.local,
                cpu_rank1.local,
                cpu_local.residual,
                cpu_local.weight,
                args.eps,
            )

            x = cpu_local.local.to(device)
            residual = cpu_local.residual.to(device)
            weight = cpu_local.weight.to(device)
            if rank == 1 and args.delay_max_us > 0 and call % args.delay_every == 0:
                time.sleep(delay_rng.uniform(0.0, args.delay_max_us / 1_000_000.0))
            engaged = library.candidate(
                queue_addr,
                x,
                residual,
                weight,
                rows,
                args.hidden,
                args.eps,
            )
            if not engaged:
                raise AssertionError(f"stress M={rows} unexpectedly returned fallback")
            # These clones are ordered consumers, not host synchronizations. They
            # preserve every result while subsequent calls exercise ring reuse.
            pending.append((call, x.clone(), residual.clone(), expected_x, expected_residual))

        torch.xpu.synchronize()
        for call, output, residual, expected_x, expected_residual in pending:
            residual_cmp = _bf16_compare(residual, expected_residual)
            if not residual_cmp["bit_exact"]:
                raise AssertionError(
                    f"rank {rank} stress call {call}: residual mismatch: {residual_cmp}"
                )
            output_cmp = _bf16_compare(output, expected_x)
            if not output_cmp["bit_exact"]:
                output_mismatches += output_cmp["mismatch_elements"]
                if output_cmp["max_bf16_ulp"] is not None:
                    max_output_ulp = max(max_output_ulp, output_cmp["max_bf16_ulp"])
            local_hashes.append(
                (
                    call,
                    _tensor_hash(output),
                    _tensor_hash(residual),
                )
            )
            completed += 1

    rank_hashes = _gather_objects(local_hashes, gloo_group)
    if rank_hashes[0] != rank_hashes[1]:
        for left, right in zip(rank_hashes[0], rank_hashes[1]):
            if left != right:
                raise AssertionError(f"cross-rank stress mismatch: rank0={left}, rank1={right}")
        raise AssertionError("cross-rank stress hash list length mismatch")
    return {
        "calls": completed,
        "chunk": args.stress_chunk,
        "delay_max_us": args.delay_max_us,
        "delay_every": args.delay_every,
        "residual_bit_exact": True,
        "cross_rank_bit_exact": True,
        "output_cpu_reference_mismatch_elements": output_mismatches,
        "output_cpu_reference_max_bf16_ulp": max_output_ulp,
    }


def _run_latency_mode(
    rank: int,
    args,
    library: FusedLibrary,
    queue_addr: int,
    rows: int,
    mode: str,
) -> tuple[float, float]:
    device = torch.device(f"xpu:{rank}")
    cpu_case = _make_case("random", rows, args.hidden, rank, args.seed + 700000 + rows)
    base_x = cpu_case.local.to(device)
    base_residual = cpu_case.residual.to(device)
    weight = cpu_case.weight.to(device)
    x = base_x.clone()
    residual = base_residual.clone()

    dist.barrier()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(args.latency_calls):
        x.copy_(base_x)
        residual.copy_(base_residual)
        if mode == "gold":
            library.gold(x, residual, weight, args.eps)
        else:
            engaged = library.candidate(
                queue_addr,
                x,
                residual,
                weight,
                rows,
                args.hidden,
                args.eps,
            )
            if not engaged:
                raise AssertionError(f"latency M={rows} unexpectedly returned fallback")
    caller_ms = 1000.0 * (time.perf_counter() - start)
    torch.xpu.synchronize()
    total_ms = 1000.0 * (time.perf_counter() - start)
    return (
        caller_ms * 1000.0 / args.latency_calls,
        total_ms * 1000.0 / args.latency_calls,
    )


def _run_latency(rank: int, args, library: FusedLibrary, queue_addr: int):
    result = []
    for rows in args.rows:
        samples = {"gold": [], "candidate": []}
        # Position-balance the two modes after untimed warmups.
        for warm_mode in ("gold", "candidate"):
            for _ in range(args.latency_warmup):
                _run_latency_mode(rank, args, library, queue_addr, rows, warm_mode)
        for repeat in range(args.latency_repeats):
            order = ("gold", "candidate") if repeat % 2 == 0 else ("candidate", "gold")
            for mode in order:
                samples[mode].append(
                    _run_latency_mode(rank, args, library, queue_addr, rows, mode)
                )
        row_result = {"rows": rows, "calls_per_sample": args.latency_calls}
        for mode in ("gold", "candidate"):
            caller = [sample[0] for sample in samples[mode]]
            total = [sample[1] for sample in samples[mode]]
            row_result[mode] = {
                "caller_us_per_call_median": statistics.median(caller),
                "caller_us_per_call_p95": _percentile(caller, 0.95),
                "synchronized_us_per_call_median": statistics.median(total),
                "synchronized_us_per_call_p95": _percentile(total, 0.95),
            }
        gold_total = row_result["gold"]["synchronized_us_per_call_median"]
        candidate_total = row_result["candidate"]["synchronized_us_per_call_median"]
        row_result["synchronized_speedup"] = gold_total / candidate_total
        result.append(row_result)
    return result


def _worker(rank: int, world: int, args, result_queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", rank=rank, world_size=world)
    gloo_group = dist.new_group(ranks=list(range(world)), backend="gloo")
    queue_addr = int(torch.xpu.current_stream().sycl_queue)
    max_bytes = max(args.rows) * args.hidden * 2
    library = FusedLibrary(
        args.candidate_so,
        rank,
        queue_addr,
        max_bytes,
        args.sock,
    )
    dist.barrier()

    exact = _run_exact_cases(rank, args, library, queue_addr, gloo_group)
    fallback = _run_fallback_cases(rank, args, library, queue_addr)
    stress = _run_stress(rank, args, library, queue_addr, gloo_group)
    latency = _run_latency(rank, args, library, queue_addr)

    dist.barrier()
    torch.xpu.synchronize()
    library.teardown()
    result_queue.put(
        (
            rank,
            {
                "exact_cases": exact,
                "fallback_cases": fallback,
                "stress": stress,
                "latency": latency,
            },
        )
    )
    dist.destroy_process_group(gloo_group)
    dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-so", required=True)
    parser.add_argument("--rows", default="1,2")
    parser.add_argument("--fallback-rows", default="3,11,44,64,128")
    parser.add_argument("--cases", default="random,cancellation,half_ulp,log_scale")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--stress-calls", type=int, default=256)
    parser.add_argument("--stress-chunk", type=int, default=8)
    parser.add_argument("--delay-max-us", type=float, default=500.0)
    parser.add_argument("--delay-every", type=int, default=7)
    parser.add_argument("--latency-calls", type=int, default=159)
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--sock", default=f"/tmp/b70_fused_ar_rmsnorm_{os.getpid()}.sock")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    args.rows = [int(value) for value in args.rows.split(",") if value]
    args.fallback_rows = [
        int(value) for value in args.fallback_rows.split(",") if value
    ]
    args.cases = [value for value in args.cases.split(",") if value]
    if args.hidden != 5120:
        raise ValueError("the fixed candidate ABI is restricted to hidden=5120")
    if not args.rows or args.rows != sorted(set(args.rows)):
        raise ValueError("rows must be a nonempty sorted unique list")
    if args.fallback_rows != sorted(set(args.fallback_rows)):
        raise ValueError("fallback-rows must be a sorted unique list")
    if any(row < 1 or row > 128 for row in args.rows + args.fallback_rows):
        raise ValueError("all row counts must be in 1..128")
    if set(args.rows) & set(args.fallback_rows):
        raise ValueError("rows and fallback-rows must be disjoint")
    if args.stress_chunk < 2:
        raise ValueError("stress-chunk must be >=2 to exercise scratch-ring reuse")
    if not os.path.exists(args.candidate_so):
        raise FileNotFoundError(args.candidate_so)

    ctx = mp.get_context("spawn")
    result_queue = ctx.SimpleQueue()
    mp.spawn(_worker, args=(2, args, result_queue), nprocs=2, join=True)
    rank_results = dict(result_queue.get() for _ in range(2))
    output_exact = all(
        case["output"]["bit_exact"]
        for rank_result in rank_results.values()
        for case in rank_result["exact_cases"]
    )
    max_output_ulp = max(
        (
            case["output"]["max_bf16_ulp"] or 0
            for rank_result in rank_results.values()
            for case in rank_result["exact_cases"]
        ),
        default=0,
    )
    summary = {
        "hostname": socket.gethostname(),
        "candidate_so": args.candidate_so,
        "hidden": args.hidden,
        "rows": args.rows,
        "fallback_rows": args.fallback_rows,
        "cases": args.cases,
        "gold": "ar_allreduce_ptr_dt + sgl_kernel.gemma_fused_add_rmsnorm",
        "residual_bit_exact_required": True,
        "cross_rank_bit_exact_required": True,
        "candidate_output_bit_exact": output_exact,
        "candidate_output_max_bf16_ulp": max_output_ulp,
        "rank_results": rank_results,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if not output_exact:
        print(
            "OUTPUT IS NOT BIT-EXACT; see candidate_output_max_bf16_ulp and per-case mismatches above",
            flush=True,
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
