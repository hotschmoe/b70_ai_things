#!/usr/bin/env python3
"""Time USM-host H2D and D2H. One buffer, then the same bytes in 1 MiB runs.

The pass line is COPY_BENCH. Bandwidth is decimal GB/s over the sync wait
on the copy queue. A run below 7.5 GB/s is reported and is not a pass.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time

import torch

from b70_kvpark.ops import DEFAULT_SO, load


def _fill(ptr: int, nbytes: int, salt: int) -> None:
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    for i in range(0, nbytes, 4096):
        buf[i] = (salt + i) & 0xFF


def _check(ptr: int, nbytes: int, salt: int) -> int:
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    bad = 0
    for i in range(0, nbytes, 4096):
        if buf[i] != ((salt + i) & 0xFF):
            bad += 1
            if bad > 8:
                break
    return bad


def _ptrs(values: list[int]) -> torch.Tensor:
    # Xe device pointers sit above 2^63-1. A signed int64 tensor rejects
    # them. The uint64 bit pattern, viewed as int64, is what copy_batch
    # casts back through uintptr_t.
    packed = torch.tensor(
        [v & 0xFFFFFFFFFFFFFFFF for v in values], dtype=torch.uint64
    )
    return packed.view(torch.int64)


def _time_copy(src: int, dst: int, nbytes: int, runs: int, sync: bool) -> float:
    src_t = _ptrs([src])
    dst_t = _ptrs([dst])
    sz_t = torch.tensor([nbytes], dtype=torch.int64)
    torch.xpu.synchronize()
    torch.ops.b70_kvpark.copy_batch(src_t, dst_t, sz_t, True)
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        torch.ops.b70_kvpark.copy_batch(src_t, dst_t, sz_t, sync)
    if not sync:
        torch.ops.b70_kvpark.copy_batch(src_t, dst_t, sz_t, True)
    elapsed = time.perf_counter() - start
    return nbytes * runs / elapsed / 1e9


def _time_fragments(src: int, dst: int, nbytes: int, piece: int, runs: int) -> float:
    n = nbytes // piece
    src_t = _ptrs([src + i * piece for i in range(n)])
    dst_t = _ptrs([dst + i * piece for i in range(n)])
    sz_t = torch.full((n,), piece, dtype=torch.int64)
    torch.xpu.synchronize()
    torch.ops.b70_kvpark.copy_batch(src_t, dst_t, sz_t, True)
    start = time.perf_counter()
    for _ in range(runs):
        torch.ops.b70_kvpark.copy_batch(src_t, dst_t, sz_t, True)
    elapsed = time.perf_counter() - start
    return nbytes * runs / elapsed / 1e9


def _time_swap(src: int, dst: int, nbytes: int, runs: int) -> float:
    # XPU swap_blocks_batch takes uint64 pointer tensors. The same USM-host
    # address is what the parking worker passes today.
    src_t = torch.tensor([src], dtype=torch.uint64)
    dst_t = torch.tensor([dst], dtype=torch.uint64)
    sz_t = torch.tensor([nbytes], dtype=torch.uint64)
    from vllm import _custom_ops as ops

    torch.xpu.synchronize()
    ops.swap_blocks_batch(src_t, dst_t, sz_t)
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        ops.swap_blocks_batch(src_t, dst_t, sz_t)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start
    return nbytes * runs / elapsed / 1e9


def main() -> int:
    so = os.environ.get("B70_KVPARK_SO", DEFAULT_SO)
    print(f"bench_copy: so={so}", flush=True)
    load()
    torch.empty(1, device="xpu")
    ordinal = int(torch.ops.b70_kvpark.engine_ordinal())
    try:
        ordered = int(torch.ops.b70_kvpark.ordered())
    except Exception as exc:
        ordered = -1
        print(f"bench_copy: ordered missing ({exc})", flush=True)
    print(f"bench_copy: engine_ordinal={ordinal} ordered={ordered}", flush=True)
    if ordinal < 0:
        print("bench_copy: copy engine missing", flush=True)
    sizes = [16 << 20, 64 << 20, 256 << 20, 1 << 30]
    runs_for = {16 << 20: 20, 64 << 20: 10, 256 << 20: 4, 1 << 30: 2}
    worst = 1e9
    for nbytes in sizes:
        host = int(torch.ops.b70_kvpark.alloc_host(nbytes)) & 0xFFFFFFFFFFFFFFFF
        back = int(torch.ops.b70_kvpark.alloc_host(nbytes)) & 0xFFFFFFFFFFFFFFFF
        dev = torch.empty(nbytes, dtype=torch.uint8, device="xpu")
        print(
            f"bench_copy: host_ptr={host:#x} dev_ptr={dev.data_ptr():#x}",
            flush=True,
        )
        _fill(host, nbytes, 7)
        h2d = _time_copy(host, dev.data_ptr(), nbytes, runs_for[nbytes], True)
        d2h = _time_copy(dev.data_ptr(), back, nbytes, runs_for[nbytes], True)
        bad = _check(back, nbytes, 7)
        frag = _time_fragments(host, dev.data_ptr(), min(nbytes, 64 << 20), 1 << 20, 4)
        swap_h2d = swap_d2h = -1.0
        try:
            swap_h2d = _time_swap(host, dev.data_ptr(), nbytes, runs_for[nbytes])
            swap_d2h = _time_swap(dev.data_ptr(), back, nbytes, runs_for[nbytes])
        except Exception as exc:
            print(f"bench_copy: swap_skip bytes={nbytes} {exc}", flush=True)
        print(
            f"bench_copy: bytes={nbytes} h2d={h2d:.3f} d2h={d2h:.3f} "
            f"frag_1MiB={frag:.3f} swap_h2d={swap_h2d:.3f} "
            f"swap_d2h={swap_d2h:.3f} bad={bad}",
            flush=True,
        )
        worst = min(worst, h2d, d2h)
        torch.ops.b70_kvpark.free_host(host)
        torch.ops.b70_kvpark.free_host(back)
        del dev
        torch.xpu.synchronize()
    print(f"bench_copy: worst_one_buffer={worst:.3f} COPY_BENCH", flush=True)
    if so.endswith("libb70_kvpark_ordered.so") and ordered != 1:
        print("bench_copy: ordered library did not export ordered()", flush=True)
        return 1
    return 0 if worst >= 7.5 and ordinal >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
