"""Device-resident compact split plans for the existing Xe2 FA kernel.

No native kernel change. The compact-plan path already trusts the supplied
partition, bypassing the legacy short-context single-split heuristic.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _plan(Lens, Splits, Work, N: tl.constexpr, TILE: tl.constexpr):
    seq = tl.program_id(0)
    i = tl.arange(0, N)
    tiles = tl.cdiv(tl.load(Lens + seq), TILE)
    actual = tl.minimum(N, tl.maximum(tiles, 1))
    base = tiles // actual
    rem = tiles % actual
    start = i * base + tl.minimum(i, rem)
    count = tl.where(i < actual, base + (i < rem).to(tl.int32), 0)
    tl.store(Splits + seq, actual)
    row = (seq * N + i) * 4
    tl.store(Work + row, seq)
    tl.store(Work + row + 1, start)
    tl.store(Work + row + 2, count)
    tl.store(Work + row + 3, i)


def fill_plan(lens, splits, work, count):
    _plan[(lens.numel(),)](lens, splits, work, N=count, TILE=64,
                          num_warps=4)


def paged_attention(q, k, v, out, cu, lens, table, max_k, count,
                    scale=None, sink=None):
    """Allocate graph-owned plan/scratch and invoke unchanged native FA."""
    batch = lens.numel()
    counts = torch.empty((batch,), device=q.device, dtype=torch.int32)
    work = torch.empty((batch * count, 4), device=q.device, dtype=torch.int32)
    zeros = torch.zeros_like(cu)
    fill_plan(lens, counts, work, count)
    result, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, out, cu, zeros, lens, None, table, None,
        1, max_k, 0.0, None, None, None,
        q.shape[-1] ** -0.5 if scale is None else scale, sink,
        False, True, -1, -1, 0.0, False, None, count,
        False, counts, work,
    )
    return result
