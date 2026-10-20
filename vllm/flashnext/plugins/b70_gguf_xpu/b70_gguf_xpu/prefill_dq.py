"""Dequant-then-oneDNN prefill. Importing this module does not touch the GPU.

Weights are dequantized into a reusable bf16 scratch, then torch.mm / torch.bmm
(oneDNN) does the GEMM. The scratch is overwritten on the same XPU stream only
after the GEMM that reads it has been queued.

B70_PREFILL_DQ_GROUPED=bmm|loop selects the expert GEMM. bmm pads each expert
up to a multiple of 8 rows. loop queues one torch.mm per occupied expert.
"""

from __future__ import annotations

import os

from b70_gguf_xpu.prefill import load
from b70_gguf_xpu.types import row_bytes

# Filled by each grouped call so a bench can print mv/mmax outside the timer.
LAST = {}

# Card 0, 2026-10-20, log c10b-dq-bench.log. Env overrides:
#   B70_PREFILL_DQ_M=off never, 0 always, N uses dq when M >= N
#   B70_PREFILL_DQ_T same for the MoE layer token count.
#   B70_PREFILL_DQ_GROUP_M same for one grouped GEMM (not the full layer).
# Default policy, wall time, dq versus the fused kernel:
#   K=2560 (Q4_K and IQ4_XS, N=12288): dq wins at every M from 512 to 12288.
#   K=640 (Q2_0, N=2560): fused wins at M=512 (238 vs 356 us); dq wins at 4096 and 12288.
#   Full expert layer: fused wins at T=512 and T=4096; bmm wins at T=12288.
# A single grouped GEMM was not faster than fused at those expert sizes, so
# that wrapper stays on the fused kernel unless the group env is set.
DQ_M_MIN = 512
DQ_M_SHORT = 4096
DQ_K_SHORT = 640
DQ_T_MIN = 12288
DQ_GROUP_M_MIN = 10**9

_SCRATCH: dict = {}


def _devkey(t):
    d = t.device
    return (d.type, d.index)


def _scratch(key, shape, dtype, device):
    import torch

    buf = _SCRATCH.get(key)
    if (
        buf is None
        or buf.dtype != dtype
        or buf.device != device
        or buf.shape != torch.Size(shape)
    ):
        buf = torch.empty(shape, dtype=dtype, device=device)
        _SCRATCH[key] = buf
    return buf


def _i32(name, n, device):
    import torch

    return _scratch(("i32", name, n, _devkey_dev(device)), (n,), torch.int32, device)


def _devkey_dev(device):
    return (device.type, device.index)


def _env_threshold(env: str):
    """None = caller default. 'off' disables. 'on' or 0 enables for every size."""
    raw = os.environ.get(env, "").strip().lower()
    if raw == "":
        return None
    if raw in ("off", "none", "-1"):
        return "off"
    if raw in ("on", "0"):
        return 0
    return int(raw)


def dense_wants_dq(m: int, k: int) -> bool:
    """True when dequant + torch.mm beat the fused dense kernel on this shape."""
    forced = _env_threshold("B70_PREFILL_DQ_M")
    if forced == "off":
        return False
    if forced is not None:
        return m >= forced
    if m >= DQ_M_SHORT:
        return True
    return m >= DQ_M_MIN and k > DQ_K_SHORT


def moe_wants_dq(t: int) -> bool:
    """True when dequant + batched GEMM beat the fused expert layer.

    C11 remeasured the fused layer and two wider schedules. None of them
    beat dequant + bmm at T>=12288, and bmm is still slower below that.
    """
    forced = _env_threshold("B70_PREFILL_DQ_T")
    if forced == "off":
        return False
    if forced is not None:
        return t >= forced
    return t >= DQ_T_MIN


def dequant(q, ggml_type: int, rows: int, cols: int, dtype=None):
    """Dequant q into reusable scratch [rows, cols]. dtype bf16 (default) or fp16."""
    import torch

    load()
    if dtype is None:
        dtype = torch.bfloat16
    key = ("dq", int(ggml_type), rows, cols, dtype, _devkey(q))
    if os.environ.get("B70_PREFILL_BUCKET") == "1":
        # Bucket counts vary by layer. Each projection consumes this view
        # before the next dequant on its queue; do not retain every count.
        key = ("dq_bucket", dtype, _devkey(q), torch.xpu.current_stream(q.device).sycl_queue)
    buf = _scratch(key, (rows, cols), dtype, q.device)
    torch.ops.b70_gguf.prefill_dequant(q, int(ggml_type), int(rows), int(cols), buf)
    return buf


def _as_bf16(x):
    import torch

    return x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)


def dense_dq(x, q, ggml_type: int, n: int, k: int, output_dtype=None):
    """BF16 dequant + mm; default FP32, optional native serving dtype."""
    import torch

    xb = _as_bf16(x)
    w = dequant(q, ggml_type, n, k)
    result = torch.mm(xb, w.t())
    return result.float() if output_dtype is None else result.to(output_dtype)


def _sanitize(ids, e: int):
    import torch

    flat = ids.reshape(-1).contiguous()
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    ok = (flat >= 0) & (flat < e)
    # Keep invalid slots on device; testing ok.all() stalls the host queue.
    return torch.where(ok, flat, torch.full_like(flat, -1))


class _Group:
    def __init__(self, flat, e: int):
        import torch

        m = int(flat.numel())
        device = flat.device
        order = _i32("order", m, device)
        starts = _i32("starts", e + 1, device)
        counts = _i32("counts", e, device)
        cursor = _i32("cursor", e, device)
        torch.ops.b70_gguf.prefill_group_ids(flat, int(e), order, starts, counts, cursor)
        st = starts.cpu()
        self.order = order
        self.starts = starts
        self.M = m
        self.E = e
        self.starts_cpu = st.tolist()
        self.mv = self.starts_cpu[-1]
        diffs = [self.starts_cpu[i + 1] - self.starts_cpu[i] for i in range(e)]
        raw = max(diffs) if diffs else 0
        self.mmax = (max(raw, 1) + 7) & ~7 if self.mv else 0
        LAST["mv"] = self.mv
        LAST["mmax"] = self.mmax
        LAST["M"] = m
        LAST["E"] = e

    def subset(self, experts):
        """Reuse the outer expert ordering without another histogram or CPU wait."""
        import torch

        child = _Group.__new__(_Group)
        child.M, child.E = self.M, len(experts)
        starts = [0]
        slices = []
        for expert in experts:
            lo, hi = self.starts_cpu[expert:expert + 2]
            slices.append(self.order[lo:hi])
            starts.append(starts[-1] + hi - lo)
        child.order = torch.cat(slices)
        child.starts = torch.tensor(starts, device=self.order.device, dtype=torch.int32)
        child.starts_cpu = starts
        child.mv = starts[-1]
        raw = max((b - a for a, b in zip(starts, starts[1:])), default=0)
        child.mmax = (raw + 7) & ~7
        return child


def _project_bmm(act, group: _Group, w, n: int, k: int):
    """act [M, K] bf16, w [E, N, K] bf16 (must stay live until this returns). y [M, N] bf16."""
    import torch

    m = group.M
    y = torch.empty((m, n), dtype=act.dtype, device=act.device) if group.mv == m else torch.zeros(
        (m, n), dtype=act.dtype, device=act.device
    )
    if group.mv == 0:
        return y
    # A pathological expert cluster (every token on one expert) would pad to
    # E * T * topk rows. Refuse that instead of allocating tens of GB.
    if group.E * group.mmax * k * 2 > 8_000_000_000:
        raise RuntimeError(
            f"bmm pad {group.E}*{group.mmax}*{k} exceeds 8 GB; use how=loop"
        )
    xpad = _scratch(("xpad", group.E, k, act.dtype, _devkey(act)), (group.E, group.mmax, k), act.dtype, act.device)
    xpad.zero_()
    torch.ops.b70_gguf.prefill_pad_grouped(act, xpad, group.order, group.starts, int(group.mv))
    # w is [E, N, K]; bmm wants [E, K, N]. The transpose view matches the
    # torch.mm(x, w.t()) layout that C8 measured at 100-141 TFLOPS.
    ypad = torch.bmm(xpad, w.transpose(1, 2))
    torch.ops.b70_gguf.prefill_unpad_scatter(ypad, y, group.order, group.starts, int(group.mv))
    return y


def _project_loop(act, group: _Group, w, n: int, k: int):
    import torch

    m = group.M
    y = torch.empty((m, n), dtype=act.dtype, device=act.device) if group.mv == m else torch.zeros(
        (m, n), dtype=act.dtype, device=act.device
    )
    if group.mv == 0:
        return y
    packed = _scratch(("packed", k, act.dtype, _devkey(act)), (group.mv, k), act.dtype, act.device)
    outp = _scratch(("packed_y", n, act.dtype, _devkey(act)), (group.mv, n), act.dtype, act.device)
    torch.ops.b70_gguf.prefill_row_gather(act, packed, group.order, int(group.mv))
    st = group.starts_cpu
    for e in range(group.E):
        a, b = st[e], st[e + 1]
        if b > a:
            torch.mm(packed[a:b], w[e].t(), out=outp[a:b])
    torch.ops.b70_gguf.prefill_row_scatter(outp, y, group.order, int(group.mv))
    return y


def _project(act, group: _Group, w, n: int, k: int, how: str):
    if how == "loop":
        return _project_loop(act, group, w, n, k)
    return _project_bmm(act, group, w, n, k)


def _reduce(slots, router, t: int, topk: int, h: int):
    import torch

    if os.environ.get("B70_PREFILL_REDUCE") in ("triton", "grouped"):
        from .prefill_reduce import reduce_slots

        return reduce_slots(slots, router, t, topk, h)
    acc = slots.view(t, topk, h).float()
    if router is not None and router.numel() == t * topk:
        acc = acc * router.reshape(t, topk, 1).float()
    return acc.sum(1)


def grouped_dq(x, q, expert_ids, ggml_type: int, n: int, k: int, how: str | None = None):
    """x [M, K], expert_ids [M] -> fp32 [M, N]."""
    import torch

    how = how or os.environ.get("B70_PREFILL_DQ_GROUPED", "bmm")
    e = int(q.numel() // (n * row_bytes(ggml_type, k)))
    flat = _sanitize(expert_ids, e)
    group = _Group(flat, e)
    xb = _as_bf16(x)
    w = dequant(q, ggml_type, e * n, k).view(e, n, k)
    return _project(xb, group, w, n, k, how).float()


def moe_dq(x, ids, q_gate, q_up, q_down, router, type_gu: int, type_down: int, h: int, f: int, how: str | None = None, type_up: int | None = None, group=None):
    """One prefill MoE layer. x [T, H] -> fp32 [T, H].

    how=bmm|loop dequants every expert. how=hybrid keeps the fused grouped
    kernel for gate and up and uses dequant + oneDNN only for the K=F down GEMM.
    """
    import torch

    how = how or os.environ.get("B70_PREFILL_DQ_GROUPED", "bmm")
    type_up = int(type_gu) if type_up is None else int(type_up)
    t = int(x.shape[0])
    topk = int(ids.shape[1])
    rb = row_bytes(type_gu, h)
    e = int(q_gate.numel() // (f * rb))
    if how == "hybrid" and type_up == int(type_gu) and group is None:
        return _moe_hybrid(x, ids, q_gate, q_up, q_down, router, type_gu, type_down, h, f, e, t, topk)
    if how == "hybrid":
        how = "bmm"
    if group is None:
        group = _Group(_sanitize(ids, e), e)
    if how == "bmm" and ((group.mmax > 8 and e * group.mmax > float(os.environ.get("B70_PREFILL_PAD_RATIO", "2")) * group.mv)
                         or e * group.mmax * h * 2 > 8_000_000_000):
        how = "loop"
    xb = _as_bf16(x)
    tok = torch.arange(t, device=x.device).unsqueeze(1).expand(t, topk).reshape(-1)
    act = xb[tok]
    wg = dequant(q_gate, type_gu, e * f, h).view(e, f, h)
    g = _project(act, group, wg, f, h, how)
    wu = dequant(q_up, type_up, e * f, h).view(e, f, h)
    u = _project(act, group, wu, f, h, how)
    hidden = (torch.nn.functional.silu(g) * u).to(act.dtype)
    wd = dequant(q_down, type_down, e * h, f).view(e, h, f)
    down = _project(hidden, group, wd, h, f, how)
    return _reduce(down, router, t, topk, h)


def _moe_hybrid(x, ids, q_gate, q_up, q_down, router, type_gu, type_down, h, f, e, t, topk):
    """Fused GU (prefill_grouped) plus dequant+oneDNN down. Fixes the short-K down path."""
    import torch

    load()
    xf = x if x.dtype == torch.float16 else x.to(torch.float16)
    tok = torch.arange(t, device=x.device).unsqueeze(1).expand(t, topk).reshape(-1)
    act = xf[tok]
    flat = ids.reshape(-1).contiguous()
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    g = torch.ops.b70_gguf.prefill_grouped(act, q_gate, flat, int(type_gu), int(f), int(h), 0)
    u = torch.ops.b70_gguf.prefill_grouped(act, q_up, flat, int(type_gu), int(f), int(h), 0)
    hidden = torch.nn.functional.silu(g) * u
    down = grouped_dq(hidden, q_down, flat, int(type_down), int(h), int(f), how="bmm")
    return _reduce(down, router, t, topk, h)
