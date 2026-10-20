"""One expert prefill layer: wide dequant, one gate+up bmm, down bmm, combine.

Scratch holds one layer of bf16 experts (gate and up packed as N=2F, plus
down). A second queue dequants down while the gate+up GEMM runs. Two full
layers of bf16 experts are 9.375 GiB and do not fit the 2x200K headroom, so
the next layer is not prefetched.

Importing this module does not touch the GPU.
"""

from __future__ import annotations

import os

LAST = {}
_STREAMS = {}


def wants_fast() -> bool:
    """True when this .so has the wide dequant and the env does not force bmm."""
    raw = os.environ.get("B70_PREFILL_DQ_LAYER", "fast").strip().lower()
    if raw in ("bmm", "off", "0", "c10"):
        return False
    from b70_gguf_xpu.prefill import load

    load()
    import torch

    return hasattr(torch.ops.b70_gguf, "prefill_dequant_wide")


def _side_stream(device):
    import torch

    key = (device.type, device.index)
    stream = _STREAMS.get(key)
    if stream is None:
        stream = torch.xpu.Stream(device=device)
        _STREAMS[key] = stream
    return stream


def _overlap_on() -> bool:
    raw = os.environ.get("B70_PREFILL_DQ_OVERLAP", "1").strip().lower()
    return raw not in ("0", "off", "no")


def moe_fast(x, ids, q_gate, q_up, q_down, router, type_gu: int, type_down: int, h: int, f: int, group=None):
    """x [T, H] -> fp32 [T, H]. ids int32 [T, topk]. router fp32 [T, topk] or empty.

    Gate and up are dequantized into one [E, 2F, H] bf16 buffer and multiplied
    as a single N=2F bmm. Down is a second bmm. Token rows are padded to mmax
    (multiple of 8) per expert. GEMM rows do not mix, so the padded tail is
    left unread and is not zeroed.
    """
    import torch

    from b70_gguf_xpu import prefill_dq
    from b70_gguf_xpu.prefill import load
    from b70_gguf_xpu.types import row_bytes

    load()
    t = int(x.shape[0])
    topk = int(ids.shape[1])
    h = int(h)
    f = int(f)
    rb = row_bytes(int(type_gu), h)
    if rb <= 0:
        raise RuntimeError(f"bad gate type {type_gu} H={h}")
    e = int(q_gate.numel() // (f * rb))
    if e <= 0 or q_up.numel() != q_gate.numel():
        raise RuntimeError("gate/up pack size")
    rb_d = row_bytes(int(type_down), f)
    if rb_d <= 0 or int(q_down.numel() // (h * rb_d)) != e:
        raise RuntimeError("down pack size")
    marks = [] if os.environ.get("B70_PREFILL_DQ_PROFILE", "0") == "1" else None

    def stamp(name: str) -> None:
        if marks is None:
            return
        ev = torch.xpu.Event(enable_timing=True)
        ev.record()
        marks.append((name, ev))

    stamp("start")
    LAST.pop("fallback", None)
    if group is None:
        group = prefill_dq._Group(prefill_dq._sanitize(ids, e), e)
    stamp("group")
    LAST["mv"] = group.mv
    LAST["mmax"] = group.mmax
    LAST["E"] = e
    LAST["T"] = t
    if group.mv == 0 or group.mmax == 0:
        return torch.zeros(t, h, dtype=torch.float32, device=x.device)
    cap = group.mmax
    LAST["cap"] = cap
    # vLLM's zero-input profile can route every token to the same experts.
    # Padding all E experts to that capacity creates multi-GiB empty GEMMs.
    max_ratio = float(os.environ.get("B70_PREFILL_PAD_RATIO", "2"))
    if (cap > 8 and e * cap > max_ratio * group.mv) or e * cap * h * 2 > 8_000_000_000:
        LAST["fallback"] = "loop_skew"
        return prefill_dq.moe_dq(
            x, ids, q_gate, q_up, q_down, router, type_gu, type_down, h, f, how="loop", group=group
        )
    xb = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
    direct = os.environ.get("B70_PREFILL_REDUCE") == "grouped"
    if not direct:
        tok = torch.arange(t, device=x.device).unsqueeze(1).expand(t, topk).reshape(-1)
        act = xb[tok]
    w_gu = prefill_dq._scratch(
        ("c13gu", f, h, _dev(x)), (e, 2 * f, h), torch.bfloat16, x.device
    )
    w_dn = prefill_dq._scratch(("c13dn", h, f, _dev(x)), (e, h, f), torch.bfloat16, x.device)
    gu_flat = w_gu.view(e * 2 * f, h)
    dn_flat = w_dn.view(e * h, f)
    torch.ops.b70_gguf.prefill_dequant_wide(q_gate, int(type_gu), e * f, h, gu_flat, 1, f)
    stamp("dq_gate")
    torch.ops.b70_gguf.prefill_dequant_wide(q_up, int(type_gu), e * f, h, gu_flat, 2, f)
    stamp("dq_up")
    overlap = _overlap_on()
    side = None
    if overlap:
        try:
            side = _side_stream(x.device)
            side.wait_stream(torch.xpu.current_stream(x.device))
            with torch.xpu.stream(side):
                torch.ops.b70_gguf.prefill_dequant_wide(q_down, int(type_down), e * h, f, dn_flat, 0, 0)
            stamp("dq_down_side")
        except (AttributeError, RuntimeError) as ex:
            LAST["overlap_error"] = str(ex)
            side = None
            torch.ops.b70_gguf.prefill_dequant_wide(q_down, int(type_down), e * h, f, dn_flat, 0, 0)
    else:
        torch.ops.b70_gguf.prefill_dequant_wide(q_down, int(type_down), e * h, f, dn_flat, 0, 0)
        stamp("dq_down")
    xpad = prefill_dq._scratch(("c13x", h, _dev(x)), (e, cap, h), torch.bfloat16, x.device)
    if direct:
        from .prefill_reduce import pad_tokens

        pad_tokens(xb, xpad, group.order, group.starts, topk)
    else:
        torch.ops.b70_gguf.prefill_pad_grouped(act, xpad, group.order, group.starts, int(group.mv))
    stamp("pad")
    y = torch.bmm(xpad, w_gu.transpose(1, 2))
    stamp("bmm_gu")
    g, u = y.split(f, dim=-1)
    hid = (torch.nn.functional.silu(g) * u).contiguous()
    stamp("silu")
    if side is not None:
        torch.xpu.current_stream(x.device).wait_stream(side)
        stamp("wait_down")
    yd = torch.bmm(hid, w_dn.transpose(1, 2))
    stamp("bmm_dn")
    if direct:
        from .prefill_reduce import reduce_grouped

        out = reduce_grouped(yd, group.order, group.starts, router, t, topk, h)
    else:
        slots = act if group.mv == act.shape[0] else torch.zeros_like(act)
        torch.ops.b70_gguf.prefill_unpad_scatter(yd, slots, group.order, group.starts, int(group.mv))
        stamp("unpad")
        out = prefill_dq._reduce(slots, router, t, topk, h)
    stamp("reduce")
    if marks is not None:
        torch.xpu.synchronize(x.device)
        parts = []
        stage_ms = {}
        for i in range(1, len(marks)):
            ms = marks[i - 1][1].elapsed_time(marks[i][1])
            stage_ms[marks[i][0]] = ms
            parts.append(f"{marks[i][0]}={ms:.2f}ms")
        LAST["profile_ms"] = stage_ms
        LAST["profile"] = " ".join(parts)
        print(f"PROFILE T={t} {LAST['profile']}", flush=True)
    LAST["cap"] = cap
    LAST["overlap"] = int(side is not None)
    return out


def _dev(t):
    d = t.device
    return (d.type, d.index)
