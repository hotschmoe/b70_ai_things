"""F1: sidecar prefill for T > 8 (eager, never captured).

Per MoE layer:
  1. card 0: ep_prefill_dispatch pushes x/w/ids (ids remapped to card 1's
     local slots) into card 1's inbox and raises a device flag; card 1's
     stream waits on the flag and copies the inbox into fresh tensors.
  2. card 0: the shared expert (when the method owns it) and its own 138
     experts through the X3/C13 prefill dispatcher (serve_prefill.moe_prefill).
  3. card 1: its 374 experts. Default is the scratch-free fused DPAS
     grouped kernel (card 1 has about 1 GiB free beside 29 GiB of experts);
     B70_EP_PREFILL_C1=fast:N uses the dequant+bmm path with N-expert tiles.
  4. ep_prefill_return pushes card 1's partial sum into card 0's psum
     region and card 0 adds it (fp32, or bf16 with B70_EP_PREFILL_RET=bf16).

Env:
  B70_EP_PREFILL=1            enable (default off)
  B70_EP_PREFILL_MAX_T        exchange size in tokens (default 12288)
  B70_EP_PREFILL_C0           fast (default) | fused      card-0 expert path
  B70_EP_PREFILL_C1           fused (default) | fast:N    card-1 expert path
  B70_EP_PREFILL_TILE0        card-0 expert tile for the fast path (default: B70_PREFILL_EXPERT_TILE)
  B70_EP_PREFILL_RET          fp32 (default) | bf16      return payload
  B70_EP_PREFILL_SPIN         device flag spin bound (default 1e9)
  B70_EP_PREFILL_PROFILE=1    per-layer event timing, printed per forward
"""

from __future__ import annotations

import os

_ON = os.environ.get("B70_EP_PREFILL") == "1"
_INIT = False
_PROF = os.environ.get("B70_EP_PREFILL_PROFILE") == "1"
_RET_BF16 = os.environ.get("B70_EP_PREFILL_RET", "fp32").lower() == "bf16"
# B70_EP_PREFILL_CTL=<file>: re-read at layer 0 of every forward when its
# mtime changes; tokens "profile=0|1 ret=fp32|bf16 c1_chunk=N" switch the
# arm at runtime so one loaded server measures several arms.
_CTL = os.environ.get("B70_EP_PREFILL_CTL", "")
_CTL_MTIME = -1.0
_C0 = os.environ.get("B70_EP_PREFILL_C0", "fast").lower()
_C1 = os.environ.get("B70_EP_PREFILL_C1", "fused").lower()
_MAX_T = int(os.environ.get("B70_EP_PREFILL_MAX_T", "12288"))
# Card 1 has about 1 GiB free. The fused grouped op allocates ~32 KB of
# scratch per expanded row (xpack, gu fp32, h, ydown fp32, slot fp32), so
# card 1 runs its experts in token sub-chunks of this size (M = chunk*topk).
_C1_CHUNK = int(os.environ.get("B70_EP_PREFILL_C1_CHUNK", "2048"))
_NLAYERS = 64
_LOGGED: set = set()
# Profile accumulators for the forward in flight: name -> [ms sum, calls].
PROF: dict = {}
_EVENTS: list = []
_LAST_LAYER = -1


def enabled() -> bool:
    return _ON


def init_if_on(hidden: int, max_topk: int, n_layer_hint: int) -> None:
    """Called once from process_weights_after_loading after the decode ep_init."""
    global _INIT, _LAST_LAYER
    if not _ON or _INIT:
        return
    import torch

    torch.ops.b70_ep.ep_prefill_init(int(_MAX_T), int(hidden), int(max_topk), int(_NLAYERS))
    _INIT = True
    _LAST_LAYER = int(n_layer_hint) - 1
    free1, total1 = torch.xpu.mem_get_info(1)
    free0, total0 = torch.xpu.mem_get_info(0)
    print(f"b70_ep: prefill mem_get_info card0 free={free0/2**30:.2f}/{total0/2**30:.2f} GiB "
          f"card1 free={free1/2**30:.2f}/{total1/2**30:.2f} GiB c1_chunk={_C1_CHUNK}", flush=True)
    print(
        f"b70_ep: prefill sidecar on max_t={_MAX_T} c0={_C0} c1={_C1} ret={'bf16' if _RET_BF16 else 'fp32'} "
        f"profile={int(_PROF)}",
        flush=True,
    )


def _note(key: str, msg: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    print(f"b70_ep: prefill {msg}", flush=True)


def _ev():
    import torch

    e = torch.xpu.Event(enable_timing=True)
    e.record()
    return e


def _mark(name: str, dev: int, pair: list) -> None:
    """pair = [start_event] -> append end event and the (name, dev) key."""
    e = _ev()
    _EVENTS.append((name, dev, pair[0], e))


def _env_override(key: str, value):
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = str(value)
    return old


def _restore(key: str, old) -> None:
    if old is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = old


def _experts(x, ids, w, gate, up, down, gt: int, ut: int, dt: int, h: int, f: int, mode: str, card: int):
    """fp32 [T, H] partial sum over the local slots in ids (-1 contributes 0)."""
    import torch

    from b70_gguf_xpu import serve_prefill

    if mode.startswith("fast"):
        tile = None
        if ":" in mode:
            tile = int(mode.split(":", 1)[1])
        elif card == 0:
            tile = os.environ.get("B70_EP_PREFILL_TILE0")
        old = _env_override("B70_PREFILL_EXPERT_TILE", tile) if tile is not None else None
        try:
            y = serve_prefill.moe_prefill(x, ids, gate, up, down, w, gt, ut, dt, h, f)
        finally:
            if tile is not None:
                _restore("B70_PREFILL_EXPERT_TILE", old)
        _note(f"c{card}-fast", f"card {card} experts: fast dispatcher (dq/bmm or fused by T) tile={tile} {serve_prefill.LAST}")
        return y
    # fused: force the single-launch DPAS grouped kernel (no bf16 scratch).
    old_t = _env_override("B70_PREFILL_DQ_T", "off")
    old_b = _env_override("B70_PREFILL_BUCKET", None)
    try:
        y = serve_prefill._moe_once(x, ids, gate, up, down, w, gt, ut, dt, h, f)
    finally:
        _restore("B70_PREFILL_DQ_T", old_t)
        _restore("B70_PREFILL_BUCKET", old_b)
    _note(f"c{card}-fused", f"card {card} experts: fused DPAS grouped kernel types={gt},{ut},{dt}")
    if y.dtype != torch.float32:
        y = y.float()
    return y


def _poll_ctl() -> None:
    global _CTL_MTIME, _PROF, _RET_BF16, _C1_CHUNK
    if not _CTL:
        return
    try:
        m = os.stat(_CTL).st_mtime
    except OSError:
        return
    if m == _CTL_MTIME:
        return
    _CTL_MTIME = m
    try:
        text = open(_CTL, encoding="utf-8").read()
    except OSError:
        return
    for tok in text.split():
        k, _, v = tok.partition("=")
        if k == "profile":
            _PROF = v == "1"
        elif k == "ret":
            _RET_BF16 = v.lower() == "bf16"
        elif k == "c1_chunk":
            _C1_CHUNK = int(v)
    _EVENTS.clear()
    print(f"b70_ep: prefill ctl profile={int(_PROF)} ret={'bf16' if _RET_BF16 else 'fp32'} c1_chunk={_C1_CHUNK}", flush=True)


def apply(method, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
    import torch

    if int(method._layer_idx) == 0:
        _poll_ctl()
    if not _INIT:
        raise RuntimeError("b70_ep prefill: init_if_on did not run (B70_EP_PREFILL=1 set after load?)")
    li = int(method._layer_idx)
    xb = x if x.dtype == torch.bfloat16 else x.to(dtype=torch.bfloat16)
    if not xb.is_contiguous():
        xb = xb.contiguous()
    w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.to(dtype=torch.float32)
    if not w.is_contiguous():
        w = w.contiguous()
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(dtype=torch.int32)
    if not ids.is_contiguous():
        ids = ids.contiguous()
    t = int(xb.shape[0])
    if t > _MAX_T:
        raise RuntimeError(f"b70_ep prefill: T={t} exceeds B70_EP_PREFILL_MAX_T={_MAX_T}")
    map0, map1, n0, e1 = method._pf_maps
    gt = int(layer.b70_gate_type_id)
    ut = int(layer.b70_up_type_id)
    dt = int(layer.b70_down_type_id)
    h = int(layer.b70_hidden_size)
    f = int(layer.b70_intermediate)

    if _PROF:
        s_disp = [_ev()]
    ids0, x1, w1, ids1 = torch.ops.b70_ep.ep_prefill_dispatch(xb, w, ids, map0, map1, int(n0), int(e1), li)
    if _PROF:
        _mark("c0_dispatch_push", 0, s_disp)

    # Shared expert on card 0 while card 1 receives and computes.
    if shared_experts is not None and method.mk_can_overlap_shared_experts:
        if _PROF:
            s_sh = [_ev()]
        src = shared_experts_input if shared_experts_input is not None else x
        shared_experts._output[shared_experts._output_idx] = shared_experts._layer(src)
        if _PROF:
            _mark("c0_shared", 0, s_sh)

    if _PROF:
        s_c0 = [_ev()]
    y0 = _experts(xb, ids0, w, layer.b70_w_gate, layer.b70_w_up, layer.b70_w_down, gt, ut, dt, h, f, _C0, 0)
    if not y0.is_contiguous():
        y0 = y0.contiguous()
    if _PROF:
        _mark("c0_experts", 0, s_c0)

    with torch.xpu.device(1):
        if _PROF:
            s_c1 = [_ev()]
        step = _C1_CHUNK if _C1_CHUNK > 0 else t
        if step >= t:
            y1 = _experts(x1, ids1, w1, method._c1_gate, method._c1_up, method._c1_down, gt, ut, dt, h, f, _C1, 1)
        else:
            y1 = torch.empty((t, h), dtype=torch.float32, device=x1.device)
            for a in range(0, t, step):
                b = min(a + step, t)
                part = _experts(x1[a:b], ids1[a:b], w1[a:b], method._c1_gate, method._c1_up, method._c1_down,
                                gt, ut, dt, h, f, _C1, 1)
                y1[a:b].copy_(part)
                del part
        if _RET_BF16:
            y1 = y1.to(dtype=torch.bfloat16)
        if not y1.is_contiguous():
            y1 = y1.contiguous()
        if _PROF:
            _mark("c1_experts", 1, s_c1)
        if li == 0 and "mem1" not in _LOGGED:
            _LOGGED.add("mem1")
            free1, total1 = torch.xpu.mem_get_info(1)
            print(f"b70_ep: prefill after layer 0 card1 free={free1/2**30:.2f} GiB "
                  f"reserved={torch.xpu.memory_reserved(1)/2**30:.2f} GiB T={t}", flush=True)

    if _PROF:
        s_ret = [_ev()]
    y = torch.ops.b70_ep.ep_prefill_return(y1, y0, li)
    if _PROF:
        _mark("c0_wait_add", 0, s_ret)
        if li == _LAST_LAYER:
            _flush_profile(t)
    return y.to(dtype=x.dtype)


def _flush_profile(t: int) -> None:
    import torch

    torch.xpu.synchronize(0)
    torch.xpu.synchronize(1)
    sums: dict = {}
    for name, dev, a, b in _EVENTS:
        ms = a.elapsed_time(b)
        acc = sums.setdefault(name, [0.0, 0])
        acc[0] += ms
        acc[1] += 1
    _EVENTS.clear()
    parts = " ".join(f"{k}={v[0]:.1f}ms/{v[1]}" for k, v in sorted(sums.items()))
    print(f"EP-PF PROFILE T={t} {parts}", flush=True)
    PROF.clear()
    PROF.update({k: v[0] for k, v in sums.items()})
    PROF["T"] = t
