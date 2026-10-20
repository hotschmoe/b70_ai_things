"""M1: one blk.0 IQ3_XXS layer, linear 128/128, moe_ep vs moe_layer.

Own-expert gate: y_ep equals y0 + y1 bitwise (same moe_layer, same add order).
Full-pack gate: report max_abs, max_rel, cosine. Start line is max_rel < 1e-2
and cosine > 0.999. T = 1,2,4,8,4096. Device time at T=1 and T=8.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GGUF_ROOT = ROOT.parent / "b70_gguf_xpu"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GGUF_ROOT))
GGUF_PY = "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py"
if GGUF_PY not in sys.path:
    sys.path.insert(0, GGUF_PY)

# types.py: IQ3_S (21) block 256 size 110, IQ2_S (22) block 256 size 82.
_SIZES = {21: (256, 110), 22: (256, 82)}
WEIGHTS = os.environ.get(
    "B70_GGUF_PATH",
    "/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
)


def row_bytes(k: int, qtype: int) -> int:
    block, size = _SIZES[int(qtype)]
    if k % block != 0:
        raise SystemExit(f"FAIL width {k} qtype {qtype}")
    return k // block * size


def load_tensor(reader, suffix: str):
    for tensor in reader.tensors:
        name = tensor.name
        if name == f"blk.0.{suffix}":
            raw = np.ascontiguousarray(tensor.data).view(np.uint8).reshape(-1).copy()
            ne = tuple(int(v) for v in tensor.shape)
            logical = tuple(reversed(ne))
            return name, int(tensor.tensor_type), logical, raw
    raise SystemExit(f"FAIL missing {suffix}")


def pack(logical, qtype, raw):
    e, n, last = logical
    rb = row_bytes(last, qtype)
    need = e * n * rb
    if raw.size != need:
        raise SystemExit(f"FAIL numel {raw.size} != {need} logical={logical} qtype={qtype}")
    return raw.reshape(e, n, rb)


def moe_call(x, gate, gt, up, ut, down, dt, ids, w, inter, hidden):
    from b70_gguf_xpu.ops import moe_layer

    return moe_layer(x, gate, gt, up, ut, down, dt, ids, w, inter, hidden)


def stats(a: torch.Tensor, b: torch.Tensor):
    d = (a - b).abs()
    denom = b.abs().clamp(min=1e-8)
    rel = float((d / denom).max())
    af = a.flatten().float()
    bf = b.flatten().float()
    cos = float(torch.nn.functional.cosine_similarity(af, bf, dim=0))
    return float(d.max()), rel, cos


def bench(fn, iters: int) -> float:
    torch.xpu.set_device(0)
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    os.environ.setdefault(
        "B70_GGUF_SO",
        str(GGUF_ROOT / "b70_gguf_xpu" / "_C.ep.so"),
    )
    import gguf

    from b70_ep_xpu import ops

    ops.load()
    torch.xpu.set_device(0)
    torch.empty(1, device="xpu:0")
    torch.empty(1, device="xpu:1")
    reader = gguf.GGUFReader(WEIGHTS, "r")
    blobs = {}
    for key, suffix in (
        ("gate", "ffn_gate_exps.weight"),
        ("up", "ffn_up_exps.weight"),
        ("down", "ffn_down_exps.weight"),
    ):
        name, qtype, logical, raw = load_tensor(reader, suffix)
        blobs[key] = (name, qtype, logical, pack(logical, qtype, raw))
        print(f"LOAD {key} {name} qtype={qtype} logical={logical} bytes={raw.size}", flush=True)
    g_name, gt, g_logical, g_pack = blobs["gate"]
    _, ut, _, u_pack = blobs["up"]
    _, dt, _, d_pack = blobs["down"]
    e, inter, hidden = g_logical
    if e != 256 or inter != 512 or hidden != 2048:
        raise SystemExit(f"FAIL unexpected gate logical {g_logical}")
    n0 = e // 2
    del g_name

    def flat(view):
        return torch.from_numpy(np.ascontiguousarray(view).reshape(-1))

    g_full = flat(g_pack).to(device="xpu:0")
    u_full = flat(u_pack).to(device="xpu:0")
    d_full = flat(d_pack).to(device="xpu:0")
    g0 = flat(g_pack[:n0]).to(device="xpu:0")
    u0 = flat(u_pack[:n0]).to(device="xpu:0")
    d0 = flat(d_pack[:n0]).to(device="xpu:0")
    g1 = flat(g_pack[n0:]).to(device="xpu:1")
    u1 = flat(u_pack[n0:]).to(device="xpu:1")
    d1 = flat(d_pack[n0:]).to(device="xpu:1")

    ops.init(8192, hidden, 16, 4)
    ops.register0(0, g0, gt, u0, ut, d0, dt, n0, inter, hidden)
    ops.register1(0, g1, u1, d1, e - n0)

    rng = np.random.default_rng(0)
    fails = 0
    for T in (1, 2, 4, 8, 4096):
        topk = 8
        x_np = rng.standard_normal((T, hidden)).astype(np.float32)
        ids_np = rng.integers(0, e, size=(T, topk), dtype=np.int32)
        w_np = rng.random((T, topk)).astype(np.float32)
        x0 = torch.from_numpy(x_np).to(device="xpu:0", dtype=torch.bfloat16)
        ids = torch.from_numpy(ids_np).to(device="xpu:0")
        w = torch.from_numpy(w_np).to(device="xpu:0")
        torch.xpu.set_device(0)
        y_full = moe_call(x0, g_full, gt, u_full, ut, d_full, dt, ids, w, inter, hidden)
        ids0 = torch.where(ids < n0, ids, torch.full_like(ids, -1))
        y0 = moe_call(x0, g0, gt, u0, ut, d0, dt, ids0, w, inter, hidden)
        torch.xpu.set_device(1)
        x1 = x0.to(device="xpu:1")
        ids1_np = np.where(ids_np >= n0, ids_np - n0, -1).astype(np.int32)
        ids1 = torch.from_numpy(ids1_np).to(device="xpu:1")
        w1 = w.to(device="xpu:1")
        y1 = moe_call(x1, g1, gt, u1, ut, d1, dt, ids1, w1, inter, hidden)
        torch.xpu.set_device(0)
        y_own = y0 + y1.to(device="xpu:0")
        y_ep = torch.ops.b70_ep.moe_ep(x0, w, ids, 0)
        torch.xpu.synchronize()
        bitwise = bool(torch.equal(y_ep, y_own))
        amax, rel, cos = stats(y_ep, y_full)
        own_max = float((y_ep - y_own).abs().max())
        ok = bitwise and rel < 1e-2 and cos > 0.999
        if not ok:
            fails += 1
        print(
            f"{'OK' if ok else 'FAIL'} T={T} bitwise={int(bitwise)} own_max={own_max:.3e} "
            f"full_max={amax:.3e} full_rel={rel:.3e} cos={cos:.6f}",
            flush=True,
        )
        if T in (1, 8):
            us_ep = bench(lambda: torch.ops.b70_ep.moe_ep(x0, w, ids, 0), 10)
            torch.xpu.set_device(0)
            us_one = bench(
                lambda: moe_call(x0, g_full, gt, u_full, ut, d_full, dt, ids, w, inter, hidden),
                10,
            )
            print(f"TIME T={T} ep_us={us_ep:.1f} one_us={us_one:.1f}", flush=True)
    if fails:
        print(f"FAIL layers={fails}", flush=True)
        return 1
    print("PASS m1", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL exception {type(exc).__name__}: {exc}", flush=True)
        raise
