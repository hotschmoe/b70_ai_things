# Validate + microbench torch.ops._xpu_C.nvfp4_gemm_w4a16 against a reference
# E2M1 dequant, using a REAL NVFP4 W4A16 layer from the 27B checkpoint.
# Runs inside the vllm-xpu image on ONE card. ASCII only.
import os, sys, time, json, struct
import torch
import torch.nn.functional as F

SO = os.environ.get("NVFP4_SO", "/opt/nvfp4so/_xpu_C.abi3.so")
torch.ops.load_library(SO)
print("[harness] loaded", SO, flush=True)

CK = "/models/qwen3.6-27b/nvfp4-modelopt"

def load_tensors(names):
    # read specific tensors from the sharded safetensors without safetensors lib
    idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
    out = {}
    from collections import defaultdict
    byfile = defaultdict(list)
    for n in names:
        byfile[idx[n]].append(n)
    for f, ns in byfile.items():
        path = f"{CK}/{f}"
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
            base = 8 + hlen
            for n in ns:
                meta = hdr[n]
                dt = meta["dtype"]; shp = meta["shape"]; s, e = meta["data_offsets"]
                fh.seek(base + s); raw = fh.read(e - s)
                tmap = {"F32": torch.float32, "BF16": torch.bfloat16,
                        "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "I8": torch.int8}
                t = torch.frombuffer(bytearray(raw), dtype=tmap[dt]).reshape(shp)
                out[n] = t
    return out

_E2M1 = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6., -0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],
                     dtype=torch.float32)

def ref_dequant_bf16(packed_u8, wscale_f8, gscale, gs=16):
    # packed [N, K/2] uint8 low-nibble-first -> bf16 [N, K]
    N, Kh = packed_u8.shape; K = Kh*2
    lo = (packed_u8 & 0x0F).to(torch.long); hi = (packed_u8 >> 4).to(torch.long)
    nib = torch.stack([lo, hi], dim=-1).reshape(N, K)
    w = _E2M1[nib]                                   # [N,K] f32
    s = (wscale_f8.to(torch.float32) * float(gscale))  # [N, K/gs]
    s = s.repeat_interleave(gs, dim=1)               # [N,K]
    return (w * s).to(torch.bfloat16)

def main():
    dev = "xpu"
    layer = "model.language_model.layers.0.mlp.gate_proj"
    t = load_tensors([f"{layer}.weight", f"{layer}.weight_scale", f"{layer}.weight_scale_2"])
    packed = t[f"{layer}.weight"]              # [N, K/2] u8
    wscale = t[f"{layer}.weight_scale"]        # [N, K/16] f8e4m3
    gscale = t[f"{layer}.weight_scale_2"].to(torch.float32).flatten()[0].item()
    N, Kh = packed.shape; K = Kh*2; gs = K // wscale.shape[1]
    print(f"[harness] {layer}: N={N} K={K} gs={gs} global={gscale:.6g}", flush=True)

    w_ref = ref_dequant_bf16(packed, wscale, gscale, gs).to(dev)   # [N,K] bf16

    # op inputs: weight NT [K/2, N] as a TRANSPOSED VIEW (strides [1, K/2] -> K
    # dim contiguous = NT). Do NOT .contiguous() (that would make strides [N,1]).
    w_nt = packed.to(dev).t()                        # [K/2, N], strides [1, K/2], NT
    scale_nt = (wscale.to(torch.float32) * gscale).to(torch.bfloat16).t().contiguous().to(dev)  # [K/16, N]

    torch.manual_seed(0)
    for M in [1, 8, 64]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        ref = F.linear(x, w_ref)                    # [M,N]
        try:
            y = torch.ops._xpu_C.nvfp4_gemm_w4a16(x, w_nt, None, scale_nt, gs)
        except Exception as e:
            print(f"[M={M}] OP CALL FAILED: {e!r}", flush=True); return
        y = y.to(torch.float32); ref32 = ref.to(torch.float32)
        rel = (y-ref32).norm() / ref32.norm().clamp_min(1e-9)
        print(f"[M={M}] rel-err vs ref = {rel.item():.4e}  y[0,:4]={y[0,:4].tolist()} ref={ref32[0,:4].tolist()}", flush=True)

    # speed: decode M=1,8 vs bf16 F.linear
    def bench(fn, iters=50):
        torch.xpu.synchronize(); t0=time.time()
        for _ in range(iters): fn()
        torch.xpu.synchronize(); return (time.time()-t0)/iters*1e3
    wbytes = packed.numel()  # 4-bit weight bytes
    for M in [1, 8]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        t_op = bench(lambda: torch.ops._xpu_C.nvfp4_gemm_w4a16(x, w_nt, None, scale_nt, gs))
        t_bf = bench(lambda: F.linear(x, w_ref))
        gbps = wbytes / (t_op/1e3) / 1e9
        print(f"[M={M}] nvfp4_op {t_op:.3f}ms ({gbps:.0f} GB/s 4bit-wt)  bf16 {t_bf:.3f}ms  speedup {t_bf/t_op:.2f}x", flush=True)

if __name__ == "__main__":
    main()
