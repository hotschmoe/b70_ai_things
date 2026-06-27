#!/usr/bin/env python3
# repack_w4a16_to_awq.py -- numerically-EXACT transcode of a compressed-tensors symmetric int4
# (pack-quantized, group_size=128) checkpoint into AutoAWQ GEMM format that sglang's XPU
# `awq_dequantize` path accepts (--quantization awq --dtype float16). CPU-only, streaming per shard.
#
# WHY exact: CT stores q_signed in [-8,7]; AWQ dequant is w=(u-z)*scale. Set z=8 -> u=q_signed+8 ->
# w=q_signed*scale, bit-identical. We pack with auto_round's WQLinear_GEMM.from_linear (the exact
# layout the kernel reads) and VALIDATE every layer with dequantize_gemm (round-trip allclose).
#
# NOTE: source Qwen3.6-27B-W4A16 is TEXT-ONLY (no vision). This output is a serve-path/fp16-speed
# smoke test, NOT the vision-retaining deliverable (that's the AutoRound auto_awq production run).
#
#   usage: python3 repack_w4a16_to_awq.py [SRC] [DST]
import json, os, sys, shutil, glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from auto_round.export.export_to_awq.utils import WQLinear_GEMM, dequantize_gemm
import torch.nn as nn

SRC = sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen3.6-27B-W4A16"
DST = sys.argv[2] if len(sys.argv) > 2 else "/models/Qwen3.6-27B-W4A16-awq-repack"
BITS, G = 4, 128
NOT_CONVERT = ["linear_attn", "visual", "lm_head", "mtp", "embed_tokens"]
os.makedirs(DST, exist_ok=True)
dev = torch.device("cpu")

def repack_layer(prefix, wp, ws, wshape):
    out_f, in_f = int(wshape[0].item()), int(wshape[1].item())
    q = unpack_from_int32(wp, BITS, torch.Size([out_f, in_f]), packed_dim=1).to(torch.int8)  # [out,in] signed
    assert q.min() >= -8 and q.max() <= 7, f"{prefix}: q out of [-8,7] -> wrong CT convention ({q.min()},{q.max()})"
    ws_f = ws.to(torch.float32)                                   # [out, in//G]
    W = (q.to(torch.float32) * ws_f.repeat_interleave(G, dim=1))  # [out, in]
    lin = nn.Linear(in_f, out_f, bias=False); lin.weight.data = W.to(torch.float16)
    scales_awq = ws_f.t().contiguous()                           # [in//G, out]
    zeros_awq = torch.full((in_f // G, out_f), 8, dtype=torch.int32)
    awq = WQLinear_GEMM.from_linear(lin, BITS, G, scales=scales_awq, zeros=zeros_awq, device=dev)
    # round-trip validation against the kernel's reference dequant
    W_rt = dequantize_gemm(awq.qweight, awq.qzeros, awq.scales, BITS, G)  # [in, out]
    err = (W_rt.float() - W.t().float()).abs().max().item()
    assert err < 1e-2, f"{prefix}: round-trip mismatch max|err|={err}"
    return {f"{prefix}.qweight": awq.qweight.contiguous(),
            f"{prefix}.qzeros":  awq.qzeros.contiguous(),
            f"{prefix}.scales":  awq.scales.to(torch.float16).contiguous()}, err

shards = sorted(glob.glob(os.path.join(SRC, "*.safetensors")))
print(f"[repack] {SRC} -> {DST}  ({len(shards)} shards)")
n_q = 0; max_err = 0.0
for si, sp in enumerate(shards):
    name = os.path.basename(sp); out = {}
    with safe_open(sp, framework="pt") as f:
        keys = list(f.keys())
        packed = {k[:-len(".weight_packed")] for k in keys if k.endswith(".weight_packed")}
        for k in keys:
            if any(k.endswith(s) for s in (".weight_packed", ".weight_scale", ".weight_shape")):
                continue  # consumed via the layer transcode
            out[k] = f.get_tensor(k)
        for p in sorted(packed):
            wp = f.get_tensor(f"{p}.weight_packed"); ws = f.get_tensor(f"{p}.weight_scale")
            wshape = f.get_tensor(f"{p}.weight_shape")
            t, err = repack_layer(p, wp, ws, wshape)
            out.update(t); n_q += 1; max_err = max(max_err, err)
    save_file(out, os.path.join(DST, name), metadata={"format": "pt"})
    print(f"  [{si+1}/{len(shards)}] {name}: {len(packed)} quant layers (cum {n_q}, max_err {max_err:.2e})")

# copy non-weight files
for f in os.listdir(SRC):
    if f.endswith(".safetensors"): continue
    s = os.path.join(SRC, f)
    if os.path.isfile(s): shutil.copy2(s, os.path.join(DST, f))

# rewrite config.json quantization_config -> awq
cfg = json.load(open(os.path.join(SRC, "config.json")))
cfg["quantization_config"] = {"quant_method": "awq", "bits": BITS, "group_size": G,
    "version": "gemm", "zero_point": False, "modules_to_not_convert": NOT_CONVERT}
cfg["torch_dtype"] = "float16"
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
print(f"[repack] DONE: {n_q} layers transcoded, max round-trip err {max_err:.2e}. config.json -> awq/fp16.")
