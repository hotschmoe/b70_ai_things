#!/usr/bin/env python3
"""Offline pre-pack of a compressed-tensors W4A8 (int-quantized, I8-stored 4-bit) model into the EXACT
int32 packed layout vLLM's XPUW4A8IntLinearKernel produces on load -- so vLLM loads the small packed
weights directly (no 28 GiB unpacked-I8 GPU transient that hangs/OOMs the B70 on the 27B).

Packing must byte-match the kernel's `_pack_int4_weight` (xpu.py):
    w_u4 = w.to(int32) + 8            # [-8,7] -> [0,15]
    w_u4 = w_u4.reshape(N, K//8, 8)
    shifts = arange(0,32,4)
    packed = ((w_u4 & 0xF) << shifts).sum(dim=2).to(int32)   # [N, K//8] int32

A weight is "quantized" iff a sibling `<prefix>.weight_scale` exists; those `<prefix>.weight` (I8) get
packed. Everything else (bf16 ignored layers, scales, norms, embeddings) is copied verbatim.
Adds quantization_config["is_prepacked_w4a8"]=true so the patched loader/kernel take the packed path.

Usage (CPU, in a torch+safetensors container):
  SRC=/models/Qwen3.6-27B-W4A8-rtn DST=/models/Qwen3.6-27B-W4A8-rtn-prepacked python offline_prepack_w4a8.py
"""
import os, json, glob, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.environ["SRC"]; DST = os.environ["DST"]
os.makedirs(DST, exist_ok=True)
SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)

def pack(w: torch.Tensor) -> torch.Tensor:
    # w: [N, K] int8 in [-8,7]  ->  [N, K//8] int32 (matches kernel _pack_int4_weight exactly)
    assert w.dtype == torch.int8 and w.shape[1] % 8 == 0, f"bad weight {w.dtype} {tuple(w.shape)}"
    w_u4 = (w.to(torch.int32) + 8).reshape(w.shape[0], w.shape[1] // 8, 8)
    return ((w_u4 & 0xF) << SHIFTS[None, None, :]).sum(dim=2).to(torch.int32)

# 1) gather all tensor names + which shards, find quantized weights (have a sibling .weight_scale)
shards = sorted(glob.glob(f"{SRC}/*.safetensors"))
names = {}  # name -> shard path
for sh in shards:
    with safe_open(sh, framework="pt") as f:
        for n in f.keys():
            names[n] = sh
scale_prefixes = {n[:-len(".weight_scale")] for n in names if n.endswith(".weight_scale")}
# Only pack STANDARD quantized Linear weights that go through the patched W4A8 create_weights path.
# Vocab layers (lm_head / embed_tokens) use VocabParallelEmbedding's own loader (expects unpacked) -> NEVER
# pack them. In the quality config they are bf16 anyway (no scale); this is a defensive guard.
SKIP_PACK = ("lm_head", "embed_tokens", "embed")
to_pack = {
    f"{p}.weight" for p in scale_prefixes
    if f"{p}.weight" in names and not any(s in p for s in SKIP_PACK)
}
print(f"[prepack] tensors={len(names)} quantized_weights_to_pack={len(to_pack)}", flush=True)

# 2) load, pack the quantized weights, copy the rest; write a single output shard
out = {}; packed_n = 0; bytes_before = 0; bytes_after = 0
for sh in shards:
    with safe_open(sh, framework="pt") as f:
        for n in f.keys():
            t = f.get_tensor(n)
            bytes_before += t.numel() * t.element_size()
            if n in to_pack:
                out[n] = pack(t).contiguous(); packed_n += 1
            else:
                out[n] = t.contiguous()
            bytes_after += out[n].numel() * out[n].element_size()
print(f"[prepack] packed {packed_n} weights | bytes {bytes_before/2**30:.1f} -> {bytes_after/2**30:.1f} GiB", flush=True)
save_file(out, f"{DST}/model.safetensors", metadata={"format": "pt"})

# 3) copy non-weight files; mark config prepacked
for fn in os.listdir(SRC):
    if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
        continue
    s = os.path.join(SRC, fn)
    if os.path.isfile(s):
        shutil.copy(s, os.path.join(DST, fn))
cfg = json.load(open(f"{DST}/config.json"))
cfg["quantization_config"]["is_prepacked_w4a8"] = True
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
print(f"[prepack] DONE -> {DST} (is_prepacked_w4a8=True)", flush=True)
