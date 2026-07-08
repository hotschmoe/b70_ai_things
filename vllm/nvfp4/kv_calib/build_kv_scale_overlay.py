# Build a non-destructive fp8-KV-scale overlay for the NVFP4 27B checkpoint.
# Input : kv_amax.json {layer_name: [k_amax, v_amax]} from the calibration hook.
# Output: <outdir>/model-kvscales.safetensors  (f32 scalar k/v_scale per full-attn layer)
#         <outdir>/model.safetensors.index.json (COPY of the real index + the new tensors)
# Naming: checkpoint convention  model.language_model.layers.N.self_attn.{k_proj.k_scale,v_proj.v_scale}
#         (matches the sibling *.k_proj.input_scale; vLLM maybe_remap_kv_scale_name routes it to
#          ...self_attn.attn.{k,v}_scale). scale = amax / (448 * HEADROOM).
import os, re, json, sys
import torch
from safetensors.torch import save_file

AMAX   = os.environ.get("KV_AMAX", "kv_amax.json")
CKPT   = os.environ.get("CKPT", "models/files/qwen3.6-27b/nvfp4-modelopt")
OUTDIR = os.environ.get("OUTDIR", "vllm/nvfp4/kv_calib/overlay")
E4M3MAX = 448.0
HEADROOM = float(os.environ.get("KV_HEADROOM", "1.0"))  # >1 => larger scale => more clip headroom
os.makedirs(OUTDIR, exist_ok=True)

amax = json.load(open(AMAX))
tensors = {}
rows = []
for ln, (ka, va) in amax.items():
    m = re.search(r"layers\.(\d+)\.", ln)
    if not m:  # skip mtp / non-layer
        print("[skip non-layer]", ln); continue
    N = int(m.group(1))
    if "mtp" in ln:  # handle the drafter separately if ever needed
        print("[skip mtp]", ln); continue
    ks = ka / (E4M3MAX * HEADROOM)
    vs = va / (E4M3MAX * HEADROOM)
    kk = f"model.language_model.layers.{N}.self_attn.k_proj.k_scale"
    vk = f"model.language_model.layers.{N}.self_attn.v_proj.v_scale"
    tensors[kk] = torch.tensor(ks, dtype=torch.float32)
    tensors[vk] = torch.tensor(vs, dtype=torch.float32)
    rows.append((N, ka, va, ks, vs))

rows.sort()
print(f"{len(rows)} full-attn layers calibrated (headroom={HEADROOM}):")
for N, ka, va, ks, vs in rows:
    print(f"  L{N:2d}  k_amax={ka:8.3f} -> k_scale={ks:.5f}   v_amax={va:8.3f} -> v_scale={vs:.5f}")

shard = os.path.join(OUTDIR, "model-kvscales.safetensors")
save_file(tensors, shard)
print("wrote", shard, "with", len(tensors), "scalar tensors")

# COPY the index and append the new tensors to weight_map
idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
for k in tensors:
    idx["weight_map"][k] = "model-kvscales.safetensors"
outidx = os.path.join(OUTDIR, "model.safetensors.index.json")
json.dump(idx, open(outidx, "w"))
print("wrote", outidx, "(", len(idx["weight_map"]), "weight_map entries )")
print("\nSERVE OVERLAY MOUNTS (add to docker run):")
abss = os.path.abspath(shard); absi = os.path.abspath(outidx)
print(f'  -v {abss}:/models/qwen3.6-27b/nvfp4-modelopt/model-kvscales.safetensors:ro')
print(f'  -v {absi}:/models/qwen3.6-27b/nvfp4-modelopt/model.safetensors.index.json:ro')
