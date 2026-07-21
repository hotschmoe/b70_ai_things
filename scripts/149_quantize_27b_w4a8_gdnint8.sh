#!/usr/bin/env bash
# 149: Quantize Qwen3.6-27B -> compressed-tensors W4A8 with the GDN linear_attn projections ALSO
# quantized (int8) -- fixes the 25.77 GiB weight-residency problem of w4a8-sqgptq (JOURNAL 2026-07-21 (g):
# its recipe ignored ALL 432 linear_attn.* tensors -> 10.3 GiB of GDN stayed bf16 -> NO single-card KV
# headroom; UTIL=0.85 @131072 was NEGATIVE KV).
#
# Scheme layout (two compressed-tensors config groups in ONE GPTQ pass):
#   group_0  W4A8   int4 group-128 weights + dynamic per-token int8 act   targets [Linear]   (same as today)
#   group_1  W8A8   int8 CHANNELWISE weights + dynamic per-token int8 act targets the 3 big GDN projections
#            re:.*linear_attn.(in_proj_qkv|in_proj_z|out_proj)$  -- regex name-targets take precedence over
#            the class target, so these 144 weights (48 layers x 3) land int8 instead of int4.
# The int8 GDN on-disk format (I8 [dout,d] + BF16 [dout,1] weight_scale, symmetric) matches the
# w8a8-sqgptq-gdnint8 precedent (models/gdn_int8_requant.py; zml-validated COHERENT, +7% decode) EXCEPT the
# values come from calibrated GPTQ instead of RTN, and -- unlike that variant -- the compressed-tensors
# metadata is CORRECT (group_1 declared, ignore list surgical), so vLLM consumes it natively:
# CompressedTensorsW8A8Int8 -> XPUInt8ScaledMMLinearKernel (the production W8A8 shelf path). NO new kernels.
#
# SURGICAL ignore list (the old blanket re:.*linear_attn.* is the whole bug): keep bf16 =
#   lm_head, re:.*visual.*, re:.*mtp.*, linear_attn.in_proj_b, linear_attn.in_proj_a
# (conv1d is nn.Conv1d, norm/A_log/dt_bias are not Linear -> never matched anyway. vLLM fuses
#  in_proj_qkvz=[qkv,z] (both int8, homogeneous OK) and in_proj_ba=[b,a] (both bf16 ignored, homogeneous OK)
#  -- the fusion boundary aligns with the quant boundary; do NOT split a fused pair across schemes.)
#
# Pipeline: A) GPU quant (49-style: model bf16 on CPU, llmcompressor SequentialPipeline onloads one layer
# at a time to the XPU; selective SmoothQuant + GPTQ actorder=None, ultrachat 128@2048) -> RAW dir.
# B) CPU prepack + finish: pack ONLY the int4 group-128 weights to int32 [out,in/8] (is_prepacked_w4a8,
# avoids the ~28 GiB unpacked-I8 load transient) -- the int8 channelwise GDN weights are detected by their
# [N,1] scale and are NOT packed (research/w4a8/offline_prepack_w4a8.py would corrupt them: it packs on
# .weight_scale presence alone -- do NOT reuse it here); graft-check vision(333)+mtp(15) from the bf16
# source if llmcompressor dropped them (vision-retention directive); split shards model/-visual/-mtp +
# index.json to match the sibling artifacts; verify counts + metadata; write GDN_INT8_NOTE.txt.
#
# Expected size (from the w4a8-sqgptq safetensors headers): main shard 24.121 GiB - 10.312 (bf16 qkv/z/out)
# + 5.156 (int8) + ~0.002 (scales) = ~18.97 GiB; + visual 0.858 + mtp 0.791 = ~20.62 GiB total (vs 25.77).
#
# Env: SRC (bf16 source), OUT, CARD (XPU card for calib onload, DEFAULT 1 -- card 0 usually serves the DD),
#      DEVICE=xpu|cpu, METHOD=gptq|rtn, SAMPLES (128), SEQLEN (2048), SMOOTH (0.8),
#      SMOOTHQUANT=selective|0 (selective: hybrid-safe explicit mappings, plain auto SQ ValueErrors on this
#      arch -- see scripts/49), DATAFREE=1 for a fast RTN pipeline-validation pass, SKIP_QUANT=1 to rerun
#      only stage B on an existing RAW dir, KEEP_RAW=1 to keep the intermediate.
# Run:  cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run --card 1 bash scripts/149_quantize_27b_w4a8_gdnint8.sh
# (GPU lease REQUIRED for stage A. Stage B is CPU-only.)
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="${SRC:-$REPO/models/files/qwen3.6-27b/bf16}"
OUT="${OUT:-$REPO/models/files/qwen3.6-27b/w4a8-sqgptq-gdnint8}"
RAW="${RAW:-$OUT-raw}"
SIBLING="$REPO/models/files/qwen3.6-27b/w8a8-sqgptq"   # preprocessor/processor fallback (JOURNAL (g) FIX 1)
CARD="${CARD:-1}"; DEVICE="${DEVICE:-xpu}"; METHOD="${METHOD:-gptq}"
SAMPLES="${SAMPLES:-128}"; SEQLEN="${SEQLEN:-2048}"; SMOOTH="${SMOOTH:-0.8}"
SMOOTHQUANT="${SMOOTHQUANT:-selective}"; DATAFREE="${DATAFREE:-0}"
SKIP_QUANT="${SKIP_QUANT:-0}"; KEEP_RAW="${KEEP_RAW:-0}"
# Surgical ignore: NO blanket re:.*linear_attn.* -- only the tiny gate projections stay bf16 (precedent:
# w8a8-sqgptq-gdnint8 kept in_proj_b/a, conv1d, A_log, dt_bias, norm bf16).
IGNORE="${IGNORE:-lm_head re:.*visual.* re:.*mtp.* re:.*linear_attn\.in_proj_b$ re:.*linear_attn\.in_proj_a$}"
# The 3 big GDN projections -> int8 (group_1). 10.31 GiB bf16 -> 5.16 GiB int8.
GDN_TARGETS="${GDN_TARGETS:-re:.*linear_attn\.in_proj_qkv$ re:.*linear_attn\.in_proj_z$ re:.*linear_attn\.out_proj$}"
LOG="$REPO/results/logs/149_w4a8_gdnint8_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$REPO/results/logs" "$ROOT/pip_cache"
[ -d "$SRC" ] || { echo "MISSING bf16 source at $SRC"; exit 1; }

GPUARGS=(-e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="")
[ "$DEVICE" = xpu ] && GPUARGS=(--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK="$CARD")
echo "=== 149 W4A8+gdnint8 quant: device=$DEVICE card=$CARD method=$METHOD samples=$SAMPLES" | tee "$LOG"
echo "=== SRC=$SRC" | tee -a "$LOG"
echo "=== RAW=$RAW  OUT=$OUT  log=$LOG" | tee -a "$LOG"

# ---------------------------------------------------------------------------------------------------------
# Stage A -- llmcompressor quant (GPU calib onload). Skippable via SKIP_QUANT=1 (stage-B-only rerun).
# ---------------------------------------------------------------------------------------------------------
if [ "$SKIP_QUANT" != 1 ]; then
docker rm -f q149_quant 2>/dev/null || true
docker run --rm --name q149_quant "${GPUARGS[@]}" --ipc=host --shm-size 32g \
  -v "$REPO:$REPO" -v "$ROOT:$ROOT" -e HF_HOME=/hf_cache -e XDG_CACHE_HOME="$ROOT/vllm_cache" \
  -e OMP_NUM_THREADS=32 -e PIP_CACHE_DIR="$ROOT/pip_cache" \
  -e SRC="$SRC" -e OUT="$RAW" -e DEVICE="$DEVICE" -e METHOD="$METHOD" \
  -e SAMPLES="$SAMPLES" -e SEQLEN="$SEQLEN" -e SMOOTH="$SMOOTH" -e IGNORE="$IGNORE" \
  -e GDN_TARGETS="$GDN_TARGETS" -e DATAFREE="$DATAFREE" -e SMOOTHQUANT="$SMOOTHQUANT" \
  --entrypoint bash "$IMG" -c '
    set -e
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    pip install -q "llmcompressor>=0.8.0" datasets accelerate 2>&1 | tail -2 || true
    python - <<PY
import os, time, torch
from transformers import AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
try: from llmcompressor.modifiers.transform import SmoothQuantModifier
except Exception: from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from compressed_tensors.quantization import preset_name_to_scheme

SRC=os.environ["SRC"]; OUT=os.environ["OUT"]; DEV=os.environ["DEVICE"]; METHOD=os.environ["METHOD"].lower()
N=int(os.environ["SAMPLES"]); SEQ=int(os.environ["SEQLEN"]); SMOOTH=float(os.environ["SMOOTH"])
DATAFREE=os.environ.get("DATAFREE","0")=="1"
SQMODE=os.environ.get("SMOOTHQUANT","selective").strip().lower()
USE_SMOOTH = SQMODE not in ("0","false","no","off","")
SELECTIVE = SQMODE in ("selective","sel","hybrid")
IGN=[p for p in os.environ["IGNORE"].split() if p]
GDN=[p for p in os.environ["GDN_TARGETS"].split() if p]
xpu_ok=hasattr(torch,"xpu") and torch.xpu.is_available()
print(f"[probe] xpu={xpu_ok} device={DEV} datafree={DATAFREE}", flush=True)
print(f"[cfg] ignore={IGN}", flush=True)
print(f"[cfg] gdn int8 targets={GDN}", flush=True)

# Two config groups in one pass. Name-regex targets take precedence over the class target [Linear]
# in compressed-tensors matching, so the GDN projections resolve to group_1 (int8 channelwise), not
# group_0 (int4 group-128). Presets == what the sibling artifacts serve with today:
#   W4A8: weights int4 sym group-128 + input int8 dynamic per-token  (w4a8-sqgptq group_0)
#   W8A8: weights int8 sym CHANNEL   + input int8 dynamic per-token  (w8a8-sqgptq group_0)
g0=preset_name_to_scheme("W4A8", ["Linear"])
g1=preset_name_to_scheme("W8A8", GDN)
g0.weights.actorder=None   # no act reorder (avoids the XPU gather device-lost; see scripts/49)
g1.weights.actorder=None
CG={"group_0": g0, "group_1": g1}
for nm,g in CG.items():
    print(f"[cfg] {nm}: targets={g.targets} w={g.weights.num_bits}b/{g.weights.strategy} "
          f"group_size={g.weights.group_size}", flush=True)

print(f"[load] {SRC} bf16 on CPU (llmcompressor onloads layers to the GPU per-step)...", flush=True)
def _load():
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16, device_map="cpu",
                 low_cpu_mem_usage=True, trust_remote_code=True)
    except (ValueError, KeyError, RuntimeError) as e:
        print(f"[load] AutoModelForCausalLM failed ({type(e).__name__}: {e}); trying VLM loaders", flush=True)
    for cls_name in ("AutoModelForImageTextToText","AutoModelForVision2Seq","AutoModel"):
        try:
            import transformers as T
            cls=getattr(T, cls_name)
            m=cls.from_pretrained(SRC, dtype=torch.bfloat16, device_map="cpu",
                 low_cpu_mem_usage=True, trust_remote_code=True)
            print(f"[load] loaded via {cls_name}", flush=True); return m
        except Exception as e:
            print(f"[load] {cls_name} failed ({type(e).__name__}: {e})", flush=True)
    raise SystemExit("FAIL: no loader could instantiate the model")
model=_load()
tok=AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

# Sanity: the GDN targets must actually match modules (regex vs the nested VLM names -- the exact
# failure mode of w8a8 serve bug A was a regex that silently matched nothing).
import re as _re
mods=dict(model.named_modules())
hit=0
for name,mod in mods.items():
    for pat in GDN:
        if _re.match(pat[3:], name):
            assert isinstance(mod, torch.nn.Linear), f"GDN target is not Linear: {name} {type(mod)}"
            hit+=1; break
print(f"[cfg] gdn targets matched {hit} modules (expect 144 = 48 layers x 3)", flush=True)
assert hit==144, f"FAIL: expected 144 GDN projection modules, matched {hit}"

if DATAFREE:
    recipe=[QuantizationModifier(config_groups=CG, ignore=IGN)]
    print(f"[quant] DATA-FREE RTN W4A8+gdnint8 (pipeline validation) ...", flush=True)
    t0=time.time()
    oneshot(model=model, recipe=recipe)
else:
    from datasets import load_dataset
    ds=load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N}]").shuffle(seed=42)
    ds=ds.map(lambda e: {"text": tok.apply_chat_template(e["messages"], tokenize=False)})
    ds=ds.map(lambda s: tok(s["text"], padding=False, max_length=SEQ, truncation=True, add_special_tokens=False),
              remove_columns=ds.column_names)
    if METHOD=="gptq":
        q=GPTQModifier(config_groups=CG, ignore=IGN)
    else:
        q=QuantizationModifier(config_groups=CG, ignore=IGN)
    # Selective SmoothQuant (Q0 Playbook-B, copied from scripts/49): explicit maps ONLY where pairing is
    # clean (full-attn q/k/v, MLP gate/up); DeltaNet linear_attn / vision / MTP are NOT smoothed -- so the
    # GDN int8 group quantizes UNsmoothed weights, same as the RTN precedent.
    # NOTE: keep this heredoc free of apostrophes/single-quotes -- it lives inside bash -c quoted.
    def _selective_sq_mappings(m):
        import torch as _t
        names = dict(m.named_modules())
        maps = []
        n_attn = n_mlp = 0
        for name, mod in names.items():
            if isinstance(mod, _t.nn.Linear) and name.endswith(".self_attn.q_proj"):
                pre = name[:-len(".self_attn.q_proj")]
                ln = pre + ".input_layernorm"
                q_ = pre + ".self_attn.q_proj"; k_ = pre + ".self_attn.k_proj"
                v_ = pre + ".self_attn.v_proj"
                if ln in names and k_ in names and v_ in names:
                    maps.append([[q_, k_, v_], ln]); n_attn += 1
                # NO o_proj<-v_proj map: qwen3 GQA dim mismatch (see scripts/49).
        for name, mod in names.items():
            if isinstance(mod, _t.nn.Linear) and name.endswith(".mlp.gate_proj"):
                pre = name[:-len(".mlp.gate_proj")]
                ln = pre + ".post_attention_layernorm"; u_ = pre + ".mlp.up_proj"
                if ln in names and u_ in names:
                    maps.append([[pre + ".mlp.gate_proj", u_], ln]); n_mlp += 1
        print("[selective-sq] attn_layers=%d mlp_layers=%d mappings=%d" % (n_attn, n_mlp, len(maps)), flush=True)
        if not maps:
            raise SystemExit("FAIL: selective SmoothQuant produced 0 mappings -- arch module names changed?")
        return maps
    if SELECTIVE:
        sq = SmoothQuantModifier(smoothing_strength=SMOOTH, mappings=_selective_sq_mappings(model))
        recipe = [sq, q]; pfx = "selective-SmoothQuant+"
    elif USE_SMOOTH:
        recipe = [SmoothQuantModifier(smoothing_strength=SMOOTH), q]; pfx = "SmoothQuant+"
    else:
        recipe = [q]; pfx = ""
    print(f"[quant] {pfx}{METHOD} W4A8+gdnint8, ignore={IGN} ...", flush=True)
    t0=time.time()
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQ, num_calibration_samples=N)
print(f"[done] calib+quant {time.time()-t0:.0f}s; saving compressed -> {OUT}", flush=True)
model.save_pretrained(OUT, save_compressed=True); tok.save_pretrained(OUT)
try:
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(SRC, trust_remote_code=True).save_pretrained(OUT)
    print("[save] processor (preprocessor_config etc.) saved", flush=True)
except Exception as e:
    print(f"[save] AutoProcessor save skipped ({type(e).__name__}: {e}) -- stage B falls back to file copy", flush=True)
print("DONE_149_STAGE_A", OUT, flush=True)
PY
  ' 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
[ "$RC" = 0 ] || { echo "=== stage A FAILED rc=$RC (log $LOG) ==="; exit "$RC"; }
grep -q "DONE_149_STAGE_A" "$LOG" || { echo "=== stage A did not complete (no DONE marker) ==="; exit 1; }
fi

# ---------------------------------------------------------------------------------------------------------
# Stage B -- CPU-only prepack + graft-check + shard split + metadata verify. NO GPU, NO lease needed.
# ---------------------------------------------------------------------------------------------------------
docker rm -f q149_pack 2>/dev/null || true
docker run --rm --name q149_pack -v "$REPO:$REPO" -v "$ROOT:$ROOT" \
  -e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" -e OMP_NUM_THREADS=16 \
  -e SRC="$SRC" -e RAW="$RAW" -e OUT="$OUT" -e SIBLING="$SIBLING" \
  --entrypoint bash "$IMG" -c '
    set -e
    python - <<PY
import os, json, glob, shutil, collections
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC=os.environ["SRC"]; RAW=os.environ["RAW"]; OUT=os.environ["OUT"]; SIB=os.environ["SIBLING"]
os.makedirs(OUT, exist_ok=True)
SHIFTS=torch.arange(0, 32, 4, dtype=torch.int32)

def pack_int4(w):
    # byte-match XPUW4A8IntLinearKernel _pack_int4_weight: [-8,7] -> +8 -> nibbles -> int32 [N, K//8]
    assert w.dtype==torch.int8 and w.shape[1]%8==0, f"bad int4 weight {w.dtype} {tuple(w.shape)}"
    assert int(w.min())>=-8 and int(w.max())<=7, f"int4 range violated [{int(w.min())},{int(w.max())}] -- refusing to pack"
    u=(w.to(torch.int32)+8).reshape(w.shape[0], w.shape[1]//8, 8)
    return ((u & 0xF) << SHIFTS[None,None,:]).sum(dim=2).to(torch.int32)

def shard_names(d):
    names={}
    for sh in sorted(glob.glob(os.path.join(d,"*.safetensors"))):
        with safe_open(sh, framework="pt") as f:
            for n in f.keys(): names[n]=sh
    return names

raw=shard_names(RAW)
scales={n[:-len(".weight_scale")] for n in raw if n.endswith(".weight_scale")}
SKIP_PACK=("lm_head","embed_tokens","embed")  # vocab layers use their own loader; bf16 anyway (defensive)

def load(names, n):
    with safe_open(names[n], framework="pt") as f:
        return f.get_tensor(n)

out={}; n_pack=0; n_int8=0
for n in sorted(raw):
    t=load(raw, n)
    p=n[:-len(".weight")] if n.endswith(".weight") else None
    if p and p in scales and not any(s in p for s in SKIP_PACK):
        sc=load(raw, p+".weight_scale")
        if sc.ndim==2 and sc.shape[1]==1:
            # CHANNELWISE int8 (group_1, the GDN projections): do NOT pack -- served unpacked by the
            # W8A8 int8 kernel. (offline_prepack_w4a8.py would have corrupted these.)
            assert t.dtype==torch.int8, f"channelwise weight not int8: {n} {t.dtype}"
            assert "linear_attn" in n, f"unexpected channelwise int8 outside linear_attn: {n}"
            out[n]=t.contiguous(); n_int8+=1
        else:
            out[n]=pack_int4(t).contiguous(); n_pack+=1
    else:
        out[n]=t.contiguous()
print(f"[pack] int4-packed={n_pack} (expect 256 = 16x4 self_attn + 64x3 mlp)  int8-kept={n_int8} (expect 144 = 48x3 GDN)", flush=True)
assert n_pack==256, f"FAIL: packed {n_pack} != 256"
assert n_int8==144, f"FAIL: int8 GDN {n_int8} != 144"

# --- graft-check: vision + mtp must be RETAINED (some llmcompressor saves silently drop them). Pull any
# missing model.visual.* / mtp.* tensors verbatim (bf16) from the SRC checkpoint.
def is_visual(n): return n.startswith("model.visual.") or n.startswith("visual.")
def is_mtp(n):    return n.startswith("mtp.") or n.startswith("model.mtp.")
src=shard_names(SRC)
grafted=0
for n in sorted(src):
    if (is_visual(n) or is_mtp(n)) and n not in out:
        out[n]=load(src, n).contiguous(); grafted+=1
nv=sum(1 for n in out if is_visual(n)); nm=sum(1 for n in out if is_mtp(n))
print(f"[graft] grafted={grafted} from SRC; visual tensors={nv} (expect 333) mtp tensors={nm} (expect 15)", flush=True)
assert nv>0 and nm>0, "FAIL: vision or MTP tensors missing from BOTH the quant output and SRC"
if nv!=333 or nm!=15: print(f"[graft] WARN: counts differ from the sibling artifacts (333/15)", flush=True)

# --- split into the house 3-shard layout + index.json (matches w4a8-sqgptq / w8a8-sqgptq siblings).
shards={"model-visual.safetensors":{}, "model-mtp.safetensors":{}, "model.safetensors":{}}
for n,t in out.items():
    if is_visual(n): shards["model-visual.safetensors"][n]=t
    elif is_mtp(n):  shards["model-mtp.safetensors"][n]=t
    else:            shards["model.safetensors"][n]=t
wmap={}; total=0
for sh,d in shards.items():
    save_file(d, os.path.join(OUT, sh), metadata={"format":"pt"})
    nb=sum(t.numel()*t.element_size() for t in d.values()); total+=nb
    for n in d: wmap[n]=sh
    print(f"[write] {sh}: {len(d)} tensors, {nb/2**30:.3f} GiB", flush=True)
json.dump({"metadata":{"total_size":total},"weight_map":wmap},
          open(os.path.join(OUT,"model.safetensors.index.json"),"w"))
print(f"[write] index.json total_size {total/2**30:.3f} GiB (w4a8-sqgptq was 25.770; expected ~20.6)", flush=True)

# --- aux files: RAW first, then SRC, then the w8a8-sqgptq sibling (JOURNAL (g) FIX 1: preprocessor_config).
AUX=("config.json","generation_config.json","recipe.yaml","tokenizer.json","tokenizer_config.json",
     "chat_template.jinja","preprocessor_config.json","processor_config.json","merges.txt","vocab.json",
     "special_tokens_map.json")
for fn in AUX:
    for d in (RAW, SRC, SIB):
        s=os.path.join(d, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(OUT, fn)); break
    else:
        print(f"[aux] note: {fn} not found in RAW/SRC/sibling (may be fine)", flush=True)

# --- metadata verify + is_prepacked_w4a8. The WHOLE POINT vs the gdnint8 precedent: metadata must be
# correct for the metadata-driven vLLM loader (GDN_INT8_NOTE.txt of the w8a8 variant documents the trap).
cfg=json.load(open(os.path.join(OUT,"config.json")))
qc=cfg.get("quantization_config") or {}
ign=qc.get("ignore") or []
assert qc, "FAIL: no quantization_config in config.json"
assert not any("linear_attn.*" in p for p in ign), f"FAIL: blanket linear_attn ignore survived: {ign}"
assert any("in_proj_b" in p for p in ign) and any("in_proj_a" in p for p in ign), \
    f"FAIL: in_proj_b/in_proj_a must STAY ignored (vLLM fused in_proj_ba is bf16): {ign}"
groups=qc.get("config_groups") or {}
g_int8=[g for g in groups.values()
        if g.get("weights",{}).get("num_bits")==8 and g.get("weights",{}).get("strategy")=="channel"]
g_int4=[g for g in groups.values()
        if g.get("weights",{}).get("num_bits")==4 and g.get("weights",{}).get("strategy")=="group"]
assert g_int8 and g_int4, f"FAIL: need both an int4-group and an int8-channel config group: {list(groups)}"
assert any("linear_attn" in t for g in g_int8 for t in g.get("targets",[])), \
    "FAIL: int8 group does not target linear_attn"
qc["is_prepacked_w4a8"]=True
cfg["quantization_config"]=qc
json.dump(cfg, open(os.path.join(OUT,"config.json"),"w"), indent=2)
print("[verify] metadata OK: surgical ignore, int8-channel GDN group present, is_prepacked_w4a8=True", flush=True)

with open(os.path.join(OUT,"GDN_INT8_NOTE.txt"),"w") as f:
    f.write(
        "qwen3.6-27b W4A8 (sq+gptq, int4 group-128 + dynamic int8 act) with the 3 big GDN\n"
        "linear_attn projections (in_proj_qkv, in_proj_z, out_proj; 48 layers x 3 = 144 weights)\n"
        "quantized int8 CHANNELWISE (I8 [dout,d] + BF16 [dout,1] weight_scale, symmetric) --\n"
        "same on-disk scheme as the w8a8-sqgptq-gdnint8 precedent, but GPTQ-calibrated and,\n"
        "UNLIKE that variant, with CORRECT compressed-tensors metadata: config_groups declares\n"
        "an int8-channel group targeting the projections and the ignore list is surgical\n"
        "(only lm_head, visual, mtp, linear_attn.in_proj_b, linear_attn.in_proj_a stay bf16;\n"
        "conv1d/A_log/dt_bias/norm are not Linear). vLLM consumes the GDN int8 natively via\n"
        "CompressedTensorsW8A8Int8 -> XPUInt8ScaledMMLinearKernel; the fused in_proj_qkvz\n"
        "(int8+int8) and in_proj_ba (bf16+bf16) stay scheme-homogeneous.\n"
        "int4 weights are PREPACKED int32 [out,in/8] (is_prepacked_w4a8=true); the int8 GDN\n"
        "weights are NOT packed. Produced by scripts/149_quantize_27b_w4a8_gdnint8.sh.\n"
        "GPU-gated acceptance still required: serve via rdy_to_serve/vllm/qwen36-27b-w4a8\n"
        "with MODEL= pointed here, Paris probe + gate_concurrent_coherence 18/18 + HumanEval+.\n")
print("DONE_149_STAGE_B", OUT, flush=True)
PY
  ' 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
[ "$RC" = 0 ] || { echo "=== stage B FAILED rc=$RC (log $LOG) ==="; exit "$RC"; }
grep -q "DONE_149_STAGE_B" "$LOG" || { echo "=== stage B did not complete ==="; exit 1; }

[ "$KEEP_RAW" = 1 ] || { echo "=== removing RAW intermediate $RAW (KEEP_RAW=1 to keep) ==="; rm -rf "$RAW"; }
echo "=== 149 DONE -> $OUT ==="; du -sh "$OUT" 2>/dev/null
echo "Next: serve gate on card 1 via MODEL=$OUT SERVED_FORCE=qwen36-27b-w4a8-sqgptq-gdnint8 \\"
echo "      rdy_to_serve/vllm/qwen36-27b-w4a8/serve.sh (int8g-v0251); then Paris probe, 18/18 gate, HumanEval+."
