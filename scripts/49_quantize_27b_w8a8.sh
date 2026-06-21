#!/usr/bin/env bash
# Quantize Qwen3.6-27B (DeltaNet) -> compressed-tensors W8A8 INT8 with GOOD calibration (SmoothQuant +
# GPTQ) to stay close to W8A16/BF16 accuracy. GPU-ACCELERATED: runs in the XPU image so llmcompressor
# uses the B70 ("Accelerator 0") for the per-layer calibration forward passes (the model itself stays on
# CPU -- 54GB BF16 in 125GB RAM -- and llmcompressor's SequentialPipeline onloads one layer at a time to
# the GPU). GPTQ actorder=False to avoid the H[perm][:,perm] gather that device-lost before.
#
# IGNORE list keeps the DeltaNet linear-attn projections + MTP head + lm_head in BF16 (quantizing them
# hurts accuracy / they are not group-divisible) -- only the standard attn/MLP linears go int8 (our kernel).
#
# Env: SRC (27B BF16 path), OUTNAME, DEVICE=xpu|cpu (default xpu), METHOD=gptq|rtn (default gptq),
#      SAMPLES (default 512), SEQLEN (default 2048), SMOOTH (0.8), SMOOTHQUANT=1|0 (default 1; set 0 to
#      SKIP SmoothQuant -- REQUIRED for the hybrid Qwen3_5 27B, whose 16/64 full-attn layers break
#      SmoothQuant's smooth-layer<->q/k/v pairing -> ValueError), IGNORE (space-sep patterns), DATAFREE=0.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="${SRC:-/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B}"
SCHEME="${SCHEME:-W8A8}"   # W8A8 | W4A16 | W8A16 | W4A8 (compressed-tensors preset)
DEVICE="${DEVICE:-xpu}"; METHOD="${METHOD:-gptq}"; SAMPLES="${SAMPLES:-512}"; SEQLEN="${SEQLEN:-2048}"; SMOOTH="${SMOOTH:-0.8}"
DATAFREE="${DATAFREE:-0}"
# Tag the output dir by method so RTN and GPTQ NEVER collide / get mixed up downstream (rtn vs gptq).
QMETH="$METHOD"; [ "$DATAFREE" = 1 ] && QMETH="rtn"
OUTNAME="${OUTNAME:-Qwen3.6-27B-${SCHEME}-${QMETH}}"
# *A16 schemes are weight-only -> SmoothQuant is a no-op; default it off for those.
case "$SCHEME" in *A16) SQD=0;; *) SQD=1;; esac
SMOOTHQUANT="${SMOOTHQUANT:-$SQD}"
# Qwen3.6-27B is a Qwen3_5 VLM (vision tower + DeltaNet/full-attn text + MTP). Quantize only the standard
# self_attn + MLP linears; keep DeltaNet (linear_attn), the WHOLE vision tower, MTP, and lm_head in BF16.
# (Confirmed against the safetensors weight map: model.language_model.layers.N.{self_attn,mlp,linear_attn},
#  model.visual.*, mtp.*, lm_head — see JOURNAL 2026-06-19.)
IGNORE="${IGNORE:-lm_head re:.*linear_attn.* re:.*visual.* re:.*mtp.*}"
LOG="$ROOT/results/quant27b_w8a8_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/models" "$ROOT/pip_cache"
[ -d "$SRC" ] || { echo "MISSING 27B weights at $SRC (download Qwen/Qwen3.6-27B first)"; exit 1; }

GPUARGS=(-e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="")
[ "$DEVICE" = xpu ] && GPUARGS=(--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0)
echo "=== 27B W8A8 quant: device=$DEVICE method=$METHOD samples=$SAMPLES ignore=[$IGNORE]  log=$LOG ==="

docker run --rm --name quant27b "${GPUARGS[@]}" --ipc=host --shm-size 32g \
  -v "$ROOT:$ROOT" -e HF_HOME=/hf_cache -e XDG_CACHE_HOME="$ROOT/vllm_cache" \
  -e OMP_NUM_THREADS=32 -e PIP_CACHE_DIR="$ROOT/pip_cache" \
  -e SRC="$SRC" -e OUT="$ROOT/models/$OUTNAME" -e DEVICE="$DEVICE" -e METHOD="$METHOD" \
  -e SAMPLES="$SAMPLES" -e SEQLEN="$SEQLEN" -e SMOOTH="$SMOOTH" -e IGNORE="$IGNORE" -e DATAFREE="$DATAFREE" \
  -e SMOOTHQUANT="$SMOOTHQUANT" -e SCHEME="$SCHEME" \
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

SRC=os.environ["SRC"]; OUT=os.environ["OUT"]; DEV=os.environ["DEVICE"]; METHOD=os.environ["METHOD"].lower()
N=int(os.environ["SAMPLES"]); SEQ=int(os.environ["SEQLEN"]); SMOOTH=float(os.environ["SMOOTH"])
DATAFREE=os.environ.get("DATAFREE","0")=="1"
SQMODE=os.environ.get("SMOOTHQUANT","1").strip().lower()
USE_SMOOTH = SQMODE not in ("0","false","no","off","")
SELECTIVE = SQMODE in ("selective","sel","hybrid")
SCHEME=os.environ.get("SCHEME","W8A8")
IGN=[p for p in os.environ["IGNORE"].split() if p]
xpu_ok=hasattr(torch,"xpu") and torch.xpu.is_available()
print(f"[probe] xpu={xpu_ok} device={DEV} datafree={DATAFREE}", flush=True)
print(f"[load] {SRC} bf16 on CPU (llmcompressor onloads layers to the GPU per-step)...", flush=True)
# Qwen3_5 is a ForConditionalGeneration VLM -> AutoModelForCausalLM may not map it; fall back gracefully.
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

if DATAFREE:
    # Fast RTN data-free validation pass (no SmoothQuant, no dataset) -> DataFreePipeline (~minutes).
    # Use this FIRST to confirm the pipeline + ignore-list work on this VLM and that vLLM can load the result.
    recipe=[QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=IGN)]
    print(f"[quant] DATA-FREE RTN W8A8, ignore={IGN} ...", flush=True)
    t0=time.time()
    oneshot(model=model, recipe=recipe)
else:
    from datasets import load_dataset
    ds=load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N}]").shuffle(seed=42)
    ds=ds.map(lambda e: {"text": tok.apply_chat_template(e["messages"], tokenize=False)})
    ds=ds.map(lambda s: tok(s["text"], padding=False, max_length=SEQ, truncation=True, add_special_tokens=False),
              remove_columns=ds.column_names)
    if METHOD=="gptq":
        q=GPTQModifier(targets="Linear", scheme=SCHEME, ignore=IGN, actorder=None)  # actorder None: no act reorder (avoids XPU gather device-lost); =False was rejected by newer llmcompressor
    else:
        q=QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=IGN)
    # SmoothQuant resolver needs exactly one smooth-layer per balance group; the hybrid Qwen3_5 (only
    # 16/64 layers carry self_attn q/k/v) breaks the auto/all-layers pairing -> ValueError before any GPU
    # work. SMOOTHQUANT=0 = GPTQ-only (strong W8A8). SMOOTHQUANT=selective = Q0 Playbook-B mappings below:
    # explicit per-layer maps ONLY where pairing is clean (full-attn q/k/v + o<-v, MLP gate/up, MoE experts),
    # skipping DeltaNet linear_attn / vision / MTP -- which is what made the auto resolver throw.
    # NOTE: keep this heredoc free of apostrophes/single-quotes -- it lives inside bash -c quoted.
    def _selective_sq_mappings(m):
        import torch as _t
        names = dict(m.named_modules())
        maps = []
        n_attn = n_mlp = n_moe = 0
        for name, mod in names.items():
            if isinstance(mod, _t.nn.Linear) and name.endswith(".self_attn.q_proj"):
                pre = name[:-len(".self_attn.q_proj")]
                ln = pre + ".input_layernorm"
                q_ = pre + ".self_attn.q_proj"; k_ = pre + ".self_attn.k_proj"
                v_ = pre + ".self_attn.v_proj"; o_ = pre + ".self_attn.o_proj"
                if ln in names and k_ in names and v_ in names:
                    maps.append([[q_, k_, v_], ln]); n_attn += 1
                # NOTE: NO o_proj<-v_proj mapping. qwen3 is GQA (kv_heads << q_heads) so v_proj out-dim
                # (kv_heads*head_dim) != o_proj in-dim (q_heads*head_dim) -> SmoothQuant scale dim mismatch
                # (RuntimeError tensor a(1024) vs b(6144)). Standard SmoothQuant omits o<-v for GQA. o_proj
                # is still GPTQ-quantized, just not smoothed (a minor refinement, safe to drop).
        for name, mod in names.items():
            if isinstance(mod, _t.nn.Linear) and name.endswith(".mlp.gate_proj"):
                pre = name[:-len(".mlp.gate_proj")]
                ln = pre + ".post_attention_layernorm"; u_ = pre + ".mlp.up_proj"
                if ln in names and u_ in names:
                    maps.append([[pre + ".mlp.gate_proj", u_], ln]); n_mlp += 1
        seen = set()
        for name, mod in names.items():
            if isinstance(mod, _t.nn.Linear) and ".mlp.experts." in name and name.endswith(".gate_proj"):
                head = name.split(".mlp.experts.")[0]
                if head in seen:
                    continue
                seen.add(head)
                ln = head + ".post_attention_layernorm"
                bal = [nm for nm in names if nm.startswith(head + ".mlp.experts.")
                       and (nm.endswith(".gate_proj") or nm.endswith(".up_proj"))]
                if ln in names and bal:
                    maps.append([bal, ln]); n_moe += 1
        print("[selective-sq] attn_layers=%d mlp_layers=%d moe_layers=%d mappings=%d"
              % (n_attn, n_mlp, n_moe, len(maps)), flush=True)
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
    print(f"[quant] {pfx}{METHOD} {SCHEME}, ignore={IGN} ...", flush=True)
    t0=time.time()
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQ, num_calibration_samples=N)
print(f"[done] calib+quant {time.time()-t0:.0f}s; saving compressed -> {OUT}", flush=True)
model.save_pretrained(OUT, save_compressed=True); tok.save_pretrained(OUT)
print("DONE_27B_W8A8", OUT, flush=True)
PY
  ' 2>&1 | tee "$LOG"
echo "=== exit ${PIPESTATUS[0]} ==="; du -sh "$ROOT/models/$OUTNAME" 2>/dev/null
