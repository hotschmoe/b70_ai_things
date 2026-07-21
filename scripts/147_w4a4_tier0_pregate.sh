#!/usr/bin/env bash
# Track C (W4A4 rotation) Phase C1 Step 2: tier0 PRE-GATE for the 14B W4A4 checkpoints.
# CPU-ONLY. Scores the (non-servable-yet) W4A4 quant OFFLINE on the tier0 corpus with the
# SAME per-token dump format as scripts/55_tier0_reference_cpu.sh, then compares vs the
# bf16 reference via evals/orchestrator/tier0_divergence.py compare
# (ppl / top1_agreement / nll_gap). Run once for the rotated quant and once with
# CKPT=...-norot QLABEL=w4a4-gptq-norot for the rotation ablation.
#
# Load path mirrors scripts/146 Phase B: compressed-tensors QDQ (run_compressed=False)
# + manual re-apply of the ONLINE transform entries (R4 input-side Hadamard) -- transformers
# does not apply transform_config on load; weight-side entries are already fused.
#
# Env:
#   CKPT    default /mnt/vm_8tb/b70/quant_work/Qwen3-14B-W4A4-quarot-gptq
#   QLABEL  default w4a4-quarot-gptq (encodes method+scheme per Model Identity rules)
#   SRC     bf16 reference source (default /specula_models/Qwen3-14B; scripts/55 arg)
#   REF     reference dump (default $ROOT/results/tier0_ref_bf16_tokens.json)
#   REF_AUTO 1 (default) = build the bf16 reference via scripts/55 if REF is missing
#            (CPU-only, safe alongside serving; ~1h for the 5KB corpus on 14B).
#   CORPUS  default $ROOT/tier0_corpus.txt (staged from evals/prompts/tier0_corpus.txt)
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
REPO=/mnt/vm_8tb/github/b70_ai_things
SPECULA=/mnt/vm_8tb/specula-build/models
IMG="${IMG:-vllm-xpu-env:int8g-v0251}"
CKPT="${CKPT:-$ROOT/quant_work/Qwen3-14B-W4A4-quarot-gptq}"
QLABEL="${QLABEL:-w4a4-quarot-gptq}"
SRC="${SRC:-/specula_models/Qwen3-14B}"
CORPUS="${CORPUS:-$ROOT/tier0_corpus.txt}"
REF="${REF:-$ROOT/results/tier0_ref_bf16_tokens.json}"
QDUMP="$ROOT/results/tier0_ref_${QLABEL}_tokens.json"
LOG="$ROOT/results/tier0_pregate_${QLABEL}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/pip_cache"

[ -d "$CKPT" ] || { echo "MISSING checkpoint $CKPT (run scripts/146 first)"; exit 1; }
# Stage the corpus from the repo if not on the runtime root yet.
[ -f "$CORPUS" ] || { echo "[stage] corpus -> $CORPUS"; cp "$REPO/evals/prompts/tier0_corpus.txt" "$CORPUS"; }
# Build the bf16 reference dump if missing (scripts/55, CPU-only).
if [ ! -f "$REF" ]; then
  if [ "${REF_AUTO:-1}" = 1 ]; then
    echo "[ref] $REF missing -> building via scripts/55_tier0_reference_cpu.sh (CPU, ~1h)"
    SRC="$SRC" CORPUS="$CORPUS" OUT="$REF" bash "$REPO/scripts/55_tier0_reference_cpu.sh" || exit 1
  else
    echo "MISSING reference dump $REF (run scripts/55_tier0_reference_cpu.sh or set REF_AUTO=1)"; exit 1
  fi
fi

echo "=== tier0 pre-gate: ckpt=$CKPT qlabel=$QLABEL ref=$REF ==="
echo "=== log: $LOG ==="

# --- score the quant offline (QDQ + online R4), scripts/55-compatible dump ---
docker run --rm --name "tier0_pregate" \
  -e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" --cpu-shares 512 \
  --ipc=host --shm-size 16g --memory 90g \
  -v "$ROOT:$ROOT" -v "$SPECULA:/specula_models:ro" \
  -e HF_HOME="$ROOT/hf_cache" -e OMP_NUM_THREADS="${OMP:-24}" -e PIP_CACHE_DIR="$ROOT/pip_cache" \
  -e CKPT="$CKPT" -e CORPUS="$CORPUS" -e OUT="$QDUMP" -e QLABEL="$QLABEL" \
  --entrypoint bash "$IMG" -c '
    set -e
    pip install -q "llmcompressor>=0.8.0" 2>&1 | tail -1 || true
    python - <<PY
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig
CKPT=os.environ["CKPT"]; CORPUS=os.environ["CORPUS"]; OUT=os.environ["OUT"]
raw=open(CORPUS).read()
passages=[p.strip() for p in raw.split("\n---\n") if len(p.strip())>40]
print(f"[quant-score] {len(passages)} passages; loading {CKPT} QDQ on CPU...", flush=True)
tok=AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, device_map="cpu",
        quantization_config=CompressedTensorsConfig(run_compressed=False),
        trust_remote_code=True).eval()

# Re-apply ONLINE transform entries (R4 down_proj input Hadamard); transformers skips
# transform_config at load. Weight-side entries are fused in the checkpoint already.
tcfg_dict=json.load(open(CKPT+"/config.json"))["quantization_config"].get("transform_config")
if tcfg_dict:
    from compressed_tensors.transform import TransformConfig, apply_transform_config
    online_groups={}
    for gname,scheme in TransformConfig.model_validate(tcfg_dict).config_groups.items():
        online=[a for a in scheme.apply if str(a.location) in ("input","output")]
        if online:
            s=scheme.model_copy(); s.apply=online
            online_groups[gname]=s
    if online_groups:
        print(f"[quant-score] applying ONLINE transforms: {list(online_groups)}", flush=True)
        apply_transform_config(model, TransformConfig(config_groups=online_groups))
else:
    print("[quant-score] no transform_config (norot ablation) -- plain QDQ", flush=True)

# Scoring loop mirrors scripts/55 EXACTLY: add_special_tokens=True, index 0 = None.
out=[]
for j,text in enumerate(passages):
    ids=tok(text, add_special_tokens=True).input_ids
    t=torch.tensor([ids])
    with torch.no_grad():
        logits=model(t).logits[0].float()
    lp=torch.log_softmax(logits, dim=-1)
    argmax=[None]; actual=[None]
    for i in range(1, len(ids)):
        argmax.append(int(logits[i-1].argmax()))
        actual.append(float(lp[i-1, ids[i]]))
    out.append({"passage": j, "token_ids": ids, "argmax_id": argmax, "actual_logprob": actual})
    print(f"[quant-score] passage {j} ntok={len(ids)}", flush=True)
json.dump({"model": CKPT, "quant": os.environ["QLABEL"], "corpus": CORPUS, "passages": out},
          open(OUT,"w"))
print("DONE_TIER0_QUANT_DUMP", OUT, flush=True)
PY
  ' 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}; [ "$RC" = 0 ] || { echo "=== quant scoring FAILED rc=$RC ==="; exit "$RC"; }

# --- compare (host python3; common.py imports are lazy) ---
echo "=== tier0 compare: $QLABEL vs bf16 ==="
( cd "$REPO/evals/orchestrator" && python3 tier0_divergence.py compare "$REF" "$QDUMP" ) | tee -a "$LOG"
echo "=== pre-gate dumps: ref=$REF quant=$QDUMP ==="
