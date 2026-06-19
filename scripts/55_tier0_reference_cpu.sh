#!/usr/bin/env bash
# Compute the TRUE bf16 Tier-0 reference OFFLINE on the box CPU (the bf16 14B won't fit one B70 to
# serve under the v1 engine). Loads Qwen3-14B bf16 on CPU, scores the SAME corpus the eval orchestrator
# uses, and writes a tier0_tokens.json-compatible dump (per-token actual-logprob + argmax). No GPU ->
# safe to run alongside a live quant server. Pull the dump to the dev box and feed tier0_divergence.compare.
#
# Tokenization MUST match vLLM /tokenize (add_special_tokens=True, same tokenizer) so token_ids align
# position-for-position with the served quants' dumps. Index 0 (first token) = None, like tier0.
#
# Env: SRC (/specula_models/Qwen3-14B), CORPUS (box path, default /mnt/vm_8tb/b70/tier0_corpus.txt),
#      OUT (/mnt/vm_8tb/b70/results/tier0_ref_bf16_tokens.json), TOPK unused (argmax only).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SPECULA=/mnt/vm_8tb/specula-build/models
SRC="${SRC:-/specula_models/Qwen3-14B}"
QLABEL="${QLABEL:-bf16}"   # label written into the dump (also used to score NON-servable quants offline)
CORPUS="${CORPUS:-/mnt/vm_8tb/b70/tier0_corpus.txt}"
OUT="${OUT:-/mnt/vm_8tb/b70/results/tier0_ref_${QLABEL}_tokens.json}"
[ -f "$CORPUS" ] || { echo "MISSING corpus at $CORPUS (scp evals/prompts/tier0_corpus.txt b70:$CORPUS)"; exit 1; }
mkdir -p "$ROOT/results" "$ROOT/pip_cache"

docker run --rm --name tier0ref \
  -e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" --ipc=host --shm-size 16g \
  -v "$ROOT:$ROOT" -v "$SPECULA:/specula_models:ro" \
  -e HF_HOME=/hf_cache -e OMP_NUM_THREADS="${OMP:-32}" -e PIP_CACHE_DIR="$ROOT/pip_cache" \
  -e SRC="$SRC" -e CORPUS="$CORPUS" -e OUT="$OUT" -e QLABEL="$QLABEL" \
  python:3.11 bash -c '
    set -e
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
    pip install -q "transformers>=4.52" accelerate "compressed-tensors" 2>&1 | tail -1
    python - <<PY
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
SRC=os.environ["SRC"]; CORPUS=os.environ["CORPUS"]; OUT=os.environ["OUT"]
raw=open(CORPUS).read()
passages=[p.strip() for p in raw.split("\n---\n") if len(p.strip())>40]
print(f"[ref] {len(passages)} passages; loading {SRC} bf16 on CPU...", flush=True)
tok=AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True).eval()
out=[]
for j,text in enumerate(passages):
    ids=tok(text, add_special_tokens=True).input_ids
    t=torch.tensor([ids])
    with torch.no_grad():
        logits=model(t).logits[0].float()           # [seq, vocab]
    lp=torch.log_softmax(logits, dim=-1)
    argmax=[None]; actual=[None]                     # index 0 = first token, no prediction (matches tier0)
    for i in range(1, len(ids)):
        argmax.append(int(logits[i-1].argmax()))
        actual.append(float(lp[i-1, ids[i]]))
    out.append({"passage": j, "token_ids": ids, "argmax_id": argmax, "actual_logprob": actual})
    print(f"[ref] passage {j} ntok={len(ids)}", flush=True)
json.dump({"model": SRC, "quant": os.environ.get("QLABEL","bf16"), "corpus": CORPUS, "passages": out}, open(OUT,"w"))
print("DONE_TIER0_REF", OUT, flush=True)
PY
  '
echo "=== wrote $OUT ==="; ls -la "$OUT" 2>/dev/null
