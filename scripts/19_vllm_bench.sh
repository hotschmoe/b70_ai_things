#!/usr/bin/env bash
# Benchmark a running llm-scaler vLLM server (default name vllm_fp8, port 18080).
# Streaming client measures TTFT and decode t/s; a long-prompt run estimates prefill.
# Captures xpu-smi during decode. Args: [container] [port] [served_model_name]
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
NAME="${1:-vllm_gemma4}"; PORT="${2:-18080}"; MODEL="${3:-gemma4}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/results/vllm_${MODEL}_${STAMP}.txt"

if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "Server not healthy on :${PORT}"; docker logs "$NAME" 2>&1 | tail -30; exit 1
fi

# xpu-smi during decode (background sampler)
( for i in $(seq 1 12); do docker exec "$NAME" xpu-smi dump -d 0 -m 0,1,2,18 -n 1 2>/dev/null | tail -1; sleep 1; done ) > "$ROOT/results/xpusmi_${MODEL}_${STAMP}.txt" 2>/dev/null &

# Python streaming client inside the container (has openai/requests + python).
docker exec -e PORT="$PORT" -e MODEL="$MODEL" "$NAME" python - <<'PY' 2>&1 | tee "$OUT"
import os, time, json, urllib.request
PORT=os.environ["PORT"]; MODEL=os.environ["MODEL"]
URL=f"http://localhost:{PORT}/v1/completions"
def run(prompt, n, label):
    body=json.dumps({"model":MODEL,"prompt":prompt,"max_tokens":n,"temperature":0,
                     "stream":True,"stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    t0=time.time(); ttft=None; toks=0; prompt_toks=None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line=raw.decode().strip()
            if not line.startswith("data:"): continue
            data=line[5:].strip()
            if data=="[DONE]": break
            obj=json.loads(data)
            ch=obj.get("choices") or [{}]
            if ch and ch[0].get("text"):
                if ttft is None: ttft=time.time()-t0
                toks+=1
            if obj.get("usage"): prompt_toks=obj["usage"].get("prompt_tokens")
    dt=time.time()-t0
    dec=(toks-1)/(dt-ttft) if ttft and toks>1 and dt>ttft else float("nan")
    print(f"[{label}] prompt_toks={prompt_toks} gen_toks={toks} TTFT={ttft*1000:.1f}ms "
          f"total={dt:.2f}s decode={dec:.2f} tok/s")
    return ttft,dec
SHORT="Explain what a GPU is in one sentence."
LONG=("Summarize the following. "+("The quick brown fox jumps over the lazy dog. "*120))
print("=== warmup ==="); run(SHORT,16,"warmup")
print("=== decode (short prompt, 128 tok) ===")
for i in range(3): run(SHORT,128,f"decode{i}")
print("=== prefill proxy (long ~1k prompt, 8 tok) -> low TTFT means fast prefill ===")
for i in range(2): run(LONG,8,f"prefill{i}")
PY

wait 2>/dev/null
echo "=== xpu-smi samples (mem/util/power during decode) ===" | tee -a "$OUT"
cat "$ROOT/results/xpusmi_${MODEL}_${STAMP}.txt" 2>/dev/null | tee -a "$OUT"
echo "=== saved $OUT ==="
