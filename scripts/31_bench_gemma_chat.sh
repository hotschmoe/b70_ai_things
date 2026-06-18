#!/usr/bin/env bash
# Proper benchmark for the instruct Gemma 4 server via /v1/chat/completions (streaming).
# Measures TTFT + decode t/s over a few runs; also a long-prompt prefill proxy.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
NAME=vllm_gemma4; PORT=18080; MODEL=gemma4
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT="$ROOT/results/gemma4_fp8_${STAMP}.txt"

echo "=== /v1/models ===" | tee "$OUT"
curl -s "http://localhost:${PORT}/v1/models" | head -c 400 | tee -a "$OUT"; echo | tee -a "$OUT"

docker exec -i -e PORT="$PORT" -e MODEL="$MODEL" "$NAME" python - <<'PY' 2>&1 | tee -a "$OUT"
import os, time, json, urllib.request
PORT=os.environ["PORT"]; MODEL=os.environ["MODEL"]
URL=f"http://localhost:{PORT}/v1/completions"
def gemma(msg): return f"<start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n"
def run(msg, n, label):
    body=json.dumps({"model":MODEL,"prompt":gemma(msg),
                     "max_tokens":n,"temperature":0,"stream":True,
                     "stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    t0=time.time(); ttft=None; toks=0; ptoks=None
    try:
        with urllib.request.urlopen(req) as r:
            for raw in r:
                line=raw.decode("utf-8","ignore").strip()
                if not line.startswith("data:"): continue
                d=line[5:].strip()
                if d=="[DONE]": break
                o=json.loads(d)
                ch=o.get("choices") or []
                if ch and ch[0].get("text"):
                    if ttft is None: ttft=time.time()-t0
                    toks+=1
                if o.get("usage"): ptoks=o["usage"].get("prompt_tokens")
    except Exception as e:
        print(f"[{label}] ERROR {e}"); return
    dt=time.time()-t0
    dec=(toks-1)/(dt-ttft) if (ttft and toks>1 and dt>ttft) else float("nan")
    print(f"[{label}] prompt_toks={ptoks} gen={toks} TTFT={ (ttft*1000 if ttft else -1):.1f}ms total={dt:.2f}s decode={dec:.2f} tok/s")
SHORT="Explain what a GPU is in one sentence."
LONG="Summarize this in one sentence: "+("The quick brown fox jumps over the lazy dog. "*150)
print("== warmup =="); run(SHORT,16,"warmup")
print("== decode (short prompt, 128 tok) =="); [run(SHORT,128,f"decode{i}") for i in range(3)]
print("== prefill proxy (~1.5k-token prompt, 8 tok; low TTFT=fast prefill) =="); [run(LONG,8,f"prefill{i}") for i in range(2)]
PY
echo "=== saved $OUT ==="
