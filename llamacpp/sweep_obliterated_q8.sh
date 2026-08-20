#!/usr/bin/env bash
# P1: Pliny OBLITERATED Qwen3.8-27B Q8_0 -- G1 + post-first p512/g128 n=5.
# Requires GGUF on disk. 2x B70 default. Do not start DD. No vLLM P2P.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
LOGDIR="${LOGDIR:-$ROOT/lmx_overnight}"
mkdir -p "$LOGDIR"
NAME="${NAME:-qwen38_oblit_q8}"
PORT="${PORT:-8010}"
GPU_COUNT="${GPU_COUNT:-2}"
GGUF="$REPO/models/files/qwen3.8-27b/obliterated-q8/Qwen3.8-27B-OBLITERATED-Q8_0.gguf"
STATUS="$LOGDIR/STATUS"
G1_LOG="$LOGDIR/p1_g1_${STAMP}.log"
BENCH_OUT="$LOGDIR/p1_phase_${STAMP}.json"

say(){ printf '%s\n' "$*"; }
set_status(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H%MZ)" "$*" | tee -a "$STATUS"; }

cd "$REPO"
if [ ! -s "$GGUF" ]; then
  sz=$(du -sm "$(dirname "$GGUF")" 2>/dev/null | awk '{print $1}')
  set_status "P1_STATUS=WAIT_FETCH dir_mb=${sz:-0}"
  echo "GGUF missing: $GGUF"
  exit 9
fi
sz=$(du -sm "$GGUF" | awk '{print $1}')
if [ "$sz" -lt 25000 ]; then
  set_status "P1_STATUS=PARTIAL_FETCH mb=$sz"
  echo "GGUF incomplete (${sz} MB), want ~29000"
  exit 9
fi

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  set_status "P1_STATUS=ATTACH name=$NAME"
else
  if ./bin/gpu-run --status 2>/dev/null | grep -Eq 'card 0: (HELD|busy)'; then
    set_status "P1_STATUS=LEASE_BUSY"
    ./bin/gpu-run --status || true
    exit 8
  fi
  set_status "P1_STATUS=START gpu=$GPU_COUNT"
  if [ "$GPU_COUNT" = "1" ]; then
    B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card 0 bash "$REPO/llamacpp/serve_qwen38_obliterated_q8.sh" start
  else
    B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run bash "$REPO/llamacpp/serve_qwen38_obliterated_q8.sh" start
  fi
  rc=$?
  [ "$rc" -eq 0 ] || { set_status "P1_STATUS=SERVE_FAIL rc=$rc"; exit 4; }
  B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run bash -c "docker wait $NAME; echo DOCKER_WAIT_DONE" \
    >"$LOGDIR/p1_wait_${STAMP}.log" 2>&1 &
  echo $! >"$LOGDIR/p1_wait.pid"
  set_status "P1_STATUS=SERVE_OK wait_pid=$(cat "$LOGDIR/p1_wait.pid")"
fi

python3 - "$PORT" "$G1_LOG" <<'PY'
import json, sys, urllib.request
port, logp = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
# discover served id
model = "Qwen3.8-27B-OBLITERATED-Q8_0"
try:
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        ids = [m.get("id") for m in json.loads(r.read()).get("data") or [] if m.get("id")]
    if ids:
        model = ids[0]
except Exception:
    pass
prompts = [
    ("paris", "What is the capital of France? Answer in one short sentence.",
     lambda t: "paris" in t.lower()),
    ("mul", "What is 17*23? Answer with just the number.",
     lambda t: "391" in t),
]
ok = True
lines = [f"model={model}"]
for name, content, pred in prompts:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = json.loads(r.read())
        text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        passed = bool(pred(text))
    except Exception as e:
        text = f"ERR {e}"
        passed = False
    ok = ok and passed
    lines.append(f"{name} {'PASS' if passed else 'FAIL'}: {text.replace(chr(10),' ')[:240]}")
open(logp, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if ok else 1)
PY
g1=$?
set_status "P1_STATUS=G1 rc=$g1 log=$G1_LOG"
[ "$g1" -eq 0 ] || exit 5

python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
  --base "http://127.0.0.1:${PORT}" \
  --model "$(python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models')); print((d.get('data') or [{}])[0].get('id') or 'qwen')")" \
  --prompt-tokens 512 --gen-tokens 128 --n 5 \
  --out "$BENCH_OUT"
brc=$?
set_status "P1_STATUS=BENCH rc=$brc out=$BENCH_OUT"
exit "$brc"
