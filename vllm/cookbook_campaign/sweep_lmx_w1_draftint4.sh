#!/usr/bin/env bash
# LMX overnight W1: 1x B70 Qwen3.8-27B GPTQ-Int4 MTP4 + cookbook 2026.08.19
# draft-INT4 (LM head + 5 MTP linears) + mixed-split v5 on digest f01e24f6.
# Compare to S1 47.58 post-first (same ckpt, no draft-INT4). Do not demote W8A8.
# Do not start DD. P2P off. One card.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
LOGDIR="${LOGDIR:-$ROOT/lmx_overnight}"
mkdir -p "$LOGDIR"
NAME="${NAME:-lmx_w1_d38}"
PORT="${PORT:-18080}"
CARD="${CARD:-0}"
SERVED="${SERVED:-qwen3.8-27b-GPTQ-Int4-mtp4-draftint4}"
STATUS="$LOGDIR/STATUS"
G1_LOG="$LOGDIR/w1_g1_${STAMP}.log"
BENCH_OUT="$LOGDIR/w1_phase_${STAMP}.json"
SERVE_LOG="$LOGDIR/w1_serve_${STAMP}.log"

say() { printf '%s\n' "$*"; }
set_status() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H%MZ)" "$*" | tee -a "$STATUS"; }

cd "$REPO"
if ./bin/gpu-run --status 2>/dev/null | grep -q "card ${CARD}: busy"; then
  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    set_status "W1_STATUS=ATTACH name=$NAME"
  else
    set_status "W1_STATUS=LEASE_BUSY"
    ./bin/gpu-run --status || true
    exit 8
  fi
else
  set_status "W1_STATUS=START"
  B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card "$CARD" bash -c "
    set -uo pipefail
    ./bin/xpu-health --card $CARD --img vllm-xpu-env:int8g-v0260 --timeout 90
    hc=\$?
    echo xpu-health rc=\$hc
    [ \"\$hc\" -eq 0 ] || exit 10
    NAME=$NAME SERVED=$SERVED DRAFT_INT4=1 MAXSEQS=8 MAXLEN=131072 \
      bash $REPO/vllm/cookbook_campaign/launch.sh dense38-gptq mtp4 off $PORT $CARD
    TIMEOUT=1200 bash $REPO/vllm/cookbook_campaign/wait_healthy.sh $PORT $NAME
  " >"$SERVE_LOG" 2>&1 || {
    set_status "W1_STATUS=SERVE_FAIL"
    docker logs "$NAME" 2>&1 | tail -100 >"$LOGDIR/w1_serve_tail_${STAMP}.log" || true
    exit 4
  }
  # Hold the card while the container lives.
  B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card "$CARD" bash -c "docker wait $NAME; echo DOCKER_WAIT_DONE" \
    >"$LOGDIR/w1_wait_${STAMP}.log" 2>&1 &
  echo $! >"$LOGDIR/w1_wait.pid"
  set_status "W1_STATUS=SERVE_OK wait_pid=$(cat "$LOGDIR/w1_wait.pid")"
fi

python3 - "$PORT" "$SERVED" "$G1_LOG" <<'PY'
import json, sys, urllib.request
port, model, logp = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
prompts = [
    ("paris", "What is the capital of France? Answer in one short sentence.",
     lambda t: "paris" in t.lower()),
    ("mul", "What is 17*23? Answer with just the number.",
     lambda t: "391" in t),
]
ok = True
lines = []
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
set_status "W1_STATUS=G1 rc=$g1 log=$G1_LOG"
[ "$g1" -eq 0 ] || exit 5

python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
  --base "http://127.0.0.1:${PORT}" \
  --model "$SERVED" \
  --prompt-tokens 512 --gen-tokens 128 --n 5 \
  --out "$BENCH_OUT"
brc=$?
set_status "W1_STATUS=BENCH rc=$brc out=$BENCH_OUT"
# Keep the serve up for the next 30m fire unless caller sets STOP=1.
if [ "${STOP:-0}" = 1 ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  set_status "W1_STATUS=STOPPED"
fi
exit "$brc"
