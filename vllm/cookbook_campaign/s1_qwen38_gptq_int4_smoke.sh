#!/usr/bin/env bash
# S1: 1-card smoke of SergiioB 3.8 GPTQ-Int4 + MTP4 on digest f01e24f6.
# Fetch image+weights first (AGASYNC stays up). Then card-0 serve, G1,
# one phase_bench p512/g128, stop S1, restore W8A8 DSpark AGASYNC.
# Do not start DD. Do not enter Phase 2. Do not demote W8A8.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
LOGDIR="${LOGDIR:-$ROOT/qwen38-w8a8-dspark}"
mkdir -p "$LOGDIR"

CKPT="$REPO/models/files/community/qwen38-27b-gptq-mtp-preserved"
HF_REPO="SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
HF_REV="9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e"
IMAGE="vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
S1_NAME="${S1_NAME:-qwen38_s1_gptq}"
S1_PORT="${S1_PORT:-18080}"
S1_CARD="${S1_CARD:-0}"
SERVED="${SERVED:-qwen3.8-27b-GPTQ-Int4-mtp4}"
AG_NAME="${AG_NAME:-qwen38_w8a8_dspark}"
STATUS="$LOGDIR/loop18_s1.status"
G1_LOG="$LOGDIR/loop18_g1.log"
BENCH_OUT="$LOGDIR/loop18_phase_bench.json"

say() { printf '%s\n' "$*"; }
stamp() { date -u +%Y-%m-%dT%H%MZ; }
set_status() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$STATUS"; }

set_status "S1_STATUS=START"

need_ckpt() {
  if [ ! -f "$CKPT/config.json" ] || [ ! -f "$CKPT/model.safetensors.index.json" ]; then
    return 0
  fi
  sz=$(du -sm "$CKPT" | awk '{print $1}')
  [ "$sz" -lt 15000 ]
}

need_image() {
  ! docker image inspect "$IMAGE" >/dev/null 2>&1
}

fetch_ckpt() {
  if ! need_ckpt; then
    say "ckpt already present: $CKPT ($(du -sh "$CKPT" | awk '{print $1}'))"
    return 0
  fi
  say "hf download $HF_REPO rev=$HF_REV -> $CKPT"
  mkdir -p "$CKPT"
  hf download "$HF_REPO" --revision "$HF_REV" --local-dir "$CKPT"
}

pull_image() {
  if ! need_image; then
    say "image already present: $IMAGE"
    return 0
  fi
  say "docker pull $IMAGE"
  docker pull "$IMAGE"
}

set_status "S1_STATUS=FETCH"
fetch_rc=0
pull_rc=0
fetch_ckpt >"$LOGDIR/loop18_hf.log" 2>&1 &
hf_pid=$!
pull_image >"$LOGDIR/loop18_pull.log" 2>&1 &
pull_pid=$!
wait "$hf_pid" || fetch_rc=$?
wait "$pull_pid" || pull_rc=$?
if [ "$fetch_rc" -ne 0 ] || need_ckpt; then
  set_status "S1_STATUS=FETCH_FAIL fetch_rc=$fetch_rc"
  say "FETCH_FAIL: see $LOGDIR/loop18_hf.log"
  exit 2
fi
if [ "$pull_rc" -ne 0 ] || need_image; then
  set_status "S1_STATUS=PULL_FAIL pull_rc=$pull_rc"
  say "PULL_FAIL: see $LOGDIR/loop18_pull.log"
  exit 3
fi
set_status "S1_STATUS=FETCH_OK"

g1_probe() {
  python3 - "$S1_PORT" "$SERVED" "$G1_LOG" <<'PY'
import json, sys, urllib.request
port, model, logp = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
prompts = [
    ("paris", "What is the capital of France? Answer in one short sentence.",
     lambda t: "paris" in t.lower()),
    ("mul", "What is 17*23? Answer with just the number.",
     lambda t: "391" in t),
    ("fib",
     "Write a Python function that returns the n-th Fibonacci number using iteration, not recursion. Only the function.",
     lambda t: (("for " in t or "while " in t or "a, b" in t or "a,b" in t)
                and "def " in t)),
]
ok = True
lines = []
for name, content, pred in prompts:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 128,
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
    preview = text.replace("\n", "\\n")[:240]
    lines.append(f"{name} {'PASS' if passed else 'FAIL'}: {preview}")
open(logp, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if ok else 1)
PY
}

restore_agasync() {
  say "restore AGASYNC W8A8 DSpark"
  docker rm -f "$S1_NAME" >/dev/null 2>&1 || true
  NAME="$AG_NAME" bash "$REPO/vllm/dflash/serve_qwen38_w8a8_dspark.sh" stop || true
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 \
    --img vllm-xpu-env:int8g-v0260 --timeout 90 \
    >"$LOGDIR/loop18_restore_health.log" 2>&1 || true
  B70_EXTRA_ENV=PUSH_AR_ALLGATHER_ASYNC=1 \
    W8A16_M_MAX=0 GRAPH=1 SPECTOK=4 MAXLEN=122880 \
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-agasync \
    PORT=18080 NAME="$AG_NAME" \
    ./bin/gpu-run bash "$REPO/vllm/dflash/serve_qwen38_w8a8_dspark.sh" start \
    >"$LOGDIR/loop18_restore.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    nohup ./bin/gpu-run bash -c "docker wait $AG_NAME; echo DOCKER_WAIT_DONE" \
      >"$LOGDIR/loop18_restore.wait.log" 2>&1 &
    echo $! >"$LOGDIR/loop18_restore.pid"
    set_status "S1_STATUS=RESTORED ag_wait_pid=$(cat "$LOGDIR/loop18_restore.pid")"
  else
    set_status "S1_STATUS=RESTORE_FAIL rc=$rc"
  fi
  return "$rc"
}

# GPU phase: stop TP=2 only after artifacts exist.
set_status "S1_STATUS=GPU"
cd "$REPO"
NAME="$AG_NAME" bash "$REPO/vllm/dflash/serve_qwen38_w8a8_dspark.sh" stop \
  >"$LOGDIR/loop18_ag_stop.log" 2>&1 || true
# Wait for previous docker-wait lease holder to drop both cards.
for i in $(seq 1 30); do
  if ./bin/gpu-run --status 2>/dev/null | grep -q 'card 0: free'; then
    break
  fi
  sleep 2
done

s1_rc=0
B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card 0 bash -c '
  set -uo pipefail
  REPO="'"$REPO"'"
  LOGDIR="'"$LOGDIR"'"
  S1_NAME="'"$S1_NAME"'"
  S1_PORT="'"$S1_PORT"'"
  S1_CARD="'"$S1_CARD"'"
  SERVED="'"$SERVED"'"
  ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260 --timeout 90
  hc=$?
  echo "xpu-health rc=$hc"
  [ "$hc" -eq 0 ] || exit 10
  NAME="$S1_NAME" MAXSEQS=8 MAXLEN=131072 \
    bash "$REPO/vllm/cookbook_campaign/launch.sh" \
      dense38-gptq mtp4 off "$S1_PORT" "$S1_CARD"
  TIMEOUT=1200 bash "$REPO/vllm/cookbook_campaign/wait_healthy.sh" "$S1_PORT" "$S1_NAME"
' >"$LOGDIR/loop18_serve.log" 2>&1 || s1_rc=$?

if [ "$s1_rc" -ne 0 ]; then
  set_status "S1_STATUS=SERVE_FAIL rc=$s1_rc"
  docker logs "$S1_NAME" 2>&1 | tail -80 >"$LOGDIR/loop18_serve_tail.log" || true
  restore_agasync || true
  exit 4
fi
set_status "S1_STATUS=SERVE_OK"

# Hold card 0 while we probe/bench (launch returned; container is -d).
hold_pid=""
B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card 0 bash -c "docker wait $S1_NAME; echo DOCKER_WAIT_DONE" \
  >"$LOGDIR/loop18_s1.wait.log" 2>&1 &
hold_pid=$!
echo "$hold_pid" >"$LOGDIR/loop18_s1.wait.pid"

g1_rc=0
g1_probe || g1_rc=$?
if [ "$g1_rc" -ne 0 ]; then
  set_status "S1_STATUS=G1_FAIL"
  say "G1 fail-closed: no published speed. see $G1_LOG"
  docker rm -f "$S1_NAME" >/dev/null 2>&1 || true
  wait "$hold_pid" 2>/dev/null || true
  restore_agasync || true
  exit 5
fi
set_status "S1_STATUS=G1_OK"

bench_rc=0
python3 -u "$REPO/vllm/cookbook_campaign/phase_bench.py" \
  --base "http://127.0.0.1:${S1_PORT}" \
  --model "$SERVED" \
  --prompt-tokens 512 --gen-tokens 128 --n 5 \
  --label loop18-s1-p512-g128 \
  --out "$BENCH_OUT" \
  >"$LOGDIR/loop18_phase_bench.log" 2>&1 || bench_rc=$?

if [ "$bench_rc" -ne 0 ]; then
  set_status "S1_STATUS=BENCH_FAIL rc=$bench_rc"
else
  set_status "S1_STATUS=BENCH_OK"
fi

docker rm -f "$S1_NAME" >/dev/null 2>&1 || true
wait "$hold_pid" 2>/dev/null || true

restore_agasync || true
if [ "$bench_rc" -ne 0 ]; then
  exit 6
fi
set_status "S1_STATUS=DONE"
exit 0
