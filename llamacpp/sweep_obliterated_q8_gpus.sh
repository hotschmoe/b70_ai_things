#!/usr/bin/env bash
# P1d: GPU_COUNT=2 vs 1 A/B on Pliny OBLITERATED Q8_0.
# Q8_DOORS=1 COMM_DIRECT=2 both. ignore-eos g128. Never COMM=3. No vLLM P2P.
# Always restore GPU_COUNT=2 unless STOP=1 (1x may OOM 29GB Q8).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
LOGDIR="${LOGDIR:-$ROOT/lmx_overnight}"
mkdir -p "$LOGDIR"
NAME="${NAME:-qwen38_oblit_q8}"
PORT="${PORT:-8010}"
STATUS="$LOGDIR/STATUS"
SUMMARY="$LOGDIR/p1d_gpus_${STAMP}.json"
LOCK="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"

say(){ printf '%s\n' "$*"; }
set_status(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H%MZ)" "$*" | tee -a "$STATUS"; }

lease_held() {
  [ -s "$LOCK.0.owner" ] || [ -s "$LOCK.1.owner" ]
}
ours_up() {
  docker ps --format '{{.Names}}' | grep -qx "$NAME"
}
foreign_holder() {
  ours_up && return 1
  [ -n "${B70_INSIDE_GPU_RUN:-}" ] && return 1
  lease_held || return 1
  return 0
}
stop_ours() {
  docker stop -t 20 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}

start_gpus() {
  local g="$1"
  set_status "P1D_STATUS=START gpu=$g doors=1 comm=2"
  MODEL_SHA256= Q8_DOORS=1 GGML_SYCL_COMM_DIRECT_Q8=2 \
    GPU_COUNT="$g" PORT="$PORT" NAME="$NAME" \
    bash "$REPO/llamacpp/serve_qwen38_obliterated_q8.sh" start
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    set_status "P1D_STATUS=SERVE_FAIL gpu=$g rc=$rc"
    docker logs "$NAME" 2>&1 | tail -80 >"$LOGDIR/p1d_serve_tail_g${g}_${STAMP}.log" || true
    return 4
  fi
  set_status "P1D_STATUS=SERVE_OK gpu=$g"
  return 0
}

run_g1() {
  local g="$1"
  local g1_log="$LOGDIR/p1d_g1_g${g}_${STAMP}.log"
  python3 - "$PORT" "$g1_log" <<'PY'
import json, sys, urllib.request
port, logp = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
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
lines = [f"model={model} gpus_g1"]
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
  local g1=$?
  set_status "P1D_STATUS=G1 gpu=$g rc=$g1 log=$g1_log"
  return "$g1"
}

run_bench() {
  local g="$1"
  local out="$LOGDIR/p1d_phase_g${g}_${STAMP}.json"
  local model
  model="$(python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models')); print((d.get('data') or [{}])[0].get('id') or 'qwen')")"
  python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
    --base "http://127.0.0.1:${PORT}" \
    --model "$model" \
    --prompt-tokens 512 --gen-tokens 128 --n 5 \
    --ignore-eos \
    --label "q8gpu${g}" \
    --out "$out"
  local brc=$?
  set_status "P1D_STATUS=BENCH gpu=$g rc=$brc out=$out"
  return "$brc"
}

restore_2x() {
  if ours_up; then
    # already 2x leftover from last arm
    return 0
  fi
  start_gpus 2 || true
}

cd "$REPO"
if [ -z "${B70_INSIDE_GPU_RUN:-}" ]; then
  if foreign_holder; then
    set_status "P1D_STATUS=LEASE_BUSY"
    ./bin/gpu-run --status || true
    docker ps --format '{{.Names}} {{.Status}}' || true
    echo "foreign GPU holder -- do not start a second serve"
    exit 8
  fi
  if ours_up || lease_held; then
    say "stop leftover $NAME so P1d can hold the lease"
    stop_ours
    sleep 2
  fi
  set_status "P1D_STATUS=EXEC_GPU_RUN stamp=$STAMP"
  exec env B70_GPU_LOCK_TIMEOUT=30 B70_INSIDE_GPU_RUN=1 STAMP="$STAMP" \
    ./bin/gpu-run bash "$REPO/llamacpp/sweep_obliterated_q8_gpus.sh"
fi

# Order: 2 (P1c match) then 1. Always restore 2x unless STOP=1.
rc_all=0
out2="$LOGDIR/p1d_phase_g2_${STAMP}.json"
out1="$LOGDIR/p1d_phase_g1_${STAMP}.json"
for g in 2 1; do
  if ours_up; then
    say "stop $NAME for gpu=$g"
    stop_ours
    sleep 2
  fi
  if ! start_gpus "$g"; then
    rc_all=4
    break
  fi
  run_g1 "$g" || { rc_all=5; break; }
  run_bench "$g" || { rc_all=$?; break; }
done

python3 - "$SUMMARY" "$out2" "$out1" <<'PY'
import json, sys
outp, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p):
    if not p:
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        return {"error": str(e), "path": p}
d2, d1 = load(a), load(b)
def med(d):
    if not d or "median_post_first_tok_s" not in d:
        return None
    return d.get("median_post_first_tok_s")
summary = {
    "gpu2": d2,
    "gpu1": d1,
    "median_gpu2": med(d2),
    "median_gpu1": med(d1),
    "paths": {"gpu2": a, "gpu1": b},
    "vs_q4km_43.8": {
        "gpu2": (med(d2) / 43.8) if med(d2) else None,
        "gpu1": (med(d1) / 43.8) if med(d1) else None,
    },
    "note": "Q8_DOORS=1 COMM_DIRECT=2. ignore_eos g128. Q8_0 weight-only. 1x may OOM.",
}
json.dump(summary, open(outp, "w"), indent=2, sort_keys=True)
print(json.dumps({k: summary[k] for k in ("median_gpu2", "median_gpu1", "vs_q4km_43.8", "paths")}, indent=2))
PY

if [ "${STOP:-0}" = "1" ]; then
  stop_ours
  set_status "P1D_STATUS=STOPPED summary=$SUMMARY rc=$rc_all"
else
  # last arm may be 1x; always put 2x back for the next fire
  stop_ours
  sleep 2
  restore_2x
  set_status "P1D_STATUS=DONE summary=$SUMMARY rc=$rc_all serve_left_up=$(ours_up && echo 1 || echo 0)"
  if ours_up; then
    set_status "P1D_STATUS=HOLD docker-wait $NAME"
    docker wait "$NAME" >/dev/null 2>&1 || true
  fi
fi
exit "$rc_all"
