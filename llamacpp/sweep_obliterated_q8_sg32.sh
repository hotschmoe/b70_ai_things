#!/usr/bin/env bash
# P1e: GGML_SYCL_MMVQ_SG32=1 vs 0 on Pliny Q8_0. In-image door only.
# Q8_DOORS=1 COMM=2 GPU=2. QUAD_SG24 env set but 0xSero JIT has no symbol.
# Never COMM=3, never FATTN_MMA=1. Always restore SG32=0 unless STOP=1.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
LOGDIR="${LOGDIR:-$ROOT/lmx_overnight}"
mkdir -p "$LOGDIR"
NAME="${NAME:-qwen38_oblit_q8}"
PORT="${PORT:-8010}"
STATUS="$LOGDIR/STATUS"
SUMMARY="$LOGDIR/p1e_sg32_${STAMP}.json"
LOCK="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"

say(){ printf '%s\n' "$*"; }
set_status(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H%MZ)" "$*" | tee -a "$STATUS"; }
lease_held() { [ -s "$LOCK.0.owner" ] || [ -s "$LOCK.1.owner" ]; }
ours_up() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }
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

start_sg() {
  local s="$1"
  set_status "P1E_STATUS=START sg32=$s doors=1 comm=2 gpu=2"
  MODEL_SHA256= Q8_DOORS=1 GGML_SYCL_COMM_DIRECT_Q8=2 GGML_SYCL_MMVQ_SG32="$s" \
    GPU_COUNT=2 PORT="$PORT" NAME="$NAME" \
    bash "$REPO/llamacpp/serve_qwen38_obliterated_q8.sh" start
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    set_status "P1E_STATUS=SERVE_FAIL sg32=$s rc=$rc"
    docker logs "$NAME" 2>&1 | tail -80 >"$LOGDIR/p1e_serve_tail_s${s}_${STAMP}.log" || true
    return 4
  fi
  set_status "P1E_STATUS=SERVE_OK sg32=$s"
  docker logs "$NAME" 2>&1 | grep -E 'mmvq_sg32|QUAD_SG|entrypoint' | tail -8 || true
  return 0
}

run_g1() {
  local s="$1"
  local g1_log="$LOGDIR/p1e_g1_s${s}_${STAMP}.log"
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
lines = [f"model={model} sg32_g1"]
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
  set_status "P1E_STATUS=G1 sg32=$s rc=$g1 log=$g1_log"
  return "$g1"
}

run_bench() {
  local s="$1"
  local out="$LOGDIR/p1e_phase_s${s}_${STAMP}.json"
  local model
  model="$(python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models')); print((d.get('data') or [{}])[0].get('id') or 'qwen')")"
  python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
    --base "http://127.0.0.1:${PORT}" \
    --model "$model" \
    --prompt-tokens 512 --gen-tokens 128 --n 5 \
    --ignore-eos \
    --label "q8sg32${s}" \
    --out "$out"
  local brc=$?
  set_status "P1E_STATUS=BENCH sg32=$s rc=$brc out=$out"
  return "$brc"
}

cd "$REPO"
if [ -z "${B70_INSIDE_GPU_RUN:-}" ]; then
  if foreign_holder; then
    set_status "P1E_STATUS=LEASE_BUSY"
    ./bin/gpu-run --status || true
    exit 8
  fi
  if ours_up || lease_held; then
    say "stop leftover $NAME so P1e can hold the lease"
    stop_ours
    sleep 2
  fi
  set_status "P1E_STATUS=EXEC_GPU_RUN stamp=$STAMP"
  exec env B70_GPU_LOCK_TIMEOUT=30 B70_INSIDE_GPU_RUN=1 STAMP="$STAMP" \
    ./bin/gpu-run bash "$REPO/llamacpp/sweep_obliterated_q8_sg32.sh"
fi

# SG32=1 first (unknown), restore 0 (P1d hold).
rc_all=0
out1="$LOGDIR/p1e_phase_s1_${STAMP}.json"
if ours_up; then stop_ours; sleep 2; fi
if start_sg 1; then
  run_g1 1 || rc_all=5
  [ "$rc_all" -eq 0 ] && run_bench 1 || rc_all=$?
else
  rc_all=4
fi

python3 - "$SUMMARY" "$out1" <<'PY'
import json, sys
outp, a = sys.argv[1], sys.argv[2]
def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"error": str(e), "path": p}
d1 = load(a)
med = d1.get("median_post_first_tok_s") if isinstance(d1, dict) else None
summary = {
    "sg32_1": d1,
    "median_sg32_1": med,
    "vs_hold_32.03": (med / 32.03) if med else None,
    "vs_q4km_43.8": (med / 43.8) if med else None,
    "note": "in-image SG32 door. DP4A2/QUAD_SG24 not in 0xSero JIT .so 258f4729.",
}
json.dump(summary, open(outp, "w"), indent=2, sort_keys=True)
print(json.dumps({k: summary[k] for k in ("median_sg32_1", "vs_hold_32.03", "vs_q4km_43.8")}, indent=2))
PY

if [ "${STOP:-0}" = "1" ]; then
  stop_ours
  set_status "P1E_STATUS=STOPPED summary=$SUMMARY rc=$rc_all"
else
  stop_ours
  sleep 2
  start_sg 0 || true
  set_status "P1E_STATUS=DONE summary=$SUMMARY rc=$rc_all serve_left_up=$(ours_up && echo 1 || echo 0)"
  if ours_up; then
    set_status "P1E_STATUS=HOLD docker-wait $NAME"
    docker wait "$NAME" >/dev/null 2>&1 || true
  fi
fi
exit "$rc_all"
