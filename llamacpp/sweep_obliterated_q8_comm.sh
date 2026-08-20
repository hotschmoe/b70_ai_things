#!/usr/bin/env bash
# P1c: COMM_DIRECT_Q8=2 vs 0 A/B on Pliny OBLITERATED Q8_0.
# Q8_DOORS=1 both arms. ignore-eos g128. Never COMM_DIRECT=3 (DEVICE_LOST).
# 2x B70 llama.cpp SYCL. No vLLM P2P. Do not start DD.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%MZ)}"
LOGDIR="${LOGDIR:-$ROOT/lmx_overnight}"
mkdir -p "$LOGDIR"
NAME="${NAME:-qwen38_oblit_q8}"
PORT="${PORT:-8010}"
GPU_COUNT="${GPU_COUNT:-2}"
STATUS="$LOGDIR/STATUS"
SUMMARY="$LOGDIR/p1c_comm_${STAMP}.json"
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

start_comm() {
  local c="$1"
  if [ "$c" = "3" ]; then
    set_status "P1C_STATUS=REFUSED comm=3 DEVICE_LOST"
    return 4
  fi
  set_status "P1C_STATUS=START comm=$c doors=1 gpu=$GPU_COUNT"
  MODEL_SHA256= Q8_DOORS=1 GGML_SYCL_COMM_DIRECT_Q8="$c" \
    GPU_COUNT="$GPU_COUNT" PORT="$PORT" NAME="$NAME" \
    bash "$REPO/llamacpp/serve_qwen38_obliterated_q8.sh" start
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    set_status "P1C_STATUS=SERVE_FAIL comm=$c rc=$rc"
    docker logs "$NAME" 2>&1 | tail -80 >"$LOGDIR/p1c_serve_tail_c${c}_${STAMP}.log" || true
    return 4
  fi
  set_status "P1C_STATUS=SERVE_OK comm=$c"
  return 0
}

run_g1() {
  local c="$1"
  local g1_log="$LOGDIR/p1c_g1_c${c}_${STAMP}.log"
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
lines = [f"model={model} comm_g1"]
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
  set_status "P1C_STATUS=G1 comm=$c rc=$g1 log=$g1_log"
  return "$g1"
}

run_bench() {
  local c="$1"
  local out="$LOGDIR/p1c_phase_c${c}_${STAMP}.json"
  local model
  model="$(python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models')); print((d.get('data') or [{}])[0].get('id') or 'qwen')")"
  python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
    --base "http://127.0.0.1:${PORT}" \
    --model "$model" \
    --prompt-tokens 512 --gen-tokens 128 --n 5 \
    --ignore-eos \
    --label "q8comm${c}" \
    --out "$out"
  local brc=$?
  set_status "P1C_STATUS=BENCH comm=$c rc=$brc out=$out"
  return "$brc"
}

cd "$REPO"
if [ -z "${B70_INSIDE_GPU_RUN:-}" ]; then
  if foreign_holder; then
    set_status "P1C_STATUS=LEASE_BUSY"
    ./bin/gpu-run --status || true
    docker ps --format '{{.Names}} {{.Status}}' || true
    echo "foreign GPU holder -- do not start a second serve"
    exit 8
  fi
  # Bounce our leftover docker-wait holder so this sweep can take the lease.
  if ours_up || lease_held; then
    say "stop leftover $NAME so P1c can hold the lease"
    stop_ours
    sleep 2
  fi
  set_status "P1C_STATUS=EXEC_GPU_RUN stamp=$STAMP"
  exec env B70_GPU_LOCK_TIMEOUT=30 B70_INSIDE_GPU_RUN=1 STAMP="$STAMP" \
    ./bin/gpu-run bash "$REPO/llamacpp/sweep_obliterated_q8_comm.sh"
fi

# Order: 2 (LOOP 6 match) then 0. Leave 2 up unless STOP=1. Never 3.
rc_all=0
out2="$LOGDIR/p1c_phase_c2_${STAMP}.json"
out0="$LOGDIR/p1c_phase_c0_${STAMP}.json"
for c in 2 0; do
  if ours_up; then
    say "stop $NAME for comm=$c"
    stop_ours
    sleep 2
  fi
  start_comm "$c" || { rc_all=4; break; }
  run_g1 "$c" || { rc_all=5; break; }
  run_bench "$c" || { rc_all=$?; }
  [ "$rc_all" -eq 0 ] || break
done

python3 - "$SUMMARY" "$out2" "$out0" <<'PY'
import json, sys
outp, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p):
    if not p:
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        return {"error": str(e), "path": p}
d2, d0 = load(a), load(b)
def med(d):
    if not d or "median_post_first_tok_s" not in d:
        return None
    return d.get("median_post_first_tok_s")
summary = {
    "comm2": d2,
    "comm0": d0,
    "median_comm2": med(d2),
    "median_comm0": med(d0),
    "paths": {"comm2": a, "comm0": b},
    "vs_q4km_43.8": {
        "comm2": (med(d2) / 43.8) if med(d2) else None,
        "comm0": (med(d0) / 43.8) if med(d0) else None,
    },
    "vs_doors1_31.78": {
        "comm2": (med(d2) / 31.78) if med(d2) else None,
        "comm0": (med(d0) / 31.78) if med(d0) else None,
    },
    "note": "Q8_DOORS=1 both. ignore_eos g128. COMM_DIRECT=3 forbidden. Q8_0 weight-only.",
}
json.dump(summary, open(outp, "w"), indent=2, sort_keys=True)
print(json.dumps({k: summary[k] for k in ("median_comm2", "median_comm0", "vs_q4km_43.8", "paths")}, indent=2))
PY

if [ "${STOP:-0}" = "1" ]; then
  stop_ours
  set_status "P1C_STATUS=STOPPED summary=$SUMMARY rc=$rc_all"
else
  if [ "$rc_all" -eq 0 ] && ! ours_up; then
    start_comm 2 || true
  elif [ "$rc_all" -eq 0 ]; then
    stop_ours
    sleep 2
    start_comm 2 || true
  fi
  set_status "P1C_STATUS=DONE summary=$SUMMARY rc=$rc_all serve_left_up=$(ours_up && echo 1 || echo 0)"
  if ours_up; then
    set_status "P1C_STATUS=HOLD docker-wait $NAME"
    docker wait "$NAME" >/dev/null 2>&1 || true
  fi
fi
exit "$rc_all"
