#!/usr/bin/env bash
# decode_optrace.sh -- per-op DECODE layer-timing trace (RESEARCH_TODO Track 1e).
#
# GOAL: find any decode matmul that is NOT landing on the int8 / nvfp4 XMX fast path
# (i.e. it silently fell back to a bf16 reorder+matmul or the oneDNN reference GEMM =
# a free-win leak). Method = ONEDNN_VERBOSE=dispatch,profile_exec captured while the
# real Qwen3.6-27B decode-shape GEMMs run, then parsed by parse_optrace.py.
#
# ------------------------------------------------------------------------------------
# TWO MODES (env MODE=micro|live). Coordinator runs this; it does touch the GPU.
#
#   MODE=micro (DEFAULT, least disruptive):
#     docker exec into an EXISTING serve container and run a fresh EAGER python tracer
#     that replays each real decode-shape proj (qkv/o/gate_up/down) at decode M
#     (M=1 single-stream, M=1+MTPTOK for the MTP verify batch) through the mounted
#     custom kernels with ONEDNN_VERBOSE on. A fresh python process => oneDNN reads
#     ONEDNN_VERBOSE at import (the live serve did not, so we cannot read its ops from
#     a running captured-graph serve any other way). This catches KERNEL/SHAPE leaks:
#     "does nvfp4_gemm_w4a16 / int8_gemm_w8a16 at THIS decode shape dispatch to
#     jit:gemm:xe (XMX) or fall to ref / a per-call bf16 reorder". It does NOT see a
#     LOADER leak (a layer whose weight got materialised to bf16 at load) -- for that
#     use MODE=live. Random weights; correct dtypes/shapes.
#     NOTE: run while the serve is idle (no in-flight requests) so the microbench GEMMs
#     do not contend; footprint is small (<0.5 GiB) and fits the UTIL headroom.
#
#   MODE=live:
#     Assumes the serve was STARTED with ONEDNN_VERBOSE=dispatch,profile_exec AND
#     --enforce-eager (GRAPH=0): a captured cudagraph replays kernels without oneDNN
#     re-emitting verbose, so the live trace only works EAGER. Launch e.g.:
#       B70_EXTRA_ENV="ONEDNN_VERBOSE=dispatch,profile_exec" GRAPH=0 ... serve ... start
#     Then this mode marks the container-log offset, fires a short single-stream forced
#     decode (DECTOK tokens, temp 0, ignore_eos, concurrency 1) via HOST:PORT, captures
#     the log delta, and parses it. This is the ONLY mode that catches a per-layer
#     LOADER dequant leak, because it traces the ACTUAL model forward.
#
# ------------------------------------------------------------------------------------
# ENV:
#   MODE=micro|live         (default micro)
#   NAME=<container>        serve container to exec/log (default nvfp4_27b; DD=b70_daily_0)
#   SCHEME=nvfp4|w8a8|both  which kernel paths to trace (default both; ops absent in the
#                           mounted .so are skipped -- run once per serve container)
#   MTPTOK=<n>              MTP verify batch size probe = 1+MTPTOK (default 5 -> M=6)
#   ZE_MASK=<n>             ZE_AFFINITY_MASK for the micro exec (default 0)
#   OUTDIR=<dir>            where to write the raw log + parsed table (default scratch)
#   -- live only --
#   HOST=<ip>              default 192.168.10.5
#   PORT=<port>            default 18080
#   MODEL=<served-id>      default = auto (first /v1/models id)
#   KEY=<api-key>          bearer token if the serve enforces --api-key
#   DECTOK=<n>             forced decode tokens (default 24)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-micro}"
NAME="${NAME:-nvfp4_27b}"
SCHEME="${SCHEME:-both}"
MTPTOK="${MTPTOK:-5}"
ZE_MASK="${ZE_MASK:-0}"
OUTDIR="${OUTDIR:-/tmp/claude-1000/optrace}"
mkdir -p "$OUTDIR"
RAW="$OUTDIR/optrace_${MODE}_${SCHEME}.log"

if [ "$MODE" = micro ]; then
  echo "=== MODE=micro: docker exec tracer in container '$NAME' (SCHEME=$SCHEME MTPTOK=$MTPTOK) ==="
  # The tracer runs INSIDE the serve container so it imports the mounted custom kernel .so.
  # ONEDNN_VERBOSE + banners both go to stdout; torch.xpu.synchronize() after each op keeps
  # oneDNN exec lines grouped under the right banner.
  docker exec -i \
    -e ONEDNN_VERBOSE=dispatch,profile_exec \
    -e NVFP4_XPU_MODE=fused \
    -e ZE_AFFINITY_MASK="$ZE_MASK" \
    -e TRACE_SCHEME="$SCHEME" \
    -e TRACE_MTPTOK="$MTPTOK" \
    "$NAME" python3 - <<'PY' 2>&1 | tee "$RAW"
import os, torch
try:
    import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C.*)
except Exception as e:
    print("FATAL import vllm_xpu_kernels._xpu_C:", repr(e), flush=True); raise
DEV = "xpu"; torch.manual_seed(0)
SCHEME = os.environ.get("TRACE_SCHEME", "both")
MTPTOK = int(os.environ.get("TRACE_MTPTOK", "5"))
ops = torch.ops._xpu_C
H = 5120
# (label, N, K) -- the quantized dense linears of Qwen3.6-27B (hidden=5120, inter=17408,
# q=24*256=6144, kv=4*256=1024 -> qkv=8192; o_proj in=6144). These are exactly the
# per-token decode GEMMs. down_proj is the K-heavy one.
SHAPES = [
    ("qkv_proj",     8192,  5120),
    ("o_proj",       5120,  6144),
    ("gate_up_proj", 34816, 5120),
    ("down_proj",    5120,  17408),
]
MS = [1, 1 + MTPTOK]  # single-stream decode, and the MTP verify batch (1+draft)

def banner(s):
    print("\n##### %s #####" % s, flush=True)

def do_nvfp4(label, N, K, M):
    xb = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
    packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=DEV)
    nv_scale = torch.rand(K // 16, N, device=DEV, dtype=torch.bfloat16) * 0.1 + 0.01
    banner("OP=%s M=%d scheme=nvfp4 op=nvfp4_gemm_w4a16" % (label, M))
    ops.nvfp4_gemm_w4a16(xb, packed.t(), None, nv_scale, 16); torch.xpu.synchronize()

def do_w8a16(label, N, K, M):
    xb = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
    s8w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=DEV)
    grp = torch.rand(K // 16, N, device=DEV, dtype=torch.bfloat16) * 0.1 + 0.01
    banner("OP=%s M=%d scheme=w8a16 op=int8_gemm_w8a16" % (label, M))
    ops.int8_gemm_w8a16(xb, s8w.t(), grp, None); torch.xpu.synchronize()

def do_w8a8(label, N, K, M):
    xs8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=DEV)
    asc = torch.rand(M, 1, device=DEV, dtype=torch.float32) * 0.01 + 0.001
    s8w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=DEV)
    pc = torch.rand(N, device=DEV, dtype=torch.float32) * 0.01 + 0.001
    banner("OP=%s M=%d scheme=w8a8 op=int8_gemm_w8a8" % (label, M))
    ops.int8_gemm_w8a8(xs8, asc, None, s8w.t(), pc, None, None, torch.bfloat16); torch.xpu.synchronize()

def do_bf16ref(label, N, K, M):
    xb = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02
    banner("OP=%s M=%d scheme=bf16ref op=F.linear" % (label, M))
    torch.nn.functional.linear(xb, w); torch.xpu.synchronize()

want = {"nvfp4": SCHEME in ("nvfp4", "both"),
        "w8a8":  SCHEME in ("w8a8", "both")}
has_nvfp4 = hasattr(ops, "nvfp4_gemm_w4a16")
has_w8a16 = hasattr(ops, "int8_gemm_w8a16")
has_w8a8  = hasattr(ops, "int8_gemm_w8a8")
print("kernel ops present: nvfp4_gemm_w4a16=%s int8_gemm_w8a16=%s int8_gemm_w8a8=%s"
      % (has_nvfp4, has_w8a16, has_w8a8), flush=True)

# warm once so oneDNN primitive CREATION (dispatch lines) is separated from the timed reps;
# then a second timed pass emits the steady-state exec lines the parser scores.
for timed in (False, True):
    if timed:
        print("\n########## TIMED PASS (steady-state exec lines below) ##########", flush=True)
    for M in MS:
        for label, N, K in SHAPES:
            do_bf16ref(label, N, K, M)              # reference for speedup-vs-bf16
            if want["nvfp4"] and has_nvfp4: do_nvfp4(label, N, K, M)
            if want["w8a8"] and has_w8a16:  do_w8a16(label, N, K, M)
            if want["w8a8"] and has_w8a8:   do_w8a8(label, N, K, M)
print("\n##### DONE #####", flush=True)
PY
  echo
  echo "=== parse ==="
  python3 "$HERE/parse_optrace.py" "$RAW" | tee "$OUTDIR/optrace_${MODE}_${SCHEME}.table.txt"
  echo "raw: $RAW"
  exit 0
fi

if [ "$MODE" = live ]; then
  HOST="${HOST:-192.168.10.5}"; PORT="${PORT:-18080}"; DECTOK="${DECTOK:-24}"
  KEY="${KEY:-}"; MODEL="${MODEL:-}"
  AUTH=(); [ -n "$KEY" ] && AUTH=(-H "Authorization: Bearer $KEY")
  if [ -z "$MODEL" ]; then
    MODEL="$(curl -s "${AUTH[@]}" "http://$HOST:$PORT/v1/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')"
  fi
  echo "=== MODE=live: container '$NAME' served='$MODEL' -> $DECTOK forced decode tokens ==="
  echo "    (serve MUST have been started with ONEDNN_VERBOSE=dispatch,profile_exec + --enforce-eager)"
  # mark current end of container log so we only capture this decode's lines
  MARK="$(docker logs "$NAME" 2>&1 | wc -l)"
  # short single-stream forced decode (concurrency 1) -> small steady decode M, no batching noise
  curl -s "${AUTH[@]}" -H "Content-Type: application/json" \
    "http://$HOST:$PORT/v1/completions" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Count slowly: one two three\",\"max_tokens\":$DECTOK,\"temperature\":0,\"ignore_eos\":true}" \
    >/dev/null || { echo "decode request failed"; exit 1; }
  sleep 2
  docker logs "$NAME" 2>&1 | tail -n +"$((MARK+1))" | grep -a onednn_verbose > "$RAW" || true
  echo "captured $(wc -l < "$RAW") onednn_verbose lines"
  echo "=== parse ==="
  python3 "$HERE/parse_optrace.py" "$RAW" | tee "$OUTDIR/optrace_${MODE}.table.txt"
  echo "raw: $RAW"
  exit 0
fi

echo "unknown MODE=$MODE (want micro|live)" >&2; exit 2
