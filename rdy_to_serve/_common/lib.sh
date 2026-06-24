#!/usr/bin/env bash
# rdy_to_serve/_common/lib.sh -- SHARED, MODEL-AGNOSTIC serve plumbing for the B70 golden path.
#
# [!!!] SWEEP GATE (ORGANIZATION.md / CLAUDE.md): any change to THIS file requires
#       `bin/serve-sweep --smoke` GREEN across ALL rdy_to_serve models before commit, and
#       `--bench` if it could move perf. This is shared infra -- a break here breaks EVERY model.
#
# [!] What may live here: ONLY model-AGNOSTIC plumbing -- the docker-run builder, the graph-capture
#     flag assembly, the multi-GPU stability env, the health wait, the gen probe, the bench wrapper.
#     If a snippet would ever need `if MODEL is MoE/dense/TP` it does NOT belong here -- it stays in
#     the model's serve.sh. Model-specific knobs + patches are LOCAL to each <model>/ dir.
#
# A model serve.sh sets env vars, optionally builds a MOUNTS=( -v a:b:ro ... ) array for its local
# patches/, then calls:  b70_dispatch "$@"
#
# Engine ported from the proven host 30_serve_w4a8_graph.sh (the recipe that actually serves on the B70).
set -uo pipefail

# Locate the repo's bin/ (xpu-health, xe-reset) relative to THIS file (rdy_to_serve/_common/lib.sh).
B70_LIBDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B70_BIN="${B70_BIN:-$(cd "$B70_LIBDIR/../../bin" 2>/dev/null && pwd)}"

# True for any run that lights both cards (TP>1 OR PP>1): drives the #41663 multi-GPU stability env,
# the mp executor, and the wedge-guard pre-flight/teardown. A PP=2/TP=1 serve is multi-card too.
b70_multicard() { [ "${TP:-1}" -gt 1 ] || [ "${PP:-1}" -gt 1 ]; }

# ---- knobs (env, with defaults) -------------------------------------------------------------------
b70_setdefaults() {
  ROOT="${ROOT:-/mnt/vm_8tb/b70}"            # GPU host: models, caches, gpu-run, 35_sweep_bench
  IMG="${IMG:?serve.sh must set IMG}"        # docker image tag (prefer a digest pin: name@sha256:..)
  CKPT="${CKPT:?serve.sh must set CKPT}"     # CONTAINER path to the model dir (/models/...)
  SERVED="${SERVED:?serve.sh must set SERVED}"
  NAME="${NAME:-vllm_$SERVED}"               # container name (override for multi-replica)
  PORT="${PORT:-8000}"
  TP="${TP:-1}"                              # tensor-parallel size; >1 -> both cards + #41663 env
  PP="${PP:-1}"                              # pipeline-parallel size; >1 -> both cards + #41663 env (TP=1/stage)
  DEVICE="${DEVICE:-0}"                      # single-card (TP=1) card pin 0|1 (for data-parallel)
  GRAPH="${GRAPH:-0}"                        # 1 = PIECEWISE XPU graph capture (the decode lever)
  CGMODE="${CGMODE:-PIECEWISE}"             # PIECEWISE works on v0230; FULL is SYCL-Graph-blocked
  DTYPE="${DTYPE:-auto}"
  UTIL="${UTIL:-0.90}"
  MAXLEN="${MAXLEN:-8192}"
  MAXSEQS="${MAXSEQS:-8}"
  OMP="${OMP:-8}"
  CAPSIZES="${CAPSIZES:-}"                  # cudagraph capture sizes, e.g. 1,2,4,8,16,32,64
  COMPILESZ="${COMPILESZ-1}"                # compile_sizes; empty for spec-decode (1 -> padded-to-2 reject)
  QUANT="${QUANT:-}"                         # --quantization (e.g. quark); empty = let vLLM infer
  NOMM="${NOMM:-}"                           # 1 = text-only serve of a VLM (skip vision profiling crash)
  KVDTYPE="${KVDTYPE:-}"                     # e.g. fp8_e5m2 (fp8-storage KV; ~2x ctx/batch)
  SPEC="${SPEC:-}"                           # raw --speculative-config JSON
  MTPTOK="${MTPTOK:-}"                       # integer -> builds the MTP spec JSON (quote-safe)
  TOOLCALL="${TOOLCALL:-}"                   # 1 = OpenAI tool calling (qwen3_coder parser)
  TOOLPARSER="${TOOLPARSER:-qwen3_coder}"
  REASONPARSER="${REASONPARSER:-}"          # e.g. qwen3 -> split <think> into reasoning_content
  SYCLKERNELS="${SYCLKERNELS:-}"            # TP>1 oneCCL: default 0 (eager) / 1 (graph capture)
  SPLITOPS="${SPLITOPS:-}"                   # extra compilation splitting_ops (comma-quoted vllm:: op list).
                                            # TP>1 + MTP REQUIRES the collectives here: vLLM records the spec
                                            # all_gather into the SYCL graph but oneCCL's sched allgather has no
                                            # graph-recordable impl -> capture crash. Listing the collectives makes
                                            # inductor partition at them so they run EAGER (decode stays captured).
                                            # Empty = vLLM's default attention-only splitting_ops (unchanged).
  # MOUNTS (model-local patch mounts: -v host:container:ro ...) and DOCKER_ENV (model-local docker
  # env: -e NAME=val ...) are set by serve.sh as plain arrays. Do NOT default them here with
  # ${ARR[@]:-} -- that injects an empty-string element that docker reads as the image name. They
  # are expanded nounset-safe at the docker run via ${ARR[@]+...}.
  # bench knobs
  IN="${IN:-512}"; OUT="${OUT:-128}"; CONC="${CONC:-1 2 4 8}"
  mkdir -p "$ROOT"/{hf_cache,vllm_cache,tmp_ssd,results} 2>/dev/null || true
}

# ---- build the vllm CLI args + docker env from the knobs ------------------------------------------
b70_build() {
  ARGS=(serve "$CKPT" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT"
        --dtype "$DTYPE" --tensor-parallel-size "$TP" --max-model-len "$MAXLEN"
        --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL"
        --no-enable-prefix-caching --trust-remote-code)
  [ -n "$QUANT" ]   && ARGS+=(--quantization "$QUANT")
  [ "${PP:-1}" -gt 1 ] && ARGS+=(--pipeline-parallel-size "$PP")
  b70_multicard     && ARGS+=(--distributed-executor-backend mp)
  [ -n "$NOMM" ]    && ARGS+=(--limit-mm-per-prompt '{"image":0,"video":0}')
  [ -n "$KVDTYPE" ] && ARGS+=(--kv-cache-dtype "$KVDTYPE")
  [ -n "$MTPTOK" ] && [ -z "$SPEC" ] && SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTPTOK}}"
  [ -n "$SPEC" ]    && ARGS+=(--speculative-config "$SPEC")
  [ -n "$TOOLCALL" ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOLPARSER")
  [ -n "$REASONPARSER" ] && ARGS+=(--reasoning-parser "$REASONPARSER")

  GENV=(); GDOCK=(); EAGER=(--enforce-eager); CC=()
  if [ "$GRAPH" = 1 ]; then
    GENV=(-e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e OMP_NUM_THREADS="$OMP")
    GDOCK=(--pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556)
    EAGER=()
    # pass_config: disable the CUDA-only inductor fusion passes that NameError under torch.compile on XPU.
    local PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
    local CAP=""; [ -n "$CAPSIZES" ] && CAP="\"cudagraph_capture_sizes\":[$CAPSIZES],"
    local CSZ=""; [ -n "$COMPILESZ" ] && CSZ="\"compile_sizes\":[$COMPILESZ],"
    local SPL=""; [ -n "$SPLITOPS" ] && SPL="\"splitting_ops\":[$SPLITOPS],"
    # IGP: use_inductor_graph_partition. Default true (the newer partitioner). Set IGP=false to fall back to
    # the LEGACY piecewise splitter -- needed when a model MIXES quantized (weight_scale) + unquantized linears
    # in one captured region: the inductor partitioner raises `KeyError: weight_scale` collecting subgraph
    # inputs, while the legacy splitter handles the custom int8 op cleanly. splitting_ops still eject collectives.
    local IGP="${IGP:-true}"
    CC=(--compilation-config "{\"cudagraph_mode\":\"$CGMODE\",\"use_inductor_graph_partition\":${IGP},${CAP}${CSZ}${SPL}$PASS}")
  fi
  ARGS+=("${EAGER[@]}" "${CC[@]}")    # GRAPH=0 -> --enforce-eager ; GRAPH=1 -> the --compilation-config capture flags

  if b70_multicard; then
    # Battlemage multi-GPU stability env (vLLM #41663): no Arc P2P, CPU oneCCL over OFI, spawn, LZ v1.
    # Applies to TP>1 AND PP>1 (PP=2/TP=1 still drives both cards through oneCCL send/recv).
    local SK="$SYCLKERNELS"; [ -z "$SK" ] && SK=$([ "$GRAPH" = 1 ] && echo 1 || echo 0)
    MGPU=(-e CCL_ENABLE_SYCL_KERNELS="$SK" -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
          -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn
          -e CCL_TOPO_P2P_ACCESS="${P2PACCESS:-0}" -e CCL_ZE_IPC_EXCHANGE="${IPCX:-pidfd}")
    SHM="32g"
  else
    MGPU=(-e ZE_AFFINITY_MASK="$DEVICE")     # pin the single-card replica to a card (0|1)
    SHM="16g"
  fi
}

b70_serve() {
  b70_build
  docker rm -f "$NAME" 2>/dev/null || true
  echo "=== serve $SERVED  IMG=$IMG  TP=$TP PP=$PP  GRAPH=$GRAPH  port=$PORT $(b70_multicard || echo "card=$DEVICE") ==="
  echo "vllm ${ARGS[*]}"
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "$SHM" -p "${PORT}:${PORT}" "${GDOCK[@]}" \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
    -v "$ROOT/tmp_ssd:/tmp_ssd" "${MOUNTS[@]+"${MOUNTS[@]}"}" \
    -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
    -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
    "${DOCKER_ENV[@]+"${DOCKER_ENV[@]}"}" \
    "${MGPU[@]}" "${GENV[@]}" --entrypoint vllm "$IMG" "${ARGS[@]}" >/dev/null
}

# Progress-aware health wait (wedge-guard Layer 3). A flat 15-min wall-clock kill is what
# SIGKILLed a slow-but-fine GRAPH=1 TP=2 capture mid-flight and wedged both cards (P2P_GPU.md J.16).
# Instead: (a) a higher ceiling for the genuinely slow TP>1 GRAPH=1 capture, and (b) a STALL abort
# keyed on container-log forward progress -- a capture still emitting log lines is NOT killed, only a
# truly hung one (no new log line for HEALTH_STALL seconds) or an exited container is.
#   HEALTH_TIMEOUT  hard ceiling secs (default 1800 for TP>1+GRAPH=1, else 900)
#   HEALTH_STALL    abort if no log progress for this many secs (default 300)
b70_wait_healthy() {
  local ceiling="${HEALTH_TIMEOUT:-}"
  [ -n "$ceiling" ] || { if b70_multicard && [ "${GRAPH:-0}" = 1 ]; then ceiling=1800; else ceiling=900; fi; }
  local stall="${HEALTH_STALL:-300}"
  echo "=== waiting for /health (ceiling ${ceiling}s, stall-abort ${stall}s; first run JIT-compiles/captures) ==="
  local start now lines last_lines last_progress
  start=$(date +%s); last_lines=0; last_progress=$start
  while :; do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && {
      echo "=== HEALTHY :$PORT  $SERVED  (TP=$TP GRAPH=$GRAPH) in $(( $(date +%s) - start ))s ==="
      curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -oE '"id":"[^"]*"' | head -1
      return 0; }
    docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && {
      echo "[!] $NAME EXITED EARLY"; docker logs "$NAME" 2>&1 | tail -30; return 1; }
    now=$(date +%s)
    lines=$(docker logs "$NAME" 2>&1 | wc -l)
    [ "$lines" -gt "$last_lines" ] && { last_lines=$lines; last_progress=$now; }
    if [ $(( now - last_progress )) -ge "$stall" ]; then
      echo "[!] STALLED: no log progress for ${stall}s -- treating as hung init/capture (NOT force-killing a live one)."
      docker logs "$NAME" 2>&1 | tail -30; return 2
    fi
    if [ $(( now - start )) -ge "$ceiling" ]; then
      echo "[!] NOT HEALTHY after ${ceiling}s ceiling"; docker logs "$NAME" 2>&1 | tail -30; return 1
    fi
    sleep 5
  done
}

b70_gen_probe() {
  echo "--- gen probe (coherence-gated) ---"
  local resp txt verdict
  resp=$(curl -s --max-time 60 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":24,\"temperature\":0}")
  txt=$(printf '%s' "$resp" | grep -oE '"text":"[^"]*"' | head -1 | sed 's/^"text":"//; s/"$//')
  echo "gen: $txt"
  # COHERENCE GATE: a broken quant/capture/load serve emits a single repeated token ("!!!!" or "is is is")
  # while LOADING + /health stay green -- that masqueraded as a 3.4x MTP win (degenerate body -> draft==target
  # -> trivial ~98% accept). Flag if one non-space char is >55% of a >=12-char reply (catches the garbage).
  verdict=$(printf '%s' "$txt" | awk '{
    s=$0; gsub(/ /,"",s); n=length(s); if(n<12){print "SHORT"; exit}
    for(i=1;i<=n;i++){ch=substr(s,i,1); c[ch]++}
    max=0; for(k in c) if(c[k]>max) max=c[k];
    if(max/n>0.55) print "GARBAGE"; else print "OK" }')
  if [ "$verdict" = "GARBAGE" ]; then
    echo "[!] INCOHERENT GENERATION (degenerate repeated-token output) -- serve is BROKEN (quant ignore-list / capture / load bug)."
    return 1
  fi
  echo "[probe coherence: ${verdict:-OK}]"
}

b70_bench() {
  local par="tp${TP}"; [ "${PP:-1}" -gt 1 ] && par="pp${PP}"   # label encodes the parallelism mode
  env NAME="$NAME" MODEL="$SERVED" LABEL="${SERVED}-${par}$([ "$GRAPH" = 1 ] && echo -graph)" \
    TOKPATH="$CKPT" PORT="$PORT" IN="$IN" OUT="$OUT" CONC="$CONC" \
    bash "$ROOT/35_sweep_bench.sh"
}

# Graceful teardown (wedge-guard Layer 2). `docker rm -f` SIGKILLs the container; killing a TP>1
# worker mid-collective is THE wedge trigger (P2P_GPU.md H.13/J.15/J.16). `docker stop -t` sends
# SIGTERM + a grace window so vLLM finishes its shutdown synchronize() cleanly before removal.
b70_stop() {
  local g="${STOP_GRACE:-30}"
  docker stop -t "$g" "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" 2>/dev/null
  echo "stopped $NAME (graceful -t${g}; GPU released if last holder)"
}
b70_logs() { exec docker logs -f "$NAME"; }

# --- wedge guard (Layers 1/4/5); all NO-OP for TP=1 so the single-card sweep path is unchanged -----

# Layer 5 (the one hard guard) + Layer 1 (pre-flight health), run BEFORE a TP>1 serve.
b70_preflight() {
  b70_multicard || return 0            # single-card serves do not wedge -- skip entirely (TP>1 or PP>1 only)
  # Layer 5: refuse the ONE deterministic wedge trigger. Override (with an xe-reset between attempts)
  # via I_KNOW_P2P_WEDGES=1 so pioneering P2P-in-serve work is still possible, just not by accident.
  if [ "${P2PACCESS:-0}" = 1 ] && [ "${I_KNOW_P2P_WEDGES:-0}" != 1 ]; then
    echo "[GUARD] BLOCKED: CCL_TOPO_P2P_ACCESS=1 in a TP>1 serve is the one deterministic wedge trigger (P2P_GPU.md H.13)."
    echo "        To experiment anyway: set I_KNOW_P2P_WEDGES=1 and run bin/xe-reset between every attempt."
    return 1
  fi
  # Layer 1: never stack a TP>1 launch onto an already-wedged box (it only deepens the corruption).
  if [ -x "$B70_BIN/xpu-health" ]; then
    echo "=== pre-flight xpu-health (TP=$TP) ==="
    IMG="$IMG" "$B70_BIN/xpu-health"; local h=$?
    if [ "$h" = 1 ]; then
      echo "[GUARD] box is WEDGED before serve."
      if [ "${B70_AUTO_RESET:-0}" = 1 ] && [ -x "$B70_BIN/xe-reset" ]; then
        echo "[GUARD] B70_AUTO_RESET=1 -> xe-reset"; "$B70_BIN/xe-reset" || return 1
      else
        echo "[GUARD] recover with: $B70_BIN/xe-reset   (or set B70_AUTO_RESET=1 to auto-recover)"; return 1
      fi
    fi
  else
    echo "[GUARD] xpu-health not found at $B70_BIN -- skipping pre-flight probe (not blocking)."
  fi
  return 0
}

# Layer 4 (post-teardown verdict + optional auto-reset). Graceful-stops, then for any multi-card run
# (TP>1 or PP>1) checks the serve log for wedge markers AND re-probes the cards; auto-resets if B70_AUTO_RESET=1.
b70_teardown() {
  local logf="${B70_LOGDIR:-/tmp}/b70_${NAME}.log"
  docker logs "$NAME" >"$logf" 2>&1 || true     # capture BEFORE removal (b70_stop deletes the container)
  b70_stop
  b70_multicard || return 0
  local wedge=0
  if grep -qE 'DEVICE_LOST|OUT_OF_RESOURCES|UR_RESULT_ERROR' "$logf" 2>/dev/null; then
    echo "[GUARD] wedge markers in serve log ($logf):"
    grep -oE 'UR_RESULT_ERROR_[A-Z_]+|DEVICE_LOST|OUT_OF_RESOURCES' "$logf" 2>/dev/null | sort -u | sed 's/^/    /'
    wedge=1
  fi
  if [ -x "$B70_BIN/xpu-health" ]; then          # probe is ground truth (markers can be benign-shutdown noise)
    IMG="$IMG" "$B70_BIN/xpu-health" || wedge=1
  fi
  if [ "$wedge" = 1 ]; then
    if [ "${B70_AUTO_RESET:-0}" = 1 ] && [ -x "$B70_BIN/xe-reset" ]; then
      echo "[GUARD] B70_AUTO_RESET=1 -> auto-recovering with xe-reset"
      "$B70_BIN/xe-reset" || echo "[GUARD] xe-reset did NOT clear it -- escalate: sudo reboot"
    else
      echo "[GUARD] BOX MAY BE WEDGED. Recover before next TP>1 start: $B70_BIN/xe-reset  (or set B70_AUTO_RESET=1)"
    fi
  fi
}

# ---- the CLI every serve.sh shares ---------------------------------------------------------------
b70_dispatch() {
  b70_setdefaults
  case "${1:-start}" in
    stop)  b70_teardown ;;
    logs)  b70_logs ;;
    bench) b70_bench ;;
    start) b70_preflight && b70_serve && b70_wait_healthy && b70_gen_probe &&
           echo "Serving. Stop with: bash serve.sh stop  (holds the GPU until then)." ;;
    run)   b70_preflight && b70_serve && b70_wait_healthy && b70_gen_probe && b70_bench; b70_teardown ;;
    smoke) b70_preflight && b70_serve && b70_wait_healthy && b70_gen_probe; local rc=$?; b70_teardown; return $rc ;;
    *) echo "usage: serve.sh [start|stop|logs|bench|run|smoke]"; return 2 ;;
  esac
}
