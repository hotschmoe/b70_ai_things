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

# ---- knobs (env, with defaults) -------------------------------------------------------------------
b70_setdefaults() {
  ROOT="${ROOT:-/mnt/vm_8tb/b70}"            # GPU host: models, caches, gpu-run, 35_sweep_bench
  IMG="${IMG:?serve.sh must set IMG}"        # docker image tag (prefer a digest pin: name@sha256:..)
  CKPT="${CKPT:?serve.sh must set CKPT}"     # CONTAINER path to the model dir (/models/...)
  SERVED="${SERVED:?serve.sh must set SERVED}"
  NAME="${NAME:-vllm_$SERVED}"               # container name (override for multi-replica)
  PORT="${PORT:-8000}"
  TP="${TP:-1}"                              # tensor-parallel size; >1 -> both cards + #41663 env
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
  # MOUNTS (model-local patch mounts: -v host:container:ro ...) is set by serve.sh as a plain array.
  # Do NOT default it here with ${MOUNTS[@]:-} -- that injects an empty-string element that docker
  # reads as the image name. It is expanded nounset-safe at the docker run via ${MOUNTS[@]+...}.
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
  [ "$TP" -gt 1 ]   && ARGS+=(--distributed-executor-backend mp)
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
    CC=(--compilation-config "{\"cudagraph_mode\":\"$CGMODE\",\"use_inductor_graph_partition\":true,${CAP}${CSZ}$PASS}")
  fi

  if [ "$TP" -gt 1 ]; then
    # Battlemage multi-GPU stability env (vLLM #41663): no Arc P2P, CPU oneCCL over OFI, spawn, LZ v1.
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
  echo "=== serve $SERVED  IMG=$IMG  TP=$TP  GRAPH=$GRAPH  port=$PORT $([ "$TP" -le 1 ] && echo "card=$DEVICE") ==="
  echo "vllm ${ARGS[*]}"
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "$SHM" -p "${PORT}:${PORT}" "${GDOCK[@]}" \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
    -v "$ROOT/tmp_ssd:/tmp_ssd" "${MOUNTS[@]+"${MOUNTS[@]}"}" \
    -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
    -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
    "${MGPU[@]}" "${GENV[@]}" --entrypoint vllm "$IMG" "${ARGS[@]}" >/dev/null
}

b70_wait_healthy() {
  echo "=== waiting for /health (up to ~15 min; first run JIT-compiles / captures) ==="
  local i
  for i in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && {
      echo "=== HEALTHY :$PORT  $SERVED  (TP=$TP GRAPH=$GRAPH) ==="
      curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -oE '"id":"[^"]*"' | head -1
      return 0; }
    docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && {
      echo "[!] $NAME EXITED EARLY"; docker logs "$NAME" 2>&1 | tail -30; return 1; }
    sleep 5
  done
  echo "[!] NOT HEALTHY after wait"; docker logs "$NAME" 2>&1 | tail -30; return 1
}

b70_gen_probe() {
  echo "--- gen probe ---"
  curl -s --max-time 60 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":24,\"temperature\":0}" \
    | grep -oE '"text":"[^"]*"' | head -1; echo
}

b70_bench() {
  env NAME="$NAME" MODEL="$SERVED" LABEL="${SERVED}-tp${TP}$([ "$GRAPH" = 1 ] && echo -graph)" \
    TOKPATH="$CKPT" PORT="$PORT" IN="$IN" OUT="$OUT" CONC="$CONC" \
    bash "$ROOT/35_sweep_bench.sh"
}

b70_stop() { docker rm -f "$NAME" 2>/dev/null; echo "stopped $NAME (GPU released if last holder)"; }
b70_logs() { exec docker logs -f "$NAME"; }

# ---- the CLI every serve.sh shares ---------------------------------------------------------------
b70_dispatch() {
  b70_setdefaults
  case "${1:-start}" in
    stop)  b70_stop ;;
    logs)  b70_logs ;;
    bench) b70_bench ;;
    start) b70_serve && b70_wait_healthy && b70_gen_probe &&
           echo "Serving. Stop with: bash serve.sh stop  (holds the GPU until then)." ;;
    run)   b70_serve && b70_wait_healthy && b70_gen_probe && b70_bench; b70_stop ;;
    smoke) b70_serve && b70_wait_healthy && b70_gen_probe; local rc=$?; b70_stop; return $rc ;;
    *) echo "usage: serve.sh [start|stop|logs|bench|run|smoke]"; return 2 ;;
  esac
}
