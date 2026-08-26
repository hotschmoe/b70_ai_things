#!/usr/bin/env bash
# Exact Qwen3.6 TP2 control with the restored inner clone.
# Direct P2P remains the default forensic arm and requires
# I_KNOW_P2P_WEDGES=1. Set P2P_ACCESS=0 for the stable PCIe-only arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMG="intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94"
MODEL_HOST="$REPO_ROOT/models/files/qwen3.6-35b-a3b/quark-w8a8-int8"
MODEL_REVISION="cced56592e8c8935f8220836b4baa04dfd389118"
EXPECTED_CONFIG_SHA256="b2a92fb7dfea6bdd94572df58b198efe6a391c81dbc5b848b15a2c43d6f9abc0"
EXPECTED_INDEX_SHA256="c973ada0f6042784fb2f8dbe53b81c6b8a78887ddc487bcf08f2c6d2d42f2f7a"
EXPECTED_CCL_SHA256="542142aca8f3d318616eae0f300aaa47dc62b217831599cb1461212f8aa4dc76"
EXPECTED_C_SHA256="5717476461048b5056a92926f2a52d73c121f69bdc75de22fd52720fb65b3007"
EXPECTED_MOE_C_SHA256="ea4c20a8dff49fc07fd799d5a2a47e8b24266a256425b41e337f852492ee3c1b"
EXPECTED_FA2_C_SHA256="b62f6f70e7c991ceffd3c326092b7c26e48dbe502e2a76535e0024d2d3f2fc5c"
EXPECTED_ATTN_SHA256="773b5539b37abf163c59caebc9956390e7c9741a458c343046dcaf2178e7104f"
EXPECTED_GROUPED_DEFAULT_SHA256="1f5ec0f22ec4e21ec59e3fc38b46818398e296db0a7e10a13163907422ac490a"
EXPECTED_MQA_SHA256="e51af18e63cf3f888bcbf9d99f8207b2521ce3182121a9b3ea9a33b490c39ca5"
EXPECTED_ALLOCATOR_SHA256="76c1301123723f643e7eae6160b7c72adc3db9f76bdb51225c686578637be57a"
EXPECTED_FUSED_MOE_SHA256="433ee08a80ab8e1c12d000e7d2a683c0e325c2971868ca31075075a863b7d81a"
EXPECTED_CCL_KERNELS_SHA256="0d549c35a558f1b216cb7d1efeaa9f86d7596ffc47b383644e075290d314f0c9"
STEVE_TIMING_REFERENCE="$SCRIPT_DIR/manifests/qwen36_steve_tp2_synced_timing_reference_20260619.json"
EXPECTED_STEVE_TIMING_REFERENCE_SHA256="bda64332cb8a9a70a9fd04536c3928931d904b25eb9368c177518bd7788efe28"
NATIVE_STACK="${NATIVE_STACK:-june9-minimal}"
case "$NATIVE_STACK" in
  june9-minimal)
    JUNE_RUNTIME="/mnt/vm_8tb/b70/steve-repro/june-xpuc-bmg-g21-a0-20260825/runtime-candidate"
    EXPECTED_XPU_C_SHA256="2d931484ee0aadd4c9fb6abf494e147a5275210a216426a1eb56add0158bef0d"
    EXPECTED_GROUPED_SHA256="f5ddc2ee3c11dcede3a7190b69d6e0dd354bb0727be7519600abaebe9fc4cd2c"
    EXPECTED_GDN_SHA256="366935b172b5c9c3cb75bee5d7bfe0434f377a6317314a9a43c853b5d02fe83b"
    KERNEL_PREFLIGHT_SUITE=full
    native_suffix=""
    ;;
  june122-checkpoint)
    JUNE_RUNTIME="/mnt/vm_8tb/b70/steve-repro/june122-xpuc-regular-20260826/runtime-candidate"
    EXPECTED_XPU_C_SHA256="631f733119e33dfe37d0691e6520749b07d1417bd55f29bde2ed56ab49e586e8"
    EXPECTED_GROUPED_SHA256="7d38d1603de9b703a2c0106a354d2d40ad449afe4f539e374b215f1895206aaa"
    EXPECTED_GDN_SHA256="ee0481c8affca919371f740e847e4d5e6a84c1ee6a7c398437e13aff32238b02"
    KERNEL_PREFLIGHT_SUITE=native-out
    native_suffix="_native_june122"
    ;;
  *) echo "NATIVE_STACK must be june9-minimal or june122-checkpoint" >&2; exit 2 ;;
esac
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
P2P_ACCESS="${P2P_ACCESS:-1}"
case "$P2P_ACCESS" in 0|1) ;; *) echo "P2P_ACCESS must be 0 or 1" >&2; exit 2 ;; esac
SOURCE_STACK="${SOURCE_STACK:-august-adapter}"
case "$SOURCE_STACK" in
  august-adapter|june-e190) ;;
  *) echo "SOURCE_STACK must be august-adapter or june-e190" >&2; exit 2 ;;
esac
MOE_TRACE="${MOE_TRACE:-0}"
case "$MOE_TRACE" in 0|1) ;; *) echo "MOE_TRACE must be 0 or 1" >&2; exit 2 ;; esac
ALLREDUCE_TRACE="${ALLREDUCE_TRACE:-0}"
case "$ALLREDUCE_TRACE" in 0|1) ;; *) echo "ALLREDUCE_TRACE must be 0 or 1" >&2; exit 2 ;; esac
ALLREDUCE_TRACE_SYNC="${ALLREDUCE_TRACE_SYNC:-$ALLREDUCE_TRACE}"
case "$ALLREDUCE_TRACE_SYNC" in 0|1) ;; *) echo "ALLREDUCE_TRACE_SYNC must be 0 or 1" >&2; exit 2 ;; esac
ALLREDUCE_TRACE_MAX_CALLS="${ALLREDUCE_TRACE_MAX_CALLS:-256}"
case "$ALLREDUCE_TRACE_MAX_CALLS" in
  ''|*[!0-9]*) echo "ALLREDUCE_TRACE_MAX_CALLS must be a non-negative integer" >&2; exit 2 ;;
esac
profile_fence_default_rows=0
[ "$P2P_ACCESS" = 1 ] && profile_fence_default_rows=8192
PROFILE_FENCE_MIN_ROWS="${PROFILE_FENCE_MIN_ROWS:-$profile_fence_default_rows}"
case "$PROFILE_FENCE_MIN_ROWS" in
  ''|*[!0-9]*) echo "PROFILE_FENCE_MIN_ROWS must be a non-negative integer" >&2; exit 2 ;;
esac
PROFILE_FENCE_STAGES="${PROFILE_FENCE_STAGES:-clone}"
IFS=, read -r -a profile_fence_stage_list <<< "$PROFILE_FENCE_STAGES"
[ "${#profile_fence_stage_list[@]}" -gt 0 ] || {
  echo "PROFILE_FENCE_STAGES must not be empty" >&2
  exit 2
}
for profile_fence_stage in "${profile_fence_stage_list[@]}"; do
  case "$profile_fence_stage" in
    pre|clone|post) ;;
    *) echo "PROFILE_FENCE_STAGES must be a comma-separated subset of pre,clone,post" >&2; exit 2 ;;
  esac
done
STALL_TIMEOUT="${STALL_TIMEOUT:-180}"
case "$STALL_TIMEOUT" in
  ''|*[!0-9]*) echo "STALL_TIMEOUT must be a non-negative integer" >&2; exit 2 ;;
esac
DECODE_TIMING="${DECODE_TIMING:-0}"
DECODE_TIMING_SYNC="${DECODE_TIMING_SYNC:-$DECODE_TIMING}"
CUDAGRAPH_REPLAY_TRACE="${CUDAGRAPH_REPLAY_TRACE:-0}"
for binary_setting in DECODE_TIMING DECODE_TIMING_SYNC CUDAGRAPH_REPLAY_TRACE; do
  case "${!binary_setting}" in
    0|1) ;;
    *) echo "$binary_setting must be 0 or 1" >&2; exit 2 ;;
  esac
done
DECODE_TIMING_SKIP_FIRST="${DECODE_TIMING_SKIP_FIRST:-32}"
DECODE_TIMING_STEP_SKIP_FIRST="${DECODE_TIMING_STEP_SKIP_FIRST:-32}"
DECODE_TIMING_STEP_EVERY="${DECODE_TIMING_STEP_EVERY:-16}"
CUDAGRAPH_REPLAY_TRACE_MAX_LINES="${CUDAGRAPH_REPLAY_TRACE_MAX_LINES:-4096}"
for integer_setting in \
  DECODE_TIMING_SKIP_FIRST \
  DECODE_TIMING_STEP_SKIP_FIRST \
  DECODE_TIMING_STEP_EVERY \
  CUDAGRAPH_REPLAY_TRACE_MAX_LINES; do
  case "${!integer_setting}" in
    ''|*[!0-9]*) echo "$integer_setting must be a non-negative integer" >&2; exit 2 ;;
  esac
done
[ "$DECODE_TIMING_STEP_EVERY" -gt 0 ] || {
  echo "DECODE_TIMING_STEP_EVERY must be greater than zero" >&2
  exit 2
}
source_suffix=""
[ "$SOURCE_STACK" = june-e190 ] && source_suffix="_june_e190"
NAME="qwen36_s2b_exactcc_clone_p2p${P2P_ACCESS}${source_suffix}${native_suffix}_${STAMP}"
PORT="${PORT:-18080}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/logs/$NAME}"
CACHE_DIR="${CACHE_DIR:-/mnt/vm_8tb/b70/vllm_cache_${NAME}}"
REPLAY_TRACE_FILE_VALUE=""
if [ "$CUDAGRAPH_REPLAY_TRACE" = 1 ]; then
  REPLAY_TRACE_FILE_VALUE="/results/logs/$NAME/cudagraph_replay_rank{rank}.jsonl"
fi
RUN_LOG="$RESULT_DIR/run.log"
SERVER_LOG="$RESULT_DIR/b70_${NAME}.log"
HF_REF="/mnt/vm_8tb/b70/hf_cache/hub/models--nameistoken--Qwen3.6-35B-A3B-Quark-W8A8-INT8/refs/main"

if [ "$P2P_ACCESS" = 1 ] && [ "${I_KNOW_P2P_WEDGES:-0}" != 1 ]; then
  echo "Refusing exact direct-P2P control without I_KNOW_P2P_WEDGES=1" >&2
  exit 2
fi
if [ -e "$CACHE_DIR" ] && [ "${ALLOW_EXISTING_CACHE:-0}" != 1 ]; then
  echo "Refusing non-fresh compilation cache: $CACHE_DIR" >&2
  exit 2
fi
if [ -n "${FORENSIC_VLLM_SRC:-}" ] || [ -n "${FORENSIC_SITECUSTOMIZE_HOST:-}" ] || [ -n "${FORENSIC_FUSED_MOE_INTERFACE_HOST:-}" ] || [ -n "${B70_EXTRA_ENV:-}" ]; then
  echo "Refusing source or environment overlays in the exact control" >&2
  exit 2
fi

FORENSIC_VLLM_SRC_VALUE=""
FORENSIC_SITECUSTOMIZE_VALUE=""
FORENSIC_FUSED_MOE_INTERFACE_VALUE=""
MIXED_WORKSPACE=0
if [ "$SOURCE_STACK" = june-e190 ]; then
  FORENSIC_VLLM_SRC_VALUE="/mnt/vm_8tb/b70/steve-s2b/vllm-e190"
  FORENSIC_SITECUSTOMIZE_VALUE="$SCRIPT_DIR/qwen36_june_source_sitecustomize.py"
  FORENSIC_FUSED_MOE_INTERFACE_VALUE="/mnt/vm_8tb/b70/steve-s2b/vllm-xpu-kernels/vllm_xpu_kernels/fused_moe_interface.py"
  MIXED_WORKSPACE=1
  [ "$(git -C "$FORENSIC_VLLM_SRC_VALUE" rev-parse HEAD)" = \
    "e190923b32e1b87fe33d08264bff9215fb7770fc" ] || {
    echo "June source worktree identity mismatch" >&2
    exit 1
  }
  [ -z "$(git -C "$FORENSIC_VLLM_SRC_VALUE" status --short)" ] || {
    echo "June source worktree is dirty" >&2
    exit 1
  }
  [ "$(sha256sum "$FORENSIC_FUSED_MOE_INTERFACE_VALUE" | awk '{print $1}')" = \
    "dca1423d7346debbc9bbf33398c3364e2a911b751b9d94423ec28b2778015b34" ] || {
    echo "Scratch-aware fused MoE interface identity mismatch" >&2
    exit 1
  }
fi

mkdir -p "$RESULT_DIR" "$(dirname "$CACHE_DIR")"
if [ "$CUDAGRAPH_REPLAY_TRACE" = 1 ]; then
  mkdir -p "/mnt/vm_8tb/b70/results/logs/$NAME"
fi

actual_config_sha256="$(sha256sum "$MODEL_HOST/config.json" | awk '{print $1}')"
actual_index_sha256="$(sha256sum "$MODEL_HOST/model.safetensors.index.json" | awk '{print $1}')"
actual_revision="$(tr -d '[:space:]' < "$HF_REF")"
[ "$actual_config_sha256" = "$EXPECTED_CONFIG_SHA256" ] || {
  echo "Model config hash mismatch: $actual_config_sha256" >&2
  exit 1
}
[ "$actual_index_sha256" = "$EXPECTED_INDEX_SHA256" ] || {
  echo "Model index hash mismatch: $actual_index_sha256" >&2
  exit 1
}
[ "$actual_revision" = "$MODEL_REVISION" ] || {
  echo "Model revision mismatch: $actual_revision" >&2
  exit 1
}
for required in \
  vllm_xpu_kernels/_C.abi3.so \
  vllm_xpu_kernels/_moe_C.abi3.so \
  vllm_xpu_kernels/_vllm_fa2_C.abi3.so \
  vllm_xpu_kernels/_xpu_C.abi3.so \
  vllm_xpu_kernels/libattn_kernels_xe_2.so \
  vllm_xpu_kernels/libgrouped_gemm_xe_2.so \
  vllm_xpu_kernels/libgrouped_gemm_xe_default.so \
  vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so \
  vllm_xpu_kernels/libmqa_logits_kernels_xe_2.so \
  vllm_xpu_kernels/xpumem_allocator.abi3.so \
  vllm_xpu_kernels/fused_moe_interface.py; do
  [ -f "$JUNE_RUNTIME/$required" ] || {
    echo "Missing June runtime artifact: $JUNE_RUNTIME/$required" >&2
    exit 1
  }
done
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/_C.abi3.so" | awk '{print $1}')" = "$EXPECTED_C_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/_moe_C.abi3.so" | awk '{print $1}')" = "$EXPECTED_MOE_C_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/_vllm_fa2_C.abi3.so" | awk '{print $1}')" = "$EXPECTED_FA2_C_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/_xpu_C.abi3.so" | awk '{print $1}')" = "$EXPECTED_XPU_C_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/libattn_kernels_xe_2.so" | awk '{print $1}')" = "$EXPECTED_ATTN_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/libgrouped_gemm_xe_2.so" | awk '{print $1}')" = "$EXPECTED_GROUPED_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/libgrouped_gemm_xe_default.so" | awk '{print $1}')" = "$EXPECTED_GROUPED_DEFAULT_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so" | awk '{print $1}')" = "$EXPECTED_GDN_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/libmqa_logits_kernels_xe_2.so" | awk '{print $1}')" = "$EXPECTED_MQA_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/xpumem_allocator.abi3.so" | awk '{print $1}')" = "$EXPECTED_ALLOCATOR_SHA256" ]
[ "$(sha256sum "$JUNE_RUNTIME/vllm_xpu_kernels/fused_moe_interface.py" | awk '{print $1}')" = "$EXPECTED_FUSED_MOE_SHA256" ]
[ "$(sha256sum "$STEVE_TIMING_REFERENCE" | awk '{print $1}')" = \
  "$EXPECTED_STEVE_TIMING_REFERENCE_SHA256" ] || {
  echo "Steve timing reference identity mismatch" >&2
  exit 1
}

# This preflight has no /dev/dri mount. It proves the pinned image artifacts
# before the first actual GPU touch of the reboot-bounded transaction.
mapfile -t runtime_hashes < <(
  docker run --rm --entrypoint sha256sum \
    -v "$JUNE_RUNTIME:/opt/june-runtime:ro" "$IMG" \
    /opt/venv/lib/libccl.so.1.0 \
    /opt/june-runtime/vllm_xpu_kernels/_xpu_C.abi3.so \
    /opt/june-runtime/vllm_xpu_kernels/libgrouped_gemm_xe_2.so \
    /opt/june-runtime/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so \
    /opt/ccl4ce/lib/ccl/kernels/kernels.spv
)
[ "${runtime_hashes[0]%% *}" = "$EXPECTED_CCL_SHA256" ] || {
  echo "Pinned oneCCL hash mismatch: ${runtime_hashes[0]}" >&2
  exit 1
}
[ "${runtime_hashes[1]%% *}" = "$EXPECTED_XPU_C_SHA256" ] || {
  echo "Pinned _xpu_C hash mismatch: ${runtime_hashes[1]}" >&2
  exit 1
}
[ "${runtime_hashes[2]%% *}" = "$EXPECTED_GROUPED_SHA256" ] || {
  echo "June grouped-GEMM hash mismatch: ${runtime_hashes[2]}" >&2
  exit 1
}
[ "${runtime_hashes[3]%% *}" = "$EXPECTED_GDN_SHA256" ] || {
  echo "June GDN hash mismatch: ${runtime_hashes[3]}" >&2
  exit 1
}
[ "${runtime_hashes[4]%% *}" = "$EXPECTED_CCL_KERNELS_SHA256" ] || {
  echo "Pinned oneCCL kernels hash mismatch: ${runtime_hashes[4]}" >&2
  exit 1
}
JUNE_HASHES_JSON="{\"_C.abi3.so\":\"$EXPECTED_C_SHA256\",\"_moe_C.abi3.so\":\"$EXPECTED_MOE_C_SHA256\",\"_xpu_C.abi3.so\":\"$EXPECTED_XPU_C_SHA256\",\"libgrouped_gemm_xe_2.so\":\"$EXPECTED_GROUPED_SHA256\",\"libgdn_attn_kernels_xe_2.so\":\"$EXPECTED_GDN_SHA256\"}"
docker run --rm --entrypoint python \
  -e PYTHONPATH=/opt/june-runtime \
  -v "$JUNE_RUNTIME:/opt/june-runtime:ro" \
  -v "$SCRIPT_DIR/qwen36_june_august_kernel_arm.py:/opt/kernel_arm.py:ro" \
  -v "$RESULT_DIR:/opt/preflight-out" \
  "$IMG" /opt/kernel_arm.py \
  --arm june-exact-control-preflight \
  --suite "$KERNEL_PREFLIGHT_SUITE" \
  --offdevice \
  --expected-package-root /opt/june-runtime/vllm_xpu_kernels \
  --expected-hashes "$JUNE_HASHES_JSON" \
  --output /opt/preflight-out/kernel_runtime_preflight.json
docker run --rm --entrypoint python \
  -e PYTHONPATH=/opt/june-runtime:/opt/b70_qwen36_site \
  -v "$JUNE_RUNTIME:/opt/june-runtime:ro" \
  -v "$SCRIPT_DIR/qwen36_s2b_sitecustomize.py:/opt/b70_qwen36_site/sitecustomize.py:ro" \
  -v "$SCRIPT_DIR/qwen36_piecewise_capture_contract.py:/opt/piecewise_contract.py:ro" \
  -v "$RESULT_DIR:/opt/preflight-out" \
  "$IMG" /opt/piecewise_contract.py \
  --output /opt/preflight-out/piecewise_capture_contract.json
docker run --rm --entrypoint python \
  -e VLLM_TARGET_DEVICE=xpu \
  -e PYTHONPATH=/opt/june-runtime:/opt/b70_qwen36_site \
  -v "$JUNE_RUNTIME:/opt/june-runtime:ro" \
  -v "$SCRIPT_DIR/qwen36_s2b_sitecustomize.py:/opt/b70_qwen36_site/sitecustomize.py:ro" \
  -v "$SCRIPT_DIR/qwen36_native_moe_route_contract.py:/opt/native_moe_contract.py:ro" \
  -v "$RESULT_DIR:/opt/preflight-out" \
  "$IMG" /opt/native_moe_contract.py \
  --output /opt/preflight-out/native_moe_route_contract.json
if [ "$SOURCE_STACK" = june-e190 ]; then
  docker run --rm --entrypoint python \
    -e VLLM_TARGET_DEVICE=xpu \
    -e PYTHONPATH=/opt/forensic_vllm:/opt/june-runtime \
    -v "$FORENSIC_VLLM_SRC_VALUE:/opt/forensic_vllm:ro" \
    -v "$JUNE_RUNTIME:/opt/june-runtime:ro" \
    -v "$FORENSIC_FUSED_MOE_INTERFACE_VALUE:/opt/june-runtime/vllm_xpu_kernels/fused_moe_interface.py:ro" \
    -v "$SCRIPT_DIR/qwen36_june_source_contract.py:/opt/june_source_contract.py:ro" \
    -v "$RESULT_DIR:/opt/preflight-out" \
    "$IMG" /opt/june_source_contract.py \
    --output /opt/preflight-out/june_source_contract.json
fi

echo "config -> image=$IMG model_revision=$MODEL_REVISION model_config=$actual_config_sha256 model_index=$actual_index_sha256"
echo "config -> TP=2 PP=1 graph=PIECEWISE splitops=default igp=false async=1 mtp=0 prefix_cache=0 maxlen=32768 maxseqs=24 util=0.90"
if [ "$SOURCE_STACK" = june-e190 ]; then
  echo "config -> native_moe=june-source moe_mixed_workspace=$MIXED_WORKSPACE call_abi=recovered-scratch-aware-june"
else
  echo "config -> native_moe=june-base moe_mixed_workspace=$MIXED_WORKSPACE call_abi=august-keywords-to-june-base"
fi
echo "config -> moe_trace=$MOE_TRACE allreduce_trace=$ALLREDUCE_TRACE allreduce_trace_sync=$ALLREDUCE_TRACE_SYNC allreduce_trace_max_calls=$ALLREDUCE_TRACE_MAX_CALLS profile_fence_min_rows=$PROFILE_FENCE_MIN_ROWS profile_fence_stages=$PROFILE_FENCE_STAGES cache_reuse=${ALLOW_EXISTING_CACHE:-0}"
echo "config -> decode_timing=$DECODE_TIMING timing_sync=$DECODE_TIMING_SYNC timing_skip=$DECODE_TIMING_SKIP_FIRST timing_step_skip=$DECODE_TIMING_STEP_SKIP_FIRST timing_step_every=$DECODE_TIMING_STEP_EVERY replay_trace=$CUDAGRAPH_REPLAY_TRACE replay_trace_max_lines=$CUDAGRAPH_REPLAY_TRACE_MAX_LINES"
echo "config -> source_stack=$SOURCE_STACK native_stack=$NATIVE_STACK p2p=$P2P_ACCESS ipc=unset/default worker_count=unset nic=eth0 push_ar=0 cache=$CACHE_DIR"
echo "config -> kernel_runtime=locally-rebuilt-June-complete-package preflight_suite=$KERNEL_PREFLIGHT_SUITE xpu_c=$EXPECTED_XPU_C_SHA256 grouped=$EXPECTED_GROUPED_SHA256 gdn=$EXPECTED_GDN_SHA256"
echo "config -> selector=level_zero:0,1 affinity=0,1 inductor_cache=/vllm_cache/torchinductor"
printf '%s\n' "${runtime_hashes[@]}"
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo "result -> exact-control CPU-only identity preflight passed"
  exit 0
fi

set +e
env -u CCL_ZE_IPC_EXCHANGE -u CCL_WORKER_COUNT \
  IMG="$IMG" \
  CKPT=/models/qwen3.6-35b-a3b/quark-w8a8-int8 \
  SERVED=qwen36-35b-a3b-quark-w8a8-int8-s2b-control \
  NAME="$NAME" \
  PORT="$PORT" \
  EXACT_STEVE_CC=1 \
  PUSH_AR=0 \
  P2PACCESS="$P2P_ACCESS" \
  MAXLEN=32768 \
  MAXSEQS=24 \
  UTIL=0.90 \
  PREFIXCACHE=0 \
  FORENSIC_VLLM_SRC="$FORENSIC_VLLM_SRC_VALUE" \
  FORENSIC_SITECUSTOMIZE_HOST="$FORENSIC_SITECUSTOMIZE_VALUE" \
  FORENSIC_FUSED_MOE_INTERFACE_HOST="$FORENSIC_FUSED_MOE_INTERFACE_VALUE" \
  XPU_KERNEL_RUNTIME_HOST="$JUNE_RUNTIME" \
  CACHE_DIR_HOST="$CACHE_DIR" \
  B70_LOGDIR="$RESULT_DIR" \
  HEALTH_STALL="$STALL_TIMEOUT" \
  B70_EXTRA_ENV="ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ZE_AFFINITY_MASK=0,1 TORCHINDUCTOR_CACHE_DIR=/vllm_cache/torchinductor VLLM_XPU_INT8_MOE_MIXED_WORKSPACE=$MIXED_WORKSPACE B70_QWEN36_MOE_TRACE=$MOE_TRACE B70_QWEN36_ALLREDUCE_TRACE=$ALLREDUCE_TRACE B70_QWEN36_ALLREDUCE_TRACE_SYNC=$ALLREDUCE_TRACE_SYNC B70_QWEN36_ALLREDUCE_TRACE_MAX_CALLS=$ALLREDUCE_TRACE_MAX_CALLS B70_QWEN36_ALLREDUCE_PROFILE_FENCE_MIN_ROWS=$PROFILE_FENCE_MIN_ROWS B70_QWEN36_ALLREDUCE_PROFILE_FENCE_STAGES=$PROFILE_FENCE_STAGES B70_QWEN36_JUNE_PROFILE_FENCE_MIN_ROWS=$PROFILE_FENCE_MIN_ROWS VLLM_XPU_DECODE_TIMING_ALLOW=$DECODE_TIMING VLLM_XPU_DECODE_TIMING=$DECODE_TIMING VLLM_XPU_DECODE_TIMING_SYNC=$DECODE_TIMING_SYNC VLLM_XPU_DECODE_TIMING_RANK=0 VLLM_XPU_DECODE_TIMING_SUMMARY=$DECODE_TIMING VLLM_XPU_DECODE_TIMING_STEP_SUMMARY=$DECODE_TIMING VLLM_XPU_DECODE_TIMING_SKIP_FIRST=$DECODE_TIMING_SKIP_FIRST VLLM_XPU_DECODE_TIMING_STEP_SKIP_FIRST=$DECODE_TIMING_STEP_SKIP_FIRST VLLM_XPU_DECODE_TIMING_STEP_EVERY=$DECODE_TIMING_STEP_EVERY VLLM_XPU_DECODE_TIMING_PRINT_EVERY=0 VLLM_XPU_CUDAGRAPH_REPLAY_TRACE_FILE=$REPLAY_TRACE_FILE_VALUE VLLM_XPU_CUDAGRAPH_REPLAY_TRACE_RANK=0 VLLM_XPU_CUDAGRAPH_REPLAY_TRACE_MAX_LINES=$CUDAGRAPH_REPLAY_TRACE_MAX_LINES" \
  I_KNOW_P2P_WEDGES=1 \
  bash "$SCRIPT_DIR/serve_qwen36_s2b_control.sh" run 2>&1 | tee "$RUN_LOG"
serve_rc="${PIPESTATUS[0]}"
set -e

# The shared run dispatcher tears down even after a failed phase and can mask
# that phase's status. Require the success artifacts and semantic evidence
# explicitly instead of trusting serve_rc alone.
metric_path="$(sed -n 's/^host_artifact=//p' "$RUN_LOG" | tail -n 1)"
json_canary_path="$(sed -n 's/^host_json_canary=//p' "$RUN_LOG" | tail -n 1)"
color_canary_path="$(sed -n 's/^host_color_canary=//p' "$RUN_LOG" | tail -n 1)"
[ "$serve_rc" = 0 ] || {
  echo "Exact control launcher returned $serve_rc" >&2
  exit "$serve_rc"
}
[ -n "$metric_path" ] && [ -f "$metric_path" ] || {
  echo "Exact control did not produce its metric artifact" >&2
  exit 1
}
for canary_path in "$json_canary_path" "$color_canary_path"; do
  [ -n "$canary_path" ] && [ -f "$canary_path" ] || {
    echo "Exact control did not complete both 16/16 repeat canaries" >&2
    exit 1
  }
done
[ -f "$SERVER_LOG" ] || {
  echo "Exact control did not preserve its server log: $SERVER_LOG" >&2
  exit 1
}

# Preserve completed client artifacts before inspecting server-side evidence.
# A rejected route is still a useful measured control and must remain paired
# with the log that caused the evidence gate to reject it.
cp "$metric_path" "$RESULT_DIR/steve_metric.json"
cp "$json_canary_path" "$RESULT_DIR/json_repeat16.json"
cp "$color_canary_path" "$RESULT_DIR/color_repeat16.json"
if [ "$CUDAGRAPH_REPLAY_TRACE" = 1 ]; then
  runtime_trace_dir="/mnt/vm_8tb/b70/results/logs/$NAME"
  compgen -G "$runtime_trace_dir/cudagraph_replay_rank*.jsonl" >/dev/null || {
    echo "CUDAGraph replay tracing produced no artifact" >&2
    exit 1
  }
  cp "$runtime_trace_dir"/cudagraph_replay_rank*.jsonl "$RESULT_DIR/"
fi

if [ "$SOURCE_STACK" = june-e190 ]; then
  required_markers=(
    '[qwen36-june-source] source_stack=e190923b'
    '[qwen36-june-source] profile clone complete'
    'Selected XPUInt8ScaledMMLinearKernel for QuarkW8A8Int8'
    'Using XPU Int8 MoE backend'
    'Asynchronous scheduling is enabled'
    'Graph capturing finished'
    "'splitting_ops': ['vllm::unified_attention_with_output'"
    "'use_inductor_graph_partition': False"
  )
else
  required_markers=(
    'restored June clone-safe custom all-reduce contract'
    'restored June prefill-replay capture contract'
    'restored June native XPU INT8 MoE route'
    'bridged June base MoE call ABI with mixed workspace off'
    'Selected XPUInt8ScaledMMLinearKernel for QuarkW8A8Int8'
    'Using XPU Int8 MoE backend'
    'kernel package=/opt/june-runtime/vllm_xpu_kernels/_xpu_C.abi3.so grouped_w8a8=_xpu_C::cutlass_grouped_gemm_w8a8_int8_interface'
    'Asynchronous scheduling is enabled'
    'Graph capturing finished'
    "'splitting_ops': ['vllm::unified_attention_with_output'"
    "'use_inductor_graph_partition': False"
  )
fi
for required in "${required_markers[@]}"; do
  rg -Fq "$required" "$SERVER_LOG" || {
    echo "Missing required server evidence: $required" >&2
    exit 1
  }
done
if rg -qi 'DEVICE_LOST|OUT_OF_RESOURCES|UR_RESULT_ERROR|output tensor.*alias|aliasing an input tensor|Triton kernel JIT compilation during inference: fused_moe_kernel' "$SERVER_LOG" "$RUN_LOG"; then
  echo "Fatal device, alias, or request-time MoE JIT marker found in exact-control evidence" >&2
  exit 1
fi
rg -Fq '"failures": []' "$RUN_LOG" || {
  echo "Semantic probe did not report an empty failure list" >&2
  exit 1
}
collective_cache_pattern='s2b_all_reduce_clone'
[ "$SOURCE_STACK" = june-e190 ] && collective_cache_pattern='vllm.*all_reduce'
rg -a -l -q "$collective_cache_pattern" "$CACHE_DIR/torch_compile_cache" || {
  echo "Persisted Inductor cache has no compiled collective evidence for $SOURCE_STACK" >&2
  exit 1
}
if [ "$DECODE_TIMING" = 1 ]; then
  python3 "$SCRIPT_DIR/summarize_qwen36_decode_timing.py" \
    --server-log "$SERVER_LOG" \
    --timing-sync "$DECODE_TIMING_SYNC" \
    --output "$RESULT_DIR/decode_timing_summary.json" \
    "$RESULT_DIR"/cudagraph_replay_rank*.jsonl
  python3 "$SCRIPT_DIR/compare_xpu_decode_timing.py" \
    --candidate "$RESULT_DIR/decode_timing_summary.json" \
    --reference "$STEVE_TIMING_REFERENCE" \
    --rank 0 \
    --output "$RESULT_DIR/decode_timing_vs_steve.json"
fi

python3 - \
  "$RESULT_DIR/steve_metric.json" \
  "$json_canary_path" \
  "$color_canary_path" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
data = json.loads(path.read_text())
assert data["model"] == "qwen36-35b-a3b-quark-w8a8-int8-s2b-control"
record = data["server_model_record"]
assert record["id"] == data["model"]
assert record["root"] == "/models/qwen3.6-35b-a3b/quark-w8a8-int8"
assert record["max_model_len"] == 32768
assert data["prompt_tokens_actual"] == 498
assert data["output_tokens_actual"] == 512
assert data["printable_ascii_fraction"] >= 0.98
assert data["tok_s_out_client_after_first_chunk_corrected"] > 0
assert data["output_token_ids_source"] == "retokenized_text"
assert len(data["output_token_ids"]) > 0
deltas = data["vllm_metric_deltas"]
assert deltas["prompt_tokens"] == 498.0
assert deltas["generation_tokens"] == 512.0
for canary_path, case, prompt_tokens in (
    (Path(sys.argv[2]), "json", 36),
    (Path(sys.argv[3]), "color", 30),
):
    canary = json.loads(canary_path.read_text())
    assert canary["protocol"] == "steve-fixed-chatml-repeat-canary"
    assert canary["model"] == data["model"]
    assert canary["server_model_record"]["root"] == record["root"]
    assert canary["server_model_record"]["max_model_len"] == 32768
    assert canary["case"] == case
    assert len(canary["prompt_token_ids"]) == prompt_tokens
    assert canary["repeats_requested"] == 16
    assert canary["repeats_completed"] == 16
    assert canary["pass_all"] is True
    assert canary["mismatch_count"] == 0
target = 85.86911405999231
speed = data["tok_s_out_client_after_first_chunk_corrected"]
ratio = speed / target
print(
    "result -> corrected_tok_s="
    f"{speed:.6f} steve_tp2_ratio={ratio:.6f} "
    f"ttft_ms={data['ttft_ms_client']:.3f} "
    f"decode_s={deltas['decode_sum_s']:.6f} artifact={path}"
)
print(
    "verdict -> speed="
    + ("within_5pct" if ratio >= 0.95 else "below_5pct_band")
    + " coherence=json16/16,color16/16"
)
PY
sha256sum \
  "$RUN_LOG" \
  "$SERVER_LOG" \
  "$RESULT_DIR/steve_metric.json" \
  "$json_canary_path" \
  "$color_canary_path"
echo "verdict -> exact control evidence complete; post-teardown health governs recovery"
