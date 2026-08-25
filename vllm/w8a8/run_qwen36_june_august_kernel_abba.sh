#!/usr/bin/env bash
# Single-card, fresh-process June/August W8A8 kernel A-B-B-A.
# Run only through: ./bin/gpu-run --card 0 bash <this-script>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMG="intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94"
JUNE_RUNTIME="/mnt/vm_8tb/b70/steve-repro/june-xpuc-bmg-g21-a0-20260825/runtime-candidate"
SUITE="${SUITE:-quant}"
PROFILE="${PROFILE:-0}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-/mnt/vm_8tb/b70/results/logs/qwen36_kernel_abba_${SUITE}_${STAMP}}"
AUGUST_RUNTIME="${AUGUST_RUNTIME:-}"
AUGUST_HASHES_JSON="${AUGUST_HASHES_JSON:-}"
RUNNER="$SCRIPT_DIR/qwen36_june_august_kernel_arm.py"
SUMMARY="$SCRIPT_DIR/qwen36_kernel_abba_summary.py"
JUNE_HASHES_JSON='{"_C.abi3.so":"5717476461048b5056a92926f2a52d73c121f69bdc75de22fd52720fb65b3007","_moe_C.abi3.so":"ea4c20a8dff49fc07fd799d5a2a47e8b24266a256425b41e337f852492ee3c1b","_xpu_C.abi3.so":"2d931484ee0aadd4c9fb6abf494e147a5275210a216426a1eb56add0158bef0d","libgrouped_gemm_xe_2.so":"f5ddc2ee3c11dcede3a7190b69d6e0dd354bb0727be7519600abaebe9fc4cd2c","libgdn_attn_kernels_xe_2.so":"366935b172b5c9c3cb75bee5d7bfe0434f377a6317314a9a43c853b5d02fe83b"}'
PINNED_AUGUST_HASHES_JSON='{"_C.abi3.so":"5717476461048b5056a92926f2a52d73c121f69bdc75de22fd52720fb65b3007","_moe_C.abi3.so":"ea4c20a8dff49fc07fd799d5a2a47e8b24266a256425b41e337f852492ee3c1b","_xpu_C.abi3.so":"ae330affe0315a5be4ac50478cc15c7874ae6e8fa9fa71cf64d5e5dff158968b","libgrouped_gemm_xe_2.so":"7692db81b65be5fdb9d4509f2397d300276451ee9551d43cedb7963eaad70e4a","libgdn_attn_kernels_xe_2.so":"cf482fd898ef965eeac70682027fe5578d5005b5eb6c51a85664a68e151a4a02"}'

case "$SUITE" in
  quant|dense|quant-dense|grouped) ;;
  *) echo "SUITE must be quant, dense, quant-dense, or grouped" >&2; exit 2 ;;
esac
case "$PROFILE" in 0|1) ;; *) echo "PROFILE must be 0 or 1" >&2; exit 2 ;; esac
[ -f "$JUNE_RUNTIME/vllm_xpu_kernels/_xpu_C.abi3.so" ] || {
  echo "Missing June runtime: $JUNE_RUNTIME" >&2
  exit 1
}
if [ "$SUITE" = grouped ] && [ -z "$AUGUST_RUNTIME" ]; then
  echo "Pinned August _xpu_C has no grouped W8A8 operator." >&2
  echo "Set AUGUST_RUNTIME and AUGUST_HASHES_JSON to a complete August rebuild." >&2
  exit 2
fi
if [ -n "$AUGUST_RUNTIME" ]; then
  [ -n "$AUGUST_HASHES_JSON" ] || {
    echo "AUGUST_RUNTIME requires AUGUST_HASHES_JSON" >&2
    exit 2
  }
  [ -f "$AUGUST_RUNTIME/vllm_xpu_kernels/_xpu_C.abi3.so" ] || {
    echo "Invalid AUGUST_RUNTIME: $AUGUST_RUNTIME" >&2
    exit 2
  }
else
  AUGUST_HASHES_JSON="$PINNED_AUGUST_HASHES_JSON"
fi

mkdir -p "$RESULT_DIR"
profile_arg=()
[ "$PROFILE" = 0 ] || profile_arg=(--profile)

post_health() {
  local run_rc=$? health_rc
  trap - EXIT
  set +e
  "$REPO_ROOT/bin/xpu-health" --card 0 2>&1 | tee "$RESULT_DIR/health_post.log"
  health_rc="${PIPESTATUS[0]}"
  set -e
  if [ "$run_rc" = 0 ] && [ "$health_rc" != 0 ]; then
    run_rc="$health_rc"
  fi
  exit "$run_rc"
}
trap post_health EXIT
"$REPO_ROOT/bin/xpu-health" --card 0 2>&1 | tee "$RESULT_DIR/health_pre.log"

run_arm() {
  local label="$1" package="$2" output="$RESULT_DIR/$1.json"
  local package_root hashes
  local mounts=(
    -v "$RUNNER:/opt/kernel_arm.py:ro"
    -v "$RESULT_DIR:/opt/out"
  )
  local envs=(
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0
    -e ZE_AFFINITY_MASK=0
  )
  if [ "$package" = june ]; then
    package_root=/opt/runtime/vllm_xpu_kernels
    hashes="$JUNE_HASHES_JSON"
    mounts+=( -v "$JUNE_RUNTIME:/opt/runtime:ro" )
    envs+=( -e PYTHONPATH=/opt/runtime )
  elif [ -n "$AUGUST_RUNTIME" ]; then
    package_root=/opt/runtime/vllm_xpu_kernels
    hashes="$AUGUST_HASHES_JSON"
    mounts+=( -v "$AUGUST_RUNTIME:/opt/runtime:ro" )
    envs+=( -e PYTHONPATH=/opt/runtime )
  else
    package_root=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
    hashes="$AUGUST_HASHES_JSON"
  fi
  echo "command -> arm=$label package=$package suite=$SUITE output=$output"
  docker run --rm --entrypoint python \
    --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    "${mounts[@]}" "${envs[@]}" "$IMG" \
    /opt/kernel_arm.py \
    --arm "$label" \
    --suite "$SUITE" \
    --expected-package-root "$package_root" \
    --expected-hashes "$hashes" \
    "${profile_arg[@]}" \
    --output "/opt/out/$label.json"
}

echo "config -> image=$IMG suite=$SUITE profile=$PROFILE selector=level_zero:0 order=june-a1,august-b1,august-b2,june-a2"
echo "config -> june=$JUNE_RUNTIME august=${AUGUST_RUNTIME:-pinned-image-package} result_dir=$RESULT_DIR"
run_arm june-a1 june
run_arm august-b1 august
run_arm august-b2 august
run_arm june-a2 june
python3 "$SUMMARY" \
  --a1 "$RESULT_DIR/june-a1.json" \
  --b1 "$RESULT_DIR/august-b1.json" \
  --b2 "$RESULT_DIR/august-b2.json" \
  --a2 "$RESULT_DIR/june-a2.json" \
  --output "$RESULT_DIR/summary.json"
sha256sum "$RESULT_DIR"/*.json
echo "verdict -> A-B-B-A evidence complete: $RESULT_DIR/summary.json"
