#!/usr/bin/env bash
# F06a: exact corrected-image direct-P2P compiled collective transaction.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMAGE="${IMAGE:-neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81}"
EXPECTED_CCL_SHA256="${EXPECTED_CCL_SHA256:-733980ab6a6eb15d2d3da0649b92052c64a9597ced48fe9188434face5298b35}"
EXPECTED_KERNELS_SHA256="${EXPECTED_KERNELS_SHA256:-0d549c35a558f1b216cb7d1efeaa9f86d7596ffc47b383644e075290d314f0c9}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/f06a_qwen38_fp8_neural_p2p/$STAMP}"

if [ "${1:-}" != --leased ]; then
  exec "$REPO/bin/gpu-run" env \
    I_KNOW_P2P_WEDGES="${I_KNOW_P2P_WEDGES:-0}" \
    STAMP="$STAMP" RESULT_DIR="$RESULT_DIR" IMAGE="$IMAGE" \
    EXPECTED_IMAGE_ID="$EXPECTED_IMAGE_ID" \
    EXPECTED_CCL_SHA256="$EXPECTED_CCL_SHA256" \
    EXPECTED_KERNELS_SHA256="$EXPECTED_KERNELS_SHA256" \
    bash "$0" --leased
fi

if [ "${I_KNOW_P2P_WEDGES:-0}" != 1 ]; then
  echo "Refusing F06a direct P2P without I_KNOW_P2P_WEDGES=1" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
actual_image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null)"
if [ "$actual_image_id" != "$EXPECTED_IMAGE_ID" ]; then
  echo "image ID mismatch: actual=$actual_image_id expected=$EXPECTED_IMAGE_ID" >&2
  exit 1
fi

hashes="$(docker run --rm --entrypoint sha256sum "$IMAGE" \
  /opt/venv/lib/libccl.so.1.0 /opt/venv/lib/ccl/kernels/kernels.spv)"
printf '%s\n' "$hashes" >"$RESULT_DIR/runtime-hashes.txt"
printf '%s\n' "$hashes" | grep -Fqx \
  "$EXPECTED_CCL_SHA256  /opt/venv/lib/libccl.so.1.0" || exit 1
printf '%s\n' "$hashes" | grep -Fqx \
  "$EXPECTED_KERNELS_SHA256  /opt/venv/lib/ccl/kernels/kernels.spv" || exit 1

{
  echo "CONFIG -> image=$IMAGE"
  echo "CONFIG -> image_id=$actual_image_id"
  echo "CONFIG -> kernel=$(uname -r)"
  echo "CONFIG -> shape=4x5120 dtype=bf16 eager=1 compiled=10"
  echo "CONFIG -> sequence=p2p0,p2p1,p2p0"
  echo "CONFIG -> p2p1_guard=I_KNOW_P2P_WEDGES=1"
  echo "CONFIG -> result_dir=$RESULT_DIR"
} | tee "$RESULT_DIR/config.txt"

run_logged() {
  local label="$1"
  shift
  echo "COMMAND -> $label: $*" | tee -a "$RESULT_DIR/commands.txt"
  "$@" 2>&1 | tee "$RESULT_DIR/$label.log"
  return "${PIPESTATUS[0]}"
}

pre_card_rc=0
pre_p2p0_rc=0
direct_rc=0
recovery_rc=0
post_card_rc=0
post_p2p0_rc=0

run_logged pre-card "$REPO/bin/xpu-health" --img "$IMAGE" || pre_card_rc=$?
run_logged pre-p2p0 env IMG="$IMAGE" \
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout 180 || pre_p2p0_rc=$?
if [ "$pre_card_rc" -ne 0 ] || [ "$pre_p2p0_rc" -ne 0 ]; then
  echo "VERDICT -> blocked: pre-health failed card=$pre_card_rc p2p0=$pre_p2p0_rc" \
    | tee "$RESULT_DIR/verdict.txt"
  exit 1
fi

run_logged direct-p2p1 env I_KNOW_P2P_WEDGES=1 IMG="$IMAGE" \
  "$REPO/bin/xpu-collective-health" --p2p 1 --timeout 180 || direct_rc=$?

if [ "$direct_rc" -ne 0 ]; then
  echo "RESULT -> direct P2P failed rc=$direct_rc; invoking xe rebind recovery" \
    | tee -a "$RESULT_DIR/verdict.txt"
  run_logged recovery-rebind "$REPO/bin/xe-reset" --method rebind || recovery_rc=$?
fi

run_logged post-card "$REPO/bin/xpu-health" --img "$IMAGE" || post_card_rc=$?
run_logged post-p2p0 env IMG="$IMAGE" \
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout 180 || post_p2p0_rc=$?

{
  echo "RESULT -> pre_card_rc=$pre_card_rc pre_p2p0_rc=$pre_p2p0_rc"
  echo "RESULT -> direct_p2p1_rc=$direct_rc recovery_rc=$recovery_rc"
  echo "RESULT -> post_card_rc=$post_card_rc post_p2p0_rc=$post_p2p0_rc"
  if [ "$direct_rc" -eq 0 ] && [ "$post_card_rc" -eq 0 ] && \
     [ "$post_p2p0_rc" -eq 0 ]; then
    echo "VERDICT -> F06a PASS: exact corrected image crosses compiled direct P2P and remains healthy"
  else
    echo "VERDICT -> F06a FAIL: do not advance to the loaded vLLM process boundary"
  fi
} | tee -a "$RESULT_DIR/verdict.txt"

if [ "$direct_rc" -ne 0 ] || [ "$recovery_rc" -ne 0 ] || \
   [ "$post_card_rc" -ne 0 ] || [ "$post_p2p0_rc" -ne 0 ]; then
  exit 1
fi
