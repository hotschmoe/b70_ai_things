#!/usr/bin/env bash
# One guarded TP2 P2P transaction for the Qwen3.6 all-reduce graph contract.
# Usage: ./bin/gpu-run env I_KNOW_P2P_WEDGES=1 bash \
#   vllm/w8a8/run_qwen36_oneccl_graph_oracle.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMG="${IMG:-intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94}"
IPCX="${IPCX:-default}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/oneccl_oracle}"
RESULT_FILE="${RESULT_FILE:-$RESULT_DIR/qwen36_tp2_oneccl_${IPCX}_${STAMP}.json}"
NAME="${NAME:-qwen36_oneccl_graph_oracle}"
ONECCL_INSTALL_DIR="${ONECCL_INSTALL_DIR:-}"
EXPECTED_KERNELS_SHA256="${EXPECTED_KERNELS_SHA256:-0d549c35a558f1b216cb7d1efeaa9f86d7596ffc47b383644e075290d314f0c9}"

if [ "${I_KNOW_P2P_WEDGES:-0}" != 1 ]; then
  echo "Refusing direct P2P oracle without I_KNOW_P2P_WEDGES=1" >&2
  exit 2
fi
case "$IPCX" in
  default|pidfd) ;;
  *) echo "IPCX must be default or pidfd" >&2; exit 2 ;;
esac

mkdir -p "$RESULT_DIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true

DOCKER_MOUNTS=()
if [ -n "$ONECCL_INSTALL_DIR" ]; then
  [ -f "$ONECCL_INSTALL_DIR/lib/libccl.so.1.0" ] || {
    echo "Missing candidate libccl.so.1.0 under $ONECCL_INSTALL_DIR" >&2
    exit 2
  }
  [ -f "$ONECCL_INSTALL_DIR/lib/ccl/kernels/kernels.spv" ] || {
    echo "Missing candidate kernels.spv under $ONECCL_INSTALL_DIR" >&2
    exit 2
  }
  CCL_ROOT_CONTAINER=/opt/ccl_candidate
  DOCKER_MOUNTS+=( -v "$ONECCL_INSTALL_DIR:$CCL_ROOT_CONTAINER:ro" )
  EXPECTED_CCL_SHA256="${EXPECTED_CCL_SHA256:-43d94d43506e30096dd099b9d53b54f932be964751e92ff0cbb8d3a37fad6700}"
else
  CCL_ROOT_CONTAINER=/opt/ccl4ce
  EXPECTED_CCL_SHA256="${EXPECTED_CCL_SHA256:-542142aca8f3d318616eae0f300aaa47dc62b217831599cb1461212f8aa4dc76}"
fi

DOCKER_ENV=(
  -e CCL_ATL_TRANSPORT=ofi
  -e CCL_TOPO_P2P_ACCESS=1
  -e CCL_LOG_LEVEL=info
  -e CCL_KERNEL_PATH="$CCL_ROOT_CONTAINER/lib/ccl/kernels"
  -e FI_TCP_IFACE=eth0
  -e CCL_KVS_IFACE=eth0
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1
  -e ZE_AFFINITY_MASK=0,1
  -e LD_PRELOAD="$CCL_ROOT_CONTAINER/lib/libccl.so.1.0"
  -e LD_LIBRARY_PATH="$CCL_ROOT_CONTAINER/lib:/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib"
  -e EXPECTED_CCL_SHA256="$EXPECTED_CCL_SHA256"
  -e EXPECTED_KERNELS_SHA256="$EXPECTED_KERNELS_SHA256"
)
if [ "$IPCX" = pidfd ]; then
  DOCKER_ENV+=( -e CCL_ZE_IPC_EXCHANGE=pidfd )
else
  # Omit both variables: Steve's June Qwen35 launcher left IPC exchange and
  # worker count unset. Docker does not inherit host variables without -e.
  :
fi

echo "config -> image=$IMG shape=[4,5120] direct=256 graph=512 ipc=$IPCX ccl=$EXPECTED_CCL_SHA256 kernels=$EXPECTED_KERNELS_SHA256"
docker run --rm --name "$NAME" --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --network=host \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$SCRIPT_DIR/qwen36_oneccl_graph_oracle.py:/opt/qwen36_oneccl_graph_oracle.py:ro" \
  -v "$RESULT_DIR:/results" \
  "${DOCKER_MOUNTS[@]+"${DOCKER_MOUNTS[@]}"}" \
  "${DOCKER_ENV[@]}" \
  --entrypoint /opt/venv/bin/torchrun "$IMG" \
  --standalone --nproc-per-node=2 /opt/qwen36_oneccl_graph_oracle.py \
  --direct-iterations 256 --graph-iterations 512 \
  --output "/results/$(basename "$RESULT_FILE")"
echo "result -> $RESULT_FILE"
