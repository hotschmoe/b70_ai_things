#!/usr/bin/env bash
# xpu-run.sh -- run a command inside the vLLM-flashnext XPU container.
#
# Other agents: this is the only supported way to touch a B70 with the
# flashnext venv. It takes the bin/gpu-run lease, bind-mounts /dev/dri,
# and puts the host Level Zero loader (libze1 1.28.2) and the host GPU
# UMD (intel-opencl-icd / libze-intel-gpu1 26.22) ahead of the pip oneAPI
# libraries that live in the venv. A loader check (ldd + dlopen) runs
# before the command and refuses to continue if a pip/image loader wins.
#
# Usage:
#   vllm/flashnext/harness/xpu-run.sh --card 1 -- python -c 'import torch; print(torch.xpu.device_count())'
#   vllm/flashnext/harness/xpu-run.sh --card 1 --port 18081 --name vllm-fn-ornith -- \
#       vllm serve /path/to/model --served-model-name hotschmoe-dev alias --port 18081
#   vllm/flashnext/harness/xpu-run.sh --card both -- ...     # TP/EP=2; no ZE_AFFINITY_MASK
#   vllm/flashnext/harness/xpu-run.sh --check                # image + ldd, no GPU, no lease
#
# The process holds the gpu-run lease until the command exits. For a
# long serve, background this script (the container stays in the
# foreground under the lease). Killing this script stops the container.
#
# Flags (before the command; "--" ends flags):
#   --card 0|1|both   Required except for --check. 0 or 1 locks that card
#                     and sets ZE_AFFINITY_MASK. both locks both cards and
#                     leaves every device visible (TP=2).
#   --name NAME       Docker container name. Default vllm-fn-<card>-<pid>.
#   --port PORT       Publish PORT on the host and in the container.
#   --shm-size SIZE   Default 16g, or 32g for --card both.
#   --memory SIZE     Container RAM cap. Default 90g (no extra swap).
#   --env KEY=VAL     Extra environment. Repeatable. Put after the built-ins
#                     so it can override them.
#   --workdir DIR     Default: repo root.
#   --check           Build the runtime image if needed, run the loader
#                     check with no /dev/dri and no gpu-run lease, then exit.
#   --skip-loader-check
#                     Do not ldd/dlopen libze_loader before the command.
#   -h | --help       Print this header.
#
# What the container has:
#   image vllm-flashnext-xpu:py312  (ubuntu 24.04 + python3.12; built on
#                                    first use from Dockerfile.runtime)
#   venv  /opt/venv  and the host path
#         /mnt/vm_8tb/b70/build/vllm-flashnext/venv
#         (console-script shebangs are #!/opt/venv/bin/python)
#   mounts  this repo, /mnt/cache, /mnt/vm_8tb/b70, /dev/dri, /dev/dri/by-path
#   user    host uid/gid plus the host render group
#   LD_LIBRARY_PATH
#         /host-umd then /usr/local/lib then venv/lib then torch/lib.
#         /host-umd has host libze_loader.so.1.28.2, libze_intel_gpu,
#         libigdgmm, libigdrcl, and the host IGC stack (libigc.so.2,
#         libigdfcl.so.2, libiga64.so.2, libopencl-clang2.so.17).
#   The same host Level Zero files are also mounted on
#   /usr/lib/x86_64-linux-gnu because libze_loader searches that path for
#   libze_intel_gpu.so.1 by name. Host IGC is mounted at /usr/local/lib
#   because NEO dlopens libigc.so.2 from that directory on the first
#   device allocation. Without it, GmmResourceInfo aborts in
#   resource_info.cpp.
#   gcc/g++ and python3.12-dev are in the image. Triton XPU also needs
#   icpx; xpu-run prepends the on-disk oneAPI 2026.1 compiler
#   (steve-repro qwen38 oneapi-root .../compiler/2026.1/bin) to PATH
#   when that binary is present. Level Zero headers from libze-dev
#   1.28.2 (same as the host loader) are mounted at /opt/level-zero.
#   The linker name libze_loader.so is the host SONAME bytes.
#   ZE_AFFINITY_MASK from --card. ONEAPI_DEVICE_SELECTOR is left unset
#   on purpose (setting both double-filters and hides the card).
#   PYTHONPATH includes vllm/flashnext/harness/patches so sitecustomize.py
#   there is imported.
#   Caches and TMPDIR are under /mnt/vm_8tb/b70/build/vllm-flashnext
#   (host /tmp is tmpfs). HF_HOME is /mnt/vm_8tb/b70/hf_cache.
#
# Env honored from the caller:
#   B70_AGENT             lease label (default: xpu-run)
#   B70_GPU_LOCK_TIMEOUT  seconds to wait for the lease (default here: 0,
#                         wait until the card is free). gpu-run's own
#                         default of 600 is overridden only when unset.
#
# Do not pass --gpus. Do not source oneAPI setvars inside the command
# unless you put /host-umd back at the front of LD_LIBRARY_PATH.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
VENV=/mnt/vm_8tb/b70/build/vllm-flashnext/venv
B70=/mnt/vm_8tb/b70
IMAGE=vllm-flashnext-xpu:py312
DOCKERFILE="$HERE/Dockerfile.runtime"
ENTRYPOINT="$HERE/xpu-entrypoint.sh"

CARD=""
NAME=""
PORT=""
SHM=""
MEMORY=90g
WORKDIR="$REPO"
CHECK=0
SKIP_LOADER=0
EXTRA_ENV=()
CMD=()

usage() {
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --card)
      CARD="${2:-}"
      shift 2
      ;;
    --card=*)
      CARD="${1#--card=}"
      shift
      ;;
    --name)
      NAME="${2:-}"
      shift 2
      ;;
    --name=*)
      NAME="${1#--name=}"
      shift
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --port=*)
      PORT="${1#--port=}"
      shift
      ;;
    --shm-size)
      SHM="${2:-}"
      shift 2
      ;;
    --shm-size=*)
      SHM="${1#--shm-size=}"
      shift
      ;;
    --memory)
      MEMORY="${2:-}"
      shift 2
      ;;
    --memory=*)
      MEMORY="${1#--memory=}"
      shift
      ;;
    --env)
      EXTRA_ENV+=("${2:-}")
      shift 2
      ;;
    --env=*)
      EXTRA_ENV+=("${1#--env=}")
      shift
      ;;
    --workdir)
      WORKDIR="${2:-}"
      shift 2
      ;;
    --workdir=*)
      WORKDIR="${1#--workdir=}"
      shift
      ;;
    --check)
      CHECK=1
      shift
      ;;
    --skip-loader-check)
      SKIP_LOADER=1
      shift
      ;;
    --)
      shift
      CMD=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      CMD=("$@")
      break
      ;;
  esac
done

if [[ "$CHECK" != 1 ]]; then
  case "$CARD" in
    0|1|both) ;;
    *)
      echo "xpu-run: --card 0|1|both is required (or use --check)" >&2
      exit 2
      ;;
  esac
  if [[ ${#CMD[@]} -eq 0 ]]; then
    echo "xpu-run: no command given" >&2
    exit 2
  fi
fi

if [[ ! -x "$VENV/bin/python" && ! -L "$VENV/bin/python" ]]; then
  echo "xpu-run: venv python missing at $VENV" >&2
  exit 1
fi
if [[ ! -x "$ENTRYPOINT" ]]; then
  chmod +x "$ENTRYPOINT"
fi

# Host UMD files. Names are the sonames the loader dlopens.
HOST_LIB=/usr/lib/x86_64-linux-gnu
declare -a HOST_FILES=(
  "$HOST_LIB/libze_loader.so.1.28.2"
  "$HOST_LIB/libze_loader.so.1"
  "$HOST_LIB/libze_intel_gpu.so.1.15.38646"
  "$HOST_LIB/libze_intel_gpu.so.1"
  "$HOST_LIB/libigdgmm.so.12.10.0"
  "$HOST_LIB/libigdgmm.so.12"
  "$HOST_LIB/intel-opencl/libigdrcl.so"
)
for f in "${HOST_FILES[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "xpu-run: missing host library $f" >&2
    exit 1
  fi
done
LOADER_BYTES=$(stat -c %s "$HOST_LIB/libze_loader.so.1.28.2")

# NEO dlopens these by soname (and looks in /usr/local/lib) on the first
# device allocation. They are not NEEDED by libze_intel_gpu, so ldd misses
# them. The host copies live in /usr/local/lib (IGC 2.36 next to UMD 26.22).
HOST_LOCAL=/usr/local/lib
declare -a HOST_IGC_FILES=()
add_igc() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "xpu-run: missing host IGC library $path" >&2
    exit 1
  fi
  HOST_IGC_FILES+=("$path")
  if [[ -L "$path" ]]; then
    HOST_IGC_FILES+=("$(readlink -f "$path")")
  fi
}
add_igc "$HOST_LOCAL/libigc.so.2"
add_igc "$HOST_LOCAL/libigdfcl.so.2"
add_igc "$HOST_LOCAL/libiga64.so.2"
add_igc "$HOST_LOCAL/libopencl-clang2.so.17"
IGC_BYTES=$(stat -c %s "$(readlink -f "$HOST_LOCAL/libigc.so.2")")

ensure_image() {
  local rebuild=0
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    rebuild=1
  else
    local created_epoch docker_epoch
    created_epoch=$(date -d "$(docker image inspect -f '{{.Created}}' "$IMAGE")" +%s)
    docker_epoch=$(date -r "$DOCKERFILE" +%s)
    if [[ "$docker_epoch" -gt "$created_epoch" ]]; then
      rebuild=1
    fi
  fi
  if [[ "$rebuild" == 0 ]]; then
    return 0
  fi
  echo "xpu-run: building $IMAGE (python3.12 userspace, no GPU UMD)" >&2
  docker build -t "$IMAGE" -f "$DOCKERFILE" "$HERE"
}

# Bind one host file at a container path. Docker stores the target bytes
# of a symlink, so the soname path and the versioned path are both real
# files with the host content.
mount_lib() {
  local src="$1"
  local dest="$2"
  DOCKER_ARGS+=(-v "${src}:${dest}:ro")
}

build_docker_args() {
  local with_gpu="$1"
  DOCKER_ARGS=(
    docker run --rm --init
    --name "$NAME"
    --user "$(id -u):$(id -g)"
    --group-add "$RENDER_GID"
    --ipc=host
    --shm-size "$SHM"
    --memory "$MEMORY"
    --memory-swap "$MEMORY"
    --ulimit memlock=-1
    --ulimit nofile=1048576:1048576
    --pids-limit -1
    -v "${VENV}:/opt/venv"
    -v "${VENV}:${VENV}"
    -v "${REPO}:${REPO}"
    -v /mnt/cache:/mnt/cache
    -v "${B70}:${B70}"
    -v "${ENTRYPOINT}:/opt/xpu-entrypoint.sh:ro"
    -w "$WORKDIR"
    -e XPU_RUN_CARD="$CARD"
    -e XPU_HOST_LOADER_BYTES="$LOADER_BYTES"
    -e XPU_HOST_IGC_BYTES="$IGC_BYTES"
    -e XPU_RUN_SKIP_LOADER_CHECK="$SKIP_LOADER"
    -e VLLM_TARGET_DEVICE=xpu
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn
    -e HOME="${B70}/build/vllm-flashnext/home"
    -e HF_HOME="${B70}/hf_cache"
    -e HF_HUB_DISABLE_TELEMETRY=1
    -e XDG_CACHE_HOME="${B70}/build/vllm-flashnext/cache/xdg"
    -e VLLM_CACHE_ROOT="${B70}/build/vllm-flashnext/cache/vllm"
    -e TRITON_CACHE_DIR="${B70}/build/vllm-flashnext/cache/triton"
    -e SYCL_CACHE_DIR="${B70}/build/vllm-flashnext/cache/sycl"
    -e SYCL_CACHE_PERSISTENT=1
    -e TMPDIR="${B70}/build/vllm-flashnext/tmp"
    -e OCL_ICD_FILENAMES=/host-umd/libigdrcl.so
    -e LEVEL_ZERO_V1_SDK_PATH=/opt/level-zero
    -e PYTHONUNBUFFERED=1
  )
  local icpx_bin="/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1/bin"
  if [[ -x "${icpx_bin}/icpx" ]]; then
    DOCKER_ARGS+=(-e "XPU_ICPX_BIN=${icpx_bin}")
  else
    echo "xpu-run: warning: oneAPI 2026.1 icpx missing at ${icpx_bin}" >&2
    echo "xpu-run: warning: Triton XPU kernels will fail to compile" >&2
  fi
  local ze_sdk="${B70}/build/vllm-flashnext/level-zero-sdk"
  if [[ ! -f "${ze_sdk}/include/level_zero/ze_api.h" ]]; then
    echo "xpu-run: missing ${ze_sdk}/include/level_zero/ze_api.h" >&2
    exit 1
  fi
  DOCKER_ARGS+=(-v "${ze_sdk}:/opt/level-zero:ro")
  # -lze_loader wants the unversioned name. The host package ships the SONAME.
  mount_lib "$HOST_LIB/libze_loader.so.1" "/host-umd/libze_loader.so"
  mount_lib "$HOST_LIB/libze_loader.so.1" "/usr/lib/x86_64-linux-gnu/libze_loader.so"
  local src base
  for src in "${HOST_FILES[@]}"; do
    base=$(basename "$src")
    mount_lib "$src" "/host-umd/${base}"
    if [[ "$base" == "libigdrcl.so" ]]; then
      # Loader search path is the multiarch lib dir, not the opencl subdir.
      # OpenCL ICD is selected by OCL_ICD_FILENAMES above.
      continue
    fi
    mount_lib "$src" "/usr/lib/x86_64-linux-gnu/${base}"
  done
  for src in "${HOST_IGC_FILES[@]}"; do
    base=$(basename "$src")
    mount_lib "$src" "/host-umd/${base}"
    mount_lib "$src" "/usr/local/lib/${base}"
  done
  if [[ "$with_gpu" == 1 ]]; then
    DOCKER_ARGS+=(
      --device /dev/dri
      -v /dev/dri/by-path:/dev/dri/by-path
    )
  else
    DOCKER_ARGS+=(--network none)
  fi
  if [[ -n "$PORT" ]]; then
    DOCKER_ARGS+=(-p "${PORT}:${PORT}")
  fi
  local kv
  for kv in "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}"; do
    DOCKER_ARGS+=(-e "$kv")
  done
  DOCKER_ARGS+=(
    --entrypoint /opt/xpu-entrypoint.sh
    "$IMAGE"
  )
  if [[ ${#CMD[@]} -gt 0 ]]; then
    DOCKER_ARGS+=("${CMD[@]}")
  fi
}

mkdir -p \
  "${B70}/build/vllm-flashnext/home" \
  "${B70}/build/vllm-flashnext/tmp" \
  "${B70}/build/vllm-flashnext/cache/xdg" \
  "${B70}/build/vllm-flashnext/cache/vllm" \
  "${B70}/build/vllm-flashnext/cache/triton" \
  "${B70}/build/vllm-flashnext/cache/sycl" \
  "${B70}/build/vllm-flashnext/logs"

RENDER_GID=$(getent group render | cut -d: -f3)
if [[ -z "$RENDER_GID" ]]; then
  echo "xpu-run: render group not found" >&2
  exit 1
fi

ensure_image

if [[ "$CHECK" == 1 ]]; then
  NAME="${NAME:-vllm-fn-check-$$}"
  CARD="${CARD:-both}"
  SHM="${SHM:-1g}"
  MEMORY="${MEMORY:-4g}"
  SKIP_LOADER="${SKIP_LOADER}"
  build_docker_args 0
  echo "xpu-run: --check (no /dev/dri, no gpu-run lease)" >&2
  "${DOCKER_ARGS[@]}"
  exit $?
fi

if [[ -z "$SHM" ]]; then
  if [[ "$CARD" == "both" ]]; then
    SHM=32g
  else
    SHM=16g
  fi
fi
if [[ -z "$NAME" ]]; then
  NAME="vllm-fn-${CARD}-$$"
fi

# A live container with this name would make docker run fail after the
# lease is held. Remove a stopped leftover; refuse a running one.
if docker inspect "$NAME" >/dev/null 2>&1; then
  state=$(docker inspect -f '{{.State.Status}}' "$NAME")
  if [[ "$state" == "running" ]]; then
    echo "xpu-run: container $NAME is already running" >&2
    exit 1
  fi
  docker rm -f "$NAME" >/dev/null
fi

build_docker_args 1

export B70_AGENT="${B70_AGENT:-xpu-run}"
if [[ -z "${B70_GPU_LOCK_TIMEOUT+x}" ]]; then
  export B70_GPU_LOCK_TIMEOUT=0
fi

GPU_RUN=("$REPO/bin/gpu-run")
if [[ "$CARD" == "0" || "$CARD" == "1" ]]; then
  GPU_RUN+=(--card "$CARD")
fi

cleanup() {
  # docker run --rm is PID 1's child. If we are killed while it is up,
  # stop the named container so the card is not left busy without a lease.
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker stop -t 20 "$NAME" >/dev/null 2>&1 || docker kill "$NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "xpu-run: card=${CARD} name=${NAME} mem=${MEMORY} shm=${SHM} port=${PORT:-none}" >&2
# gpu-run execs the command and waits. Disable our EXIT trap's stop once
# docker itself is the foreground child: an EXIT from a clean return would
# race docker --rm. The trap still covers INT/TERM during the wait.
set +e
"${GPU_RUN[@]}" "${DOCKER_ARGS[@]}"
rc=$?
set -e
trap - EXIT INT TERM
exit "$rc"
