#!/usr/bin/env bash
# Inside the xpu-run container. Re-asserts the library order, checks that
# libze_loader resolves to the host mount, then execs the user command.
# Do not source oneAPI setvars here: it would put pip libraries first.
set -euo pipefail

VENV_LIB=/opt/venv/lib
TORCH_LIB=/opt/venv/lib/python3.12/site-packages/torch/lib
export PATH="/opt/venv/bin:${PATH:-/usr/bin:/bin}"
# Triton XPU compiles its Level Zero helper with icpx -fsycl. This icpx
# is oneAPI 2026.1, matching the venv's intel-sycl-rt. It is not put on
# LD_LIBRARY_PATH, so it cannot outrank the host Level Zero loader.
if [[ -n "${XPU_ICPX_BIN:-}" && -x "${XPU_ICPX_BIN}/icpx" ]]; then
  export PATH="${XPU_ICPX_BIN}:${PATH}"
fi
export LEVEL_ZERO_V1_SDK_PATH="${LEVEL_ZERO_V1_SDK_PATH:-/opt/level-zero}"
export VIRTUAL_ENV=/opt/venv
# /usr/local/lib is where the host IGC stack is mounted. NEO dlopens
# libigc.so.2 from that directory; LD_LIBRARY_PATH also covers a soname
# lookup because the container ld cache does not list these bind mounts.
export LD_LIBRARY_PATH="/host-umd:/usr/local/lib:${VENV_LIB}:${TORCH_LIB}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness/patches${PYTHONPATH:+:${PYTHONPATH}}"

# ZE_AFFINITY_MASK is the only device pin. Do not also set
# ONEAPI_DEVICE_SELECTOR to the same index: the Level Zero loader filters
# first, and SYCL would then look up an index that no longer exists.
case "${XPU_RUN_CARD:-}" in
  0|1) export ZE_AFFINITY_MASK="${XPU_RUN_CARD}" ;;
  both|"") unset ZE_AFFINITY_MASK || true ;;
esac

check_loader_override() {
  local base
  echo "xpu-run: UMD override label=${XPU_UMD_LABEL:-unknown} clang=${XPU_UMD_CLANG:-unset}"
  echo "xpu-run: LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  echo "xpu-run: ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK-<unset>} card=${XPU_RUN_CARD:-<unset>}"
  for base in libze_loader.so.1 libze_intel_gpu.so.1 libigdgmm.so.12 libigdrcl.so \
              libigc.so.2 libigdfcl.so.2 libiga64.so.2; do
    if [[ ! -e "/host-umd/${base}" ]]; then
      echo "xpu-run: FAIL override missing /host-umd/${base}" >&2
      ls -l /host-umd >&2 || true
      return 1
    fi
  done
  if [[ ! -e /usr/local/lib/libigc.so.2 || ! -e /usr/local/lib/libigdfcl.so.2 || ! -e /usr/local/lib/libiga64.so.2 ]]; then
    echo "xpu-run: FAIL override IGC is not mounted under /usr/local/lib" >&2
    ls -l /usr/local/lib >&2 || true
    return 1
  fi
  if [[ -n "${XPU_UMD_CLANG:-}" && ! -e "/usr/local/lib/${XPU_UMD_CLANG}" && ! -e "/host-umd/${XPU_UMD_CLANG}" ]]; then
    echo "xpu-run: FAIL override clang ${XPU_UMD_CLANG} is not mounted" >&2
    return 1
  fi
  echo "xpu-run: override file sizes"
  stat -c '%n %s' \
    /host-umd/libze_loader.so.1 \
    /host-umd/libze_intel_gpu.so.1 \
    /host-umd/libigdgmm.so.12 \
    /host-umd/libigdrcl.so \
    /host-umd/libigc.so.2 \
    /host-umd/libigdfcl.so.2 \
    /host-umd/libiga64.so.2
  local so
  for so in /host-umd/libze_loader.so.1 /host-umd/libze_intel_gpu.so.1 \
            /host-umd/libigc.so.2 /host-umd/libigdfcl.so.2 /host-umd/libiga64.so.2; do
    echo "=== ldd ${so} ==="
    ldd "$so"
    if ldd "$so" | grep -q 'not found'; then
      echo "xpu-run: FAIL ldd not found for ${so}" >&2
      return 1
    fi
  done
  XPU_UMD_CLANG="${XPU_UMD_CLANG:-}" /usr/bin/python3.12 - <<'PY'
import ctypes
import os
import sys

ctypes.CDLL("libze_loader.so.1")
ctypes.CDLL("libze_intel_gpu.so.1")
ctypes.CDLL("libigdgmm.so.12")
ctypes.CDLL("libigc.so.2")
ctypes.CDLL("libigdfcl.so.2")
ctypes.CDLL("libiga64.so.2")
clang = os.environ.get("XPU_UMD_CLANG", "")
if clang:
    ctypes.CDLL(clang)
seen = {}
for line in open("/proc/self/maps"):
    parts = line.split()
    if not parts:
        continue
    path = parts[-1]
    base = os.path.basename(path)
    for key in ("libze_loader", "libze_intel_gpu", "libigdgmm", "libigc.so",
                "libigdfcl", "libiga64", "libopencl-clang", "libigc-clang"):
        if key in base and key not in seen:
            seen[key] = path
            print("RESOLVED", key, path)
ok = True
allowed = ("/host-umd/", "/usr/local/lib/", "/usr/lib/x86_64-linux-gnu/")
for key in ("libze_loader", "libze_intel_gpu", "libigdgmm", "libigc.so", "libigdfcl", "libiga64"):
    path = seen.get(key, "")
    if not path.startswith(allowed):
        print("FAIL", key, "resolved outside override mounts:", path or "<missing>")
        ok = False
clang_path = seen.get("libopencl-clang", "")
if clang and not clang_path.startswith(allowed):
    print("FAIL clang resolved outside override mounts:", clang_path or "<missing>")
    ok = False
expect_loader = os.environ.get("XPU_HOST_LOADER_BYTES", "")
expect_igc = os.environ.get("XPU_HOST_IGC_BYTES", "")
loader = seen.get("libze_loader", "")
igc = seen.get("libigc.so", "")
if expect_loader and loader:
    sz = os.path.getsize(loader)
    print("xpu-run: libze_loader bytes", sz)
    if str(sz) != expect_loader:
        print("FAIL loader size", sz, "!= selected", expect_loader)
        ok = False
if expect_igc and igc:
    sz = os.path.getsize(igc)
    print("xpu-run: libigc bytes", sz)
    if str(sz) != expect_igc:
        print("FAIL libigc size", sz, "!= selected", expect_igc)
        ok = False
sys.exit(0 if ok else 1)
PY
  if [[ -d /dev/dri ]]; then
    if ! command -v sycl-ls >/dev/null 2>&1; then
      echo "xpu-run: FAIL sycl-ls not on PATH" >&2
      return 1
    fi
    echo "=== sycl-ls ==="
    sycl-ls
    echo "=== torch.xpu ==="
    /opt/venv/bin/python - <<'PY'
import torch
count = torch.xpu.device_count()
print("torch.xpu.device_count", count)
if count < 1:
    raise SystemExit("no xpu device")
print("torch.xpu.name", torch.xpu.get_device_name(0))
x = torch.ones((4, 4), device="xpu")
print("matmul_sum", float((x @ x).sum()))
print("DEVICE_QUERY_OK")
PY
  fi
  echo "xpu-run: loader check ok (UMD override ${XPU_UMD_LABEL:-} ahead of pip)"
}

check_loader() {
  if [[ "${XPU_UMD_MODE:-}" == "override" ]]; then
    check_loader_override
    return
  fi
  local loader=/host-umd/libze_loader.so.1.28.2
  local soname=/host-umd/libze_loader.so.1
  local gpu=/host-umd/libze_intel_gpu.so.1
  local igc=/host-umd/libigc.so.2
  echo "xpu-run: LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  echo "xpu-run: ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK-<unset>} card=${XPU_RUN_CARD:-<unset>}"
  if [[ ! -f "$loader" || ! -f "$soname" || ! -f "$gpu" || ! -f "$igc" ]]; then
    echo "xpu-run: FAIL host UMD or IGC is not mounted under /host-umd" >&2
    ls -l /host-umd >&2 || true
    return 1
  fi
  if [[ ! -f /usr/local/lib/libigc.so.2 || ! -f /usr/local/lib/libopencl-clang2.so.17 ]]; then
    echo "xpu-run: FAIL host IGC is not mounted under /usr/local/lib" >&2
    ls -l /usr/local/lib >&2 || true
    return 1
  fi
  echo "=== ldd ${loader} ==="
  ldd "$loader"
  echo "=== ldd ${soname} ==="
  ldd "$soname"
  echo "=== ldd ${gpu} ==="
  ldd "$gpu"
  local sz
  sz=$(stat -c %s "$loader")
  echo "xpu-run: libze_loader.so.1.28.2 bytes=${sz}"
  if [[ -n "${XPU_HOST_LOADER_BYTES:-}" && "$sz" != "${XPU_HOST_LOADER_BYTES}" ]]; then
    echo "xpu-run: FAIL loader size ${sz} != host file ${XPU_HOST_LOADER_BYTES}" >&2
    return 1
  fi
  echo "xpu-run: libze_loader / libze_intel_gpu files outside /host-umd:"
  find /opt/venv/lib /usr/lib /usr/local/lib \
      \( -name 'libze_loader*' -o -name 'libze_intel_gpu*' \) \
      ! -path '/usr/lib/x86_64-linux-gnu/libze_loader.so.1*' \
      ! -path '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1*' \
      -print 2>/dev/null || true
  /usr/bin/python3.12 - <<'PY'
import ctypes
import os
import sys

ctypes.CDLL("libze_loader.so.1")
ctypes.CDLL("libze_intel_gpu.so.1")
ctypes.CDLL("libigc.so.2")
ctypes.CDLL("libigdfcl.so.2")
ctypes.CDLL("libopencl-clang2.so.17")
seen = {}
for line in open("/proc/self/maps"):
    parts = line.split()
    if not parts:
        continue
    path = parts[-1]
    for key in ("libze_loader", "libze_intel_gpu", "libigdgmm", "libigc.so", "libigdfcl", "libopencl-clang"):
        if key in os.path.basename(path) and key not in seen:
            seen[key] = path
            print("RESOLVED", key, path)
ok = True
loader = seen.get("libze_loader", "")
gpu = seen.get("libze_intel_gpu", "")
gmm = seen.get("libigdgmm", "")
igc = seen.get("libigc.so", "")
fcl = seen.get("libigdfcl", "")
clang = seen.get("libopencl-clang", "")
if not loader.startswith("/host-umd/libze_loader.so.1.28.2") and loader != "/host-umd/libze_loader.so.1":
    # The soname mount is the same host bytes. Accept either host path.
    if not loader.startswith("/host-umd/"):
        print("FAIL libze_loader resolved outside /host-umd:", loader or "<missing>")
        ok = False
if not gpu.startswith("/host-umd/"):
    print("FAIL libze_intel_gpu resolved outside /host-umd:", gpu or "<missing>")
    ok = False
if gmm and not gmm.startswith("/host-umd/") and "/usr/lib/x86_64-linux-gnu/libigdgmm.so.12" not in gmm:
    print("FAIL libigdgmm resolved to", gmm)
    ok = False
host_igc_ok = ("/host-umd/", "/usr/local/lib/")
if not igc.startswith(host_igc_ok):
    print("FAIL libigc resolved outside host mounts:", igc or "<missing>")
    ok = False
if not fcl.startswith(host_igc_ok):
    print("FAIL libigdfcl resolved outside host mounts:", fcl or "<missing>")
    ok = False
if not clang.startswith(host_igc_ok):
    print("FAIL libopencl-clang resolved outside host mounts:", clang or "<missing>")
    ok = False
expect = os.environ.get("XPU_HOST_IGC_BYTES", "")
if expect and igc:
    sz = os.path.getsize(igc)
    print("xpu-run: libigc bytes", sz)
    if str(sz) != expect:
        print("FAIL libigc size", sz, "!= host", expect)
        ok = False
for path in (loader, gpu, gmm, igc, fcl, clang):
    if "1.32" in path or "26.27" in path:
        print("FAIL pip or image UMD won:", path)
        ok = False
sys.exit(0 if ok else 1)
PY
  echo "xpu-run: loader check ok (host libze 1.28.2 and host IGC 2 ahead of pip)"
}

if [[ "${XPU_RUN_SKIP_LOADER_CHECK:-0}" != "1" ]]; then
  check_loader
fi

if [[ $# -eq 0 ]]; then
  exit 0
fi
exec "$@"
