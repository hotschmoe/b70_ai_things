#!/usr/bin/env bash
# LOOP 46: bake Steve 4ceafd1 + 2dd55f38 + 44fc into intel/vllm:0.21.0-xpu.
# CPU only. Do not take the GPU lease. Do not bind-overlay (D15).
# Writes tag intel/vllm:0.21.0-xpu-s2b.
set -euo pipefail
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STEVE="${STEVE:-$ROOT/steve-s2b}"
SRC_IMG="${SRC_IMG:-intel/vllm:0.21.0-xpu}"
DST_IMG="${DST_IMG:-intel/vllm:0.21.0-xpu-s2b}"
NAME="${NAME:-loop46_s2b_bake}"
LOG="${LOG:-$ROOT/qwen38-w8a8-dspark/loop46_bake.log}"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
CCL_SO="$STEVE/oneccl-install/lib/libccl.so.1.0"
XPU_SO="$STEVE/xpu-c-install/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_SO="$STEVE/xpu-c-install/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
VLLM_SRC="$STEVE/vllm"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== LOOP 46 bake $DST_IMG $(date -u +%Y-%m-%dT%H%M%SZ) ==="
test -f "$CCL_SO" && test -f "$XPU_SO" && test -f "$GDN_SO"
test -d "$VLLM_SRC/vllm"
echo "ccl=$(sha256sum "$CCL_SO" | awk '{print $1}')"
echo "xpu=$(sha256sum "$XPU_SO" | awk '{print $1}')"
echo "gdn=$(sha256sum "$GDN_SO" | awk '{print $1}')"
echo "vllm=$(git -C "$VLLM_SRC" rev-parse HEAD)"

docker rm -f "$NAME" >/dev/null 2>&1 || true
# No --device /dev/dri. CPU copy only. AGASYNC stays up.
docker run -d --name "$NAME" --entrypoint sleep \
  -v "$STEVE:/steve:ro" \
  "$SRC_IMG" 7200
docker exec "$NAME" bash -lc '
set -euo pipefail
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
rm -rf /opt/ccl4ce
cp -a /steve/oneccl-install /opt/ccl4ce
# Torch libtorch_xpu RPATH is $ORIGIN/../../../.. = /opt/venv/lib. Overlay
# never won that search (D15). Bake the public so onto the RPATH path AND
# replace 2021.15/2021.17 so.1 so no leftover Intel so.1 can load.
for dest in \
  /opt/venv/lib/libccl.so.1.0 \
  /opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1.0 \
  /opt/intel/oneapi/ccl/2021.17/lib/libccl.so.1.0
do
  cp -f /opt/ccl4ce/lib/libccl.so.1.0 "$dest"
  chmod 755 "$dest"
done
ln -sfn libccl.so.1.0 /opt/venv/lib/libccl.so.1
ln -sfn libccl.so.1.0 /opt/venv/lib/libccl.so
# 2021.17 so.1 is already a symlink to so.1.0; refresh in case a bind left a file.
ln -sfn libccl.so.1.0 /opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1
ln -sfn libccl.so.1.0 /opt/intel/oneapi/ccl/2021.17/lib/libccl.so.1
# Public kernels.spv (matches Steve validated 0d549c35).
mkdir -p /opt/intel/oneapi/ccl/2021.17/lib/ccl/kernels
cp -f /opt/ccl4ce/lib/ccl/kernels/kernels.spv \
  /opt/intel/oneapi/ccl/2021.17/lib/ccl/kernels/kernels.spv
cp -f /steve/xpu-c-install/vllm_xpu_kernels/_xpu_C.abi3.so "$PKGD/_xpu_C.abi3.so"
cp -f /steve/xpu-c-install/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so \
  "$PKGD/libgdn_attn_kernels_xe_2.so"
# 44fc python tree over the image 8df6feb7d checkout. Skip .git.
rm -rf /opt/vllm
mkdir -p /opt/vllm
tar -C /steve/vllm --exclude .git -cf - . | tar -C /opt/vllm -xf -
echo 44fc8fde09fc311d3099dab10366b672d9142ea4 > /opt/vllm/.b70_vllm_head
echo BAKED_OK
echo -n ccl_venv=; sha256sum /opt/venv/lib/libccl.so.1.0
echo -n ccl_202117=; sha256sum /opt/intel/oneapi/ccl/2021.17/lib/libccl.so.1.0
echo -n xpu=; sha256sum "$PKGD/_xpu_C.abi3.so"
echo -n gdn=; sha256sum "$PKGD/libgdn_attn_kernels_xe_2.so"
echo -n vllm_head=; cat /opt/vllm/.b70_vllm_head
test -f /opt/vllm/vllm/_xpu_ops.py
ls -l /opt/ccl4ce/lib/libccl.so.1.0 "$PKGD/_xpu_C.abi3.so"
'
echo "=== commit $DST_IMG ==="
docker commit \
  --change 'ENV CCL_ROOT=/opt/ccl4ce' \
  --change 'LABEL b70.s2b="4ceafd1+2dd55f38+44fc8fde0"' \
  "$NAME" "$DST_IMG"
docker rm -f "$NAME"
docker image inspect "$DST_IMG" --format '{{.Id}} {{.Size}} {{.Created}}'
echo "=== bake done $(date -u +%Y-%m-%dT%H%M%SZ) ==="
