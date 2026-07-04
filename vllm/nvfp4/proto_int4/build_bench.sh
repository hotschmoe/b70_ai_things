#!/usr/bin/env bash
set -o pipefail
IMG=vllm-xpu-env:int8g-v0240
SRC=/mnt/vm_8tb/github/b70_ai_things/vllm/nvfp4/proto_int4
OUT=/mnt/vm_8tb/b70/int4_dpas_build
docker run --rm -v "$SRC":/src -v "$OUT":/out --entrypoint bash "$IMG" -c '
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
for p in 8 4 2; do
  n=s$p
  icpx -fsycl -std=c++17 -O2 -DPREC=$p -fsycl-targets=intel_gpu_bmg_g31 \
    /src/bench.cpp -o /out/bench_$n 2>/out/bench_build_$n.err
  echo "bench_$n rc=$?"
  [ -s /out/bench_build_$n.err ] && tail -5 /out/bench_build_$n.err
done
ls -la /out/bench_s* '
