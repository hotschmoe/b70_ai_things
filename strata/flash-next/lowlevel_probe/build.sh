#!/usr/bin/env bash
# Offline CPU-only AOT build: no device access and no GPU lease needed.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/build"
docker run --rm --network none -u "$(id -u):$(id -g)" -v "$D:/w" -w /w \
  --entrypoint bash "${BUILD_IMAGE:-strata-ab-build:ocloc}" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
g++ -std=c++17 -O2 -Wall -Wextra ze_metric_enum.cpp -lze_loader -o build/ze_metric_enum
icpx -O2 -std=c++20 -Wall -fsycl -fsycl-targets=spir64_gen \
  -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt" \
  -fsycl-device-code-split=per_kernel readiness_probe.cpp -o build/readiness_probe
sha256sum readiness_probe.cpp pattern.hpp ze_metric_enum.cpp run_readiness.sh build/readiness_probe build/ze_metric_enum > build/sha256.txt
' 2>&1 | tee "$D/build/build.log"
