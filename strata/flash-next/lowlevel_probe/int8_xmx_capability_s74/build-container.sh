#!/usr/bin/env bash
# CPU-only normal -fsycl AOT. This script never executes the produced binary.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
command -v icpx sha256sum >/dev/null
[[ -w /out && -d /out/tmp ]]
export TMPDIR=/out/tmp
cd /out
sha256sum -c /src/expected-container.sha256
icpx --version > compiler-version.txt
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen
       -fsycl-default-sub-group-size=16 -fsycl-device-code-split=per_kernel
       -fp-model=precise -Wall -Wextra -fsycl-max-parallel-link-jobs=4)
printf '%q ' icpx "${FLAGS[@]}" -Xsycl-target-backend=spir64_gen \
  '-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt' \
  /src/int8_xmx_capability.cpp -o /out/int8_xmx_capability_s74 > compile-command.txt
printf '\n' >> compile-command.txt
icpx "${FLAGS[@]}" -Xsycl-target-backend=spir64_gen \
  '-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt' \
  /src/int8_xmx_capability.cpp -o /out/int8_xmx_capability_s74
sha256sum -c /src/expected-container.sha256
sha256sum /src/int8_xmx_capability.cpp /out/int8_xmx_capability_s74 > build-sha256.txt
