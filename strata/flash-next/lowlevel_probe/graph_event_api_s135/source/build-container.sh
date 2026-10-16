#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && -w /out/tmp ]]
cd /out
sha256sum -c /src/expected-container.sha256
icpx --version > compiler-version.txt
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32
  -fsycl-device-code-split=per_kernel -fp-model=precise -fsycl-max-parallel-link-jobs=4
  -Xsycl-target-backend=spir64_gen '-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt'
  -I/native/sycl/include -I/native/include -Wall -Wextra -Wno-deprecated-declarations)
printf '%q ' icpx "${FLAGS[@]}" /src/probe.cpp -lze_loader -o cross_queue_graph > command.txt
printf '\n' >> command.txt
TIMEFORMAT='compile_link_seconds=%R'
{ time icpx "${FLAGS[@]}" /src/probe.cpp -lze_loader -o cross_queue_graph; } 2> compiler.stderr-time
sha256sum -c /src/expected-container.sha256
sha256sum /src/probe.cpp cross_queue_graph > build-sha256.txt
sha256sum cross_queue_graph /src/probe.cpp > runtime.sha256
