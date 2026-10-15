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
printf '%q ' icpx "${FLAGS[@]}" /src/probe.cpp -lze_loader -o paired_graph_s103 > command.txt
printf '\n' >> command.txt
TIMEFORMAT='compile_link_seconds=%R'
{ time icpx "${FLAGS[@]}" /src/probe.cpp -lze_loader -o paired_graph_s103; } 2> compiler.stderr-time
sha256sum -c /src/expected-container.sha256
sha256sum /src/probe.cpp paired_graph_s103 > build-sha256.txt
UNITRACE=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace
printf '%s  %s\n' 00fc84e5abbf19eddcd9010095b74857f4ec7df97c401ee0af7d70fb07ff1d68 "$UNITRACE" | sha256sum -c -
sha256sum paired_graph_s103 "$UNITRACE" "$(dirname -- "$UNITRACE")/libunitrace_tool.so" /src/probe.cpp > runtime.sha256
