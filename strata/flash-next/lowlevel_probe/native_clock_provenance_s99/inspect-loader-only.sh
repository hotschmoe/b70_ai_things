#!/usr/bin/env bash
# Direct glibc loader --list mode, not ldd's shell with LD_PRELOAD injected.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
test ! -e /dev/dri
unset LD_PRELOAD LD_AUDIT
NATIVE=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/attn-value-resident-s94-build/build/strata-attn-value-resident-s94
PROF=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/libunitrace_tool.so
/lib64/ld-linux-x86-64.so.2 --list --preload "$PROF" "$NATIVE" > /out/reconstructed-loader-direct.txt
LIB=$(awk '/libstdc\+\+.so.6 =>/ {print $3}' /out/reconstructed-loader-direct.txt)
test -n "$LIB" && test -f "$LIB"
realpath "$LIB" > /out/libstdcxx-realpath.txt
sha256sum "$LIB" > /out/libstdcxx.sha256
objdump -T "$LIB" | grep '_ZNSt6chrono3_V212steady_clock3nowEv' > /out/steady-clock-symbol.txt
objdump -d --disassemble=_ZNSt6chrono3_V212steady_clock3nowEv "$LIB" > /out/steady-clock.asm
readelf -d "$LIB" > /out/libstdcxx-dynamic.txt
grep -n 'CLOCK_MONOTONIC' /usr/include/x86_64-linux-gnu/bits/time.h > /out/clock-id-header.txt
printf '%s\n' 'PASS_OFFLINE_LOADER_RECONSTRUCTION_ONLY_NOT_S95_MAPS_PROOF'
