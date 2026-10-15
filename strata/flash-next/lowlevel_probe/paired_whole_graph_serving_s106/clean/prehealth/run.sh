#!/usr/bin/env bash
set -uo pipefail
cd /mnt/vm_8tb/github/b70_ai_things
D=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/paired-whole-graph-s106-prep/root-prehealth
IMG=b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78
date -u +%FT%TZ > "$D/start.txt"
uname -r > "$D/kernel.txt"
cat /proc/sys/kernel/random/boot_id > "$D/boot-id.txt"
bin/xpu-health --img "$IMG" --timeout 400 > "$D/card-health.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$D/card-health.exit"
[ "$rc" = 0 ] || exit "$rc"
bin/xpu-collective-health --img "$IMG" --timeout 400 > "$D/collective-health.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$D/collective-health.exit"
date -u +%FT%TZ > "$D/end.txt"
exit "$rc"
