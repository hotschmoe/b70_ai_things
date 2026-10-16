#!/usr/bin/env bash
set -euo pipefail
[[ -z ${LD_PRELOAD:-} ]] || { echo 'STOP unexpected preload'; exit 2; }
unset UNITRACE_StartPaused UNITRACE_SystemTime SYCL_UR_TRACE UR_LOG_LOADER UR_LOG_LEVEL_ZERO
exec /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/q5-head-c6-main-s133-build/build/strata-s133 "$@"
