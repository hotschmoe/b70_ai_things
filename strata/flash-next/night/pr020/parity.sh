#!/bin/bash
# All six kernel-PR parity self-tests on 0.2.0-beta builds, card 1, runtime image (CR 26.22).
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
B=/mnt/vm_8tb/b70/results/pr-20261016/builds
P=/mnt/vm_8tb/b70/results/pr-020/parity
run() { # run <log> "<env -e list>" <cmd>
  local log=$1 envs=$2; shift 2
  echo "== $(date +%T) $log"
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=pr020-parity $REPO/bin/gpu-run --card 1 bash -c "timeout 960 docker run --rm --name par_$log --network none --device /dev/dri -e ONEAPI_DEVICE_SELECTOR=level_zero:1 -e SYCL_CACHE_PERSISTENT=0 $envs -v $B:$B:ro -v $P:$P --entrypoint bash $IMAGE -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; timeout 900 $*; echo exit=\$?'; docker kill par_$log >/dev/null 2>&1" > $P/$log.log 2>&1
  tail -3 $P/$log.log
}
run kreg "" $B/strata-kreg020-parity --selftest
run iq "" $B/strata-iq020-parity --iq
run iq-plain "" $B/strata-iq020-parity
run da-off "" $B/strata-da020-parity --selftest --write-ref $P/qvb_ref.txt
run da-on "-e STRATA_QSA_TOPK_SCAN=1 -e STRATA_QSA_SCORES_TRANSPOSED=1 -e STRATA_ATTN_MERGE_ACTIVE=1" $B/strata-da020-parity --selftest --ref $P/qvb_ref.txt
run pattn "" $B/strata-pattn020-parity --selftest
run hcq8 "" $B/strata-hcq8020-parity --selftest
run hcf "" $B/strata-hcf020-parity --selftest
echo PARITY_DONE
