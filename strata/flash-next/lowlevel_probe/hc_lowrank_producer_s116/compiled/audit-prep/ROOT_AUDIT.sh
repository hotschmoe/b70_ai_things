#!/usr/bin/env bash
# SOURCE ONLY. Root full-read/CPU_AUDIT_GO then cpu-run outside this script.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$HERE/audit.py" \
 --prep /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/hc-producer-s116-oracle-prep \
 --expected-prep-manifest-sha256 3a92134aeae6ceb56a12912a6aa4c6c113cfcca7bc4c31d51ecea0c500306d55 \
 --expected-exe-sha256 33e390536480f83f169a63e1f9d2f2e06d8e341279a956c84a29a255951c46d3 \
 --expected-object-sha256 82ab8f17718b5fc336620b4322469175157d457c290c4bb33878ef8d06f5bfaa \
 --expected-images-sha256 bb526113208d267f403e12f36996fc0593e8c261dbe2ba3d009ed418c623386c
