source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri ]]
cd /out
icpx --version > compiler.txt
# Two exact source spellings, no alternate flags/retry/cache/main build.
for variant in 0 1; do
 icpx -O3 -DNDEBUG -std=c++20 -fp-model=precise -fsycl -DUNIQUE_C6=$variant /src/repro.cpp -o "repro$variant" > "compile$variant.stdout" 2> "compile$variant.stderr"
 env STRATA_HEAD_Q5K_RPW2=0 STRATA_Q5_HEAD_RUNTIME_ONES=01010101 STRATA_Q5_HEAD_RUNTIME_ONES_C6=01010101 "./repro$variant" > "run$variant.stdout" 2> "run$variant.stderr"
 printf '%s\n' "$?" > "run$variant.exit"
done
