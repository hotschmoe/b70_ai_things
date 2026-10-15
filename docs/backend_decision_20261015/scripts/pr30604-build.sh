#!/usr/bin/env bash
# Incremental SYCL rebuild of e467e5b5 + PR 30604 (copy of build-sycl). Run under bin/cpu-run.
cd /mnt/vm_8tb/b70/build/backend-decision
docker run --rm --name bd-pr30604-build --memory 90g --network none --user 1000:1000 -v $PWD/llama.cpp-pr30604:/src -w /src --entrypoint bash strata-ab-build:ocloc -c \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cmake --build build-sycl -j4 --target llama-server llama-bench 2>&1 | grep -E 'error|FAILED|Linking|Building' | tail -20"
