#!/bin/bash
N=/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/night; R=/mnt/vm_8tb/b70/results/pr-020
cd $N; export PR_OUT=$R/runs
echo "#### hcq8 32k"; PR_CONFIG=pr-020-layer-c1-32k.json ./pr_validate.sh $R/hcq8.spec --c 1 --tokens 512 --reps 2 --prefill 4096 2>&1 | grep -E "^==|FAILED|^hcq8-32k"
echo "#### da c2"; PR_CONFIG=pr-020-layer-c2-16k.json ./pr_validate.sh $R/dac2.spec --c 2 --tokens 512 --reps 2 --prefill '' 2>&1 | grep -E "^==|FAILED|^da-c2"
echo "#### ep"; PR_CONFIG=pr-020-ep-c1.json ./pr_validate.sh $R/ep.spec --c 1 --tokens 512 --reps 2 --prefill 4096,32768 2>&1 | grep -E "^==|FAILED|^ep-"
echo "#### oracle"; $R/oracle.sh
