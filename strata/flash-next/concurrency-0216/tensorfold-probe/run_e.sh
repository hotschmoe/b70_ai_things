#!/usr/bin/env bash
W=/mnt/vm_8tb/b70/build/tensorfold; OUT=$W/out/e_two_replicas; mkdir -p $OUT
M0=tf-qwen38-27b-gptq-int4-g128-dflash2-r0; M1=tf-qwen38-27b-gptq-int4-g128-dflash2-r1
for spec in "2 a" "8 a" "14 a" "16 a" "16 b" "16 c"; do set -- $spec
  echo "## spread n=$1 rep $2 $(date +%T)"
  python3 /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/concurrency-0216/tensorfold-probe/spread_client.py $OUT/spread_$1_$2.json --n $1 --tokens 256 --model $M0 --model $M1 http://127.0.0.1:18144 http://127.0.0.1:18145 2>&1 | head -18
done
echo "## done $(date +%T)"
