#!/usr/bin/env bash
# usage: run_bench_c.sh URL MODEL TAG LEVELS(comma) PREFILL_SIZES(comma or empty) [EQ(1/0)]
URL=$1; MODEL=$2; TAG=$3; LEVELS=$4; SIZES=$5; EQ=${6:-1}
W=/mnt/vm_8tb/b70/build/tensorfold; T=$W/main/tools; OUT=$W/out/$TAG; mkdir -p $OUT
cd $T
echo "## bench_openai decode $(date +%T)"
python3 bench_openai.py $URL $MODEL --tokens 256 --reps 3 --output $OUT/openai.json 2>&1 | tail -12
for n in ${LEVELS//,/ }; do
  echo "## spread_client n=$n $(date +%T)"
  python3 /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/concurrency-0216/tensorfold-probe/spread_client.py $OUT/spread_$n.json --n $n --tokens 256 --model $MODEL $URL 2>&1 | head -22
done
if [ "$EQ" = 1 ]; then
  for n in ${LEVELS//,/ }; do
    echo "## bench_concurrent --alone --serial n=$n $(date +%T)"
    python3 bench_concurrent.py $URL $MODEL --levels $n --tokens 256 --reps 2 --alone --serial --output $OUT/conc_$n.json 2>&1 | tail -25
  done
fi
for s in ${SIZES//,/ }; do
  echo "## bench_cache prefill $s $(date +%T)"
  python3 bench_cache.py $URL $MODEL --prompt-file $W/prompts/prompt_$s.json --expected-prompt-tokens $s --tokens 16 --repeats 2 --streams 2 --require-cold --output $OUT/cache_$s.json 2>&1 | tail -14
done
echo "## done $(date +%T)"
