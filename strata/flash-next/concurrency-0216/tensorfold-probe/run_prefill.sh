#!/usr/bin/env bash
# usage: run_prefill.sh URL MODEL TAG SIZES  (cold prefill via bench_cache + short decode)
URL=$1; MODEL=$2; TAG=$3; SIZES=$4; W=/mnt/vm_8tb/b70/build/tensorfold; OUT=$W/out/$TAG; mkdir -p $OUT; cd $W/main/tools
python3 bench_openai.py $URL $MODEL --tokens 256 --reps 2 --output $OUT/openai.json 2>&1 | tail -4 | cut -c1-260
for s in ${SIZES//,/ }; do echo "## prefill $s"; python3 bench_cache.py $URL $MODEL --prompt-file $W/prompts/prompt_$s.json --expected-prompt-tokens $s --tokens 16 --repeats 1 --streams 1 --require-cold --output $OUT/cache_$s.json 2>&1 | tail -3; done
echo "## done"
