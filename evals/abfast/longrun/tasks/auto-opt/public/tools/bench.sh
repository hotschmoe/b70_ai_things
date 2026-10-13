#!/bin/bash
# usage: tools/bench.sh [N] [SEED]   -- build, generate an input, check against the oracle if small, time 3 runs.
set -e
cd "$(dirname "$0")/.."
N=${1:-2500}; SEED=${2:-1}
cargo build --release --offline 2>&1 | tail -2
mkdir -p /tmp/nnjoin_bench
python3 tools/gen.py "$N" "$SEED" 30000 10 100 > /tmp/nnjoin_bench/in.txt
for i in 1 2 3; do
  /usr/bin/time -f "run $i: %e s wall, %U s user" target/release/nnjoin /tmp/nnjoin_bench/in.txt > /tmp/nnjoin_bench/out.txt
done
md5sum /tmp/nnjoin_bench/out.txt
if [ "$N" -le 400 ]; then python3 tools/expected.py /tmp/nnjoin_bench/in.txt | cmp - /tmp/nnjoin_bench/out.txt && echo "matches oracle"; fi
