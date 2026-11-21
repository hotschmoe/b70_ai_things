#!/usr/bin/env bash
# Run Strata's real server.py (mock engine) and TensorFold's fake_serve (scripted engine, production Info profile) side by
# side on loopback test ports, run parity.py against both, stop both. CPU only: no GPU, no weights, no production port.
#   run_parity.sh [TF_WORKTREE] [extra parity.py args]
# Env: TF_ENV="TF_FAKE_SAMPLED=1 ..." extra environment for fake_serve (SAMPLED models an engine that samples);
#      TF_EXTRA_ARGS extra fake_serve flags; PARITY_CTX (default 4096); PARITY_OUT (logs and report directory).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
wt=${1:-/mnt/vm_8tb/b70/build/tensorfold/dd-api}; shift || true
out=${PARITY_OUT:-/mnt/vm_8tb/b70/results/tf-flashnext/workers/API}
ctx=${PARITY_CTX:-4096}
sport=${PARITY_STRATA_PORT:-18961}; tport=${PARITY_TF_PORT:-18962}
for p in $sport $tport; do
  if ss -ltn "sport = :$p" | tail -n +2 | grep -q .; then echo "port $p busy" >&2; exit 2; fi
done
mkdir -p "$out"
python3 "$here/scenarios.py" "$out/scripts.json" >/dev/null
py=/mnt/vm_8tb/b70/build/strata-0216/venv/bin/python
"$py" "$here/strata_mock_launch.py" "$sport" "$ctx" >"$out/strata_mock.log" 2>&1 & spid=$!
env TF_FAKE_PROFILE=dd TF_FAKE_TOKENIZER=/mnt/cache/b70/devmodel/Qwen/Qwen3.8-Flash-Next TF_FAKE_SCRIPTS="$out/scripts.json" \
  TF_FAKE_CONTEXT="$ctx" ${TF_ENV:-} \
  nice -n 19 "$wt/zig-out/server/fake_serve" serve "$wt/zig/tests/server/fixtures" --name hotschmoe-dd \
  --alias qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216 --alias qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-tensorfold-372bf1b2 \
  --parallel 4 --max-tokens 32768 --temperature 0 --thinking --reasoning-effort medium --thinking-budget 16384 \
  --host 127.0.0.1 --port "$tport" ${TF_EXTRA_ARGS:-} >"$out/tf_fake.log" 2>&1 & tpid=$!
cleanup() { kill "$spid" "$tpid" 2>/dev/null; wait "$spid" "$tpid" 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 60); do
  curl -s -m 2 "http://127.0.0.1:$sport/v1/models" >/dev/null && curl -s -m 2 "http://127.0.0.1:$tport/v1/models" >/dev/null && break
  sleep 1
done
python3 "$here/parity.py" --strata "http://127.0.0.1:$sport" --tf "http://127.0.0.1:$tport" --json "$out/parity.json" "$@"
rc=$?
echo "logs: $out/strata_mock.log $out/tf_fake.log ; report: $out/parity.json"
exit $rc
