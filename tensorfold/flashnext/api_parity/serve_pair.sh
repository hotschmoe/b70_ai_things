#!/usr/bin/env bash
# Start the Strata mock and the TensorFold fake server side by side and wait (foreground; stop with SIGTERM). Ports 18961/18962.
here=$(cd "$(dirname "$0")" && pwd)
wt=${1:-/mnt/vm_8tb/b70/build/tensorfold/dd-api}
out=${PARITY_OUT:-/mnt/vm_8tb/b70/results/tf-flashnext/workers/API}
ctx=${PARITY_CTX:-4096}
python3 "$here/scenarios.py" "$out/scripts.json" >/dev/null
/mnt/vm_8tb/b70/build/strata-0216/venv/bin/python "$here/strata_mock_launch.py" 18961 "$ctx" >"$out/strata_mock.log" 2>&1 & spid=$!
env TF_FAKE_PROFILE=dd TF_FAKE_TOKENIZER=/mnt/cache/b70/devmodel/Qwen/Qwen3.8-Flash-Next TF_FAKE_SCRIPTS="$out/scripts.json" TF_FAKE_CONTEXT="$ctx" ${TF_ENV:-} \
  nice -n 19 "$wt/zig-out/server/fake_serve" serve "$wt/zig/tests/server/fixtures" --name hotschmoe-dd \
  --alias qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216 --alias qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-tensorfold-372bf1b2 \
  --parallel 4 --max-tokens 32768 --temperature 0 --thinking --reasoning-effort medium --thinking-budget 16384 \
  --host 127.0.0.1 --port 18962 >"$out/tf_fake.log" 2>&1 & tpid=$!
trap 'kill $spid $tpid 2>/dev/null' EXIT TERM INT
wait
