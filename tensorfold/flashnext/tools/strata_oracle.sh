#!/usr/bin/env bash
# strata_oracle.sh -- Strata 0.2.16 reference tokens + logits for the TensorFold Flash-Next GGUF port.  NOT RUN yet.
#
# What it does (see tensorfold/flashnext/GGUF-MAP.md, section "Oracle"):
#   1. tokenizes 8 fixed prompts (CPU only) -> $OUT/prompts.json
#   2. starts the qualified engine binary /mnt/vm_8tb/b70/build/strata-0216/strata-dd-0216 in --serve mode inside the
#      strata-sycl-dev:020 container, UNDER bin/gpu-run (BOTH cards leased for the container's whole life; the lease
#      is taken by the python client around "docker run", see strata_oracle.py docker_cmd), configuration =
#      hotschmoe-dd-0216.json with spec/MTP OFF ("plain" mode, cleaner logits), fp16 KV, BF16 HC reads, no vision
#   3. drives the engine's stdin protocol directly (raw token ids in, greedy tokens + top-20 logprobs out), so NO HTTP
#      port is opened (the 18134 port in the brief is only used by the optional "http" mode below); each prompt also
#      dumps full-vocab logits via STRATA_LOGITS_DUMP, from which first-token logits and top-64+logsumexp per position
#      are extracted;  4. sends QUIT, waits, docker stop as fallback, removes the raw dump, checks teardown.
#
# Cards: 0 and 1 (whole box).  Expected duration: ~15-25 min wall (engine load is the unknown; UNVERIFIED), of which
# decoding 8 x 256 tokens + the 4.8k-token prompt is a few minutes.  Announce it to other agents first (CLAUDE.md).
# Outputs: $OUT/oracle.json prompts.json first_logits/*.f32 engine.log engine_plan.json  (OUT below)
# Timings from this run are NOT speed measurements (no cpu-run --quiet window, debug dump on).
#
# Usage:
#   tensorfold/flashnext/tools/strata_oracle.sh run        [extra args for "strata_oracle.py run", e.g. --max-new 128 --only p0_en_story_raw]
#   tensorfold/flashnext/tools/strata_oracle.sh plan       # print the exact docker/engine command, start nothing
#   tensorfold/flashnext/tools/strata_oracle.sh compare --tf tf_result.json [--min-top1 0.98]
#   tensorfold/flashnext/tools/strata_oracle.sh http       # write an HTTP server config and print the 18134 launch command (does not start)
# Env: ORACLE_OUT (default below), ORACLE_HEALTH=1 (xpu-health before/after), ORACLE_MODE=plain|qualified.
set -euo pipefail
repo=/mnt/vm_8tb/github/b70_ai_things
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
py=/mnt/vm_8tb/b70/build/strata-0216/venv/bin/python      # has 'regex' (tokenizer) and numpy
out=${ORACLE_OUT:-/mnt/vm_8tb/b70/results/tf-flashnext/oracle}
mode=${ORACLE_MODE:-plain}
name=tf-strata-oracle
cmd=${1:-run}; shift || true
client=("$py" -B "$here/strata_oracle.py")

case "$cmd" in
  plan)    exec "${client[@]}" run --out "$out" --mode "$mode" --print-only "$@" ;;
  compare) exec "${client[@]}" compare --oracle "$out/oracle.json" "$@" ;;
  http)
    cfgfile=$("${client[@]}" http-config --out "$out" --mode "$mode")
    echo "config: $cfgfile"
    echo "To start (holds both cards; DD must be stopped; foreground):"
    echo "  $repo/strata/flash-next/dd/nl_launch_0216.sh tf-strata-oracle-http $cfgfile 18134"
    echo "Stop: docker stop tf-strata-oracle-http.  (Unverified: the server may add flags such as --batch/--vision itself.)"
    exit 0 ;;
  run) ;;
  *) echo "usage: $0 run|plan|compare|http [args]" >&2; exit 2 ;;
esac

# ---- preconditions (read-only) ------------------------------------------------------------------------------
if systemctl --user is-active --quiet hotschmoe-dd 2>/dev/null; then
  echo "refusing: hotschmoe-dd.service is active and holds both GPU leases; stop it first (systemctl --user stop hotschmoe-dd)" >&2; exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx "$name"; then echo "refusing: container $name already running" >&2; exit 1; fi
"$repo/bin/gpu-run" --status
mkdir -p "$out"
echo "== strata_oracle: cards 0+1, mode=$mode, out=$out, ~15-25 min (UNVERIFIED), no HTTP port =="

cleanup() {   # teardown evidence is printed below; this only guarantees no container survives a signal
  docker stop -t 30 "$name" >/dev/null 2>&1 || true
}
trap cleanup INT TERM

if [ "${ORACLE_HEALTH:-1}" = 1 ]; then "$repo/bin/gpu-run" "$repo/bin/xpu-health" || { echo "pre-health failed; not starting" >&2; exit 1; }; fi

rc=0
"${client[@]}" run --out "$out" --mode "$mode" "$@" || rc=$?

# ---- teardown evidence ----------------------------------------------------------------------------------------
docker stop -t 30 "$name" >/dev/null 2>&1 || true
echo "-- teardown: containers named $name: $(docker ps -a --format '{{.Names}}' | grep -cx "$name" || true) (want 0)"
"$repo/bin/gpu-run" --status
if [ "${ORACLE_HEALTH:-1}" = 1 ]; then "$repo/bin/gpu-run" "$repo/bin/xpu-health" || echo "POST-HEALTH FAILED: run bin/xe-reset per CLAUDE.md before more GPU work" >&2; fi
ls -la "$out" "$out/first_logits" 2>/dev/null | head -40
exit $rc
