#!/usr/bin/env bash
# hotschmoe-dd on TensorFold Flash-Next (systemd: hotschmoe-dd-tensorfold.service).
# Runs in the foreground under the caller's bin/gpu-run lease (both cards).
#   TF_DD_CONFIG  variant file (default tf-dd-b2.env next to this script)
#   TF_LISTEN     tcp  -> serve 127.0.0.1:${TF_PORT:-18124} directly (default)
#                 unix -> serve on an owner-only Unix socket; tf_clamp_proxy.py owns 127.0.0.1:18124
#   TF_PORT       tcp port (default 18124 = production backend; gpu_window.sh uses 18150)
#   TF_EXTRA_ARGS extra serve flags from the env file (word-split; final production flags land here)
#   TF_DD_DRYRUN=1 prints the command and exits.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
config=${TF_DD_CONFIG:-$here/tf-dd-b2.env}
# Clear inherited engine knobs first; the env file is the only source of engine flags.
unset FN_GGUF_REF XPU_PLE_DEVICE XPU_PLE_CHECK XPU_TP_HOST_AR XPU_MTP_DEPTH FN_MTP_LANE_POLICY XPU_FN_RECORD XPU_FN_GATE XPU_FN_DESC XPU_FN_KV
source "$config"
listen=${TF_LISTEN:-tcp}
port=${TF_PORT:-18124}
sock_dir=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/hotschmoe-dd-tf
sock=$sock_dir/tf.sock
cmd=("$TF_BIN" serve "$TF_MODEL_DIR" --tp 2)
if [[ $listen == unix ]]; then
  mkdir -p -m 700 "$sock_dir"; chmod 700 "$sock_dir"; rm -f "$sock"
  cmd+=(--unix-socket "$sock")
else
  cmd+=(--host 127.0.0.1 --port "$port")
fi
cmd+=(--name "$TF_NAME")
for a in $TF_ALIASES; do cmd+=(--alias "$a"); done
cmd+=(--parallel "$TF_PARALLEL" --context "$TF_CONTEXT" --max-tokens "$TF_MAX_TOKENS"
      --prompt-cache-gib "$TF_PROMPT_CACHE_GIB"
      --thinking --reasoning-effort "$TF_REASONING_EFFORT" --thinking-budget "$TF_THINKING_BUDGET"
      --dashboard --no-update-check)
[[ -n ${TF_TEMPERATURE:-} ]] && cmd+=(--temperature "$TF_TEMPERATURE")
for a in ${TF_EXTRA_ARGS:-}; do cmd+=("$a"); done
printf '%s [dd_launch_tf] variant=%s config=%s listen=%s port=%s commit=%s\n' "$(date -u '+%F %T UTC')" "$TF_VARIANT" "$config" "$listen" "$port" "$TF_COMMIT"
printf '%s [dd_launch_tf] env: XPU_TP=%s FN_GGUF_ACT=%s XPU_CHUNK=%s XPU_CHUNK_SOLO=%s XPU_MTP_CONF=%s XPU_FN_KV=%s XPU_FN_DESC=%s extra=[%s]\n' "$(date -u '+%F %T UTC')" "$XPU_TP" "$FN_GGUF_ACT" "$XPU_CHUNK" "$XPU_CHUNK_SOLO" "$XPU_MTP_CONF" "${XPU_FN_KV:-unset}" "${XPU_FN_DESC:-unset}" "${TF_EXTRA_ARGS:-}"
printf '%s [dd_launch_tf] COMMAND:' "$(date -u '+%F %T UTC')"; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ ${TF_DD_DRYRUN:-0} == 1 ]]; then exit 0; fi
have=$(sha256sum "$TF_BIN" | cut -d' ' -f1)
if [[ $have != "$TF_BIN_SHA256" ]]; then
  echo "dd_launch_tf: binary sha256 $have != frozen $TF_BIN_SHA256; refusing to serve" >&2; exit 1
fi
if [[ $listen == tcp ]] && ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
  echo "dd_launch_tf: 127.0.0.1:$port already in use (Strata hotschmoe-dd still up?)" >&2; exit 1
fi
cd "$(dirname "$TF_BIN")/../../.."
exec "${cmd[@]}"
