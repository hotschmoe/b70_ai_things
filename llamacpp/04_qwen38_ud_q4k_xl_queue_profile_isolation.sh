#!/usr/bin/env bash
# Two-arm isolation of SYCL queue enable_profiling with timing barriers unreachable.
# CPU-only validation is the default. GPU execution requires:
#   ./bin/gpu-run bash llamacpp/04_qwen38_ud_q4k_xl_queue_profile_isolation.sh full 2
# or, for TP=1:
#   ./bin/gpu-run --card 0 bash llamacpp/04_qwen38_ud_q4k_xl_queue_profile_isolation.sh full 1
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-static}"
GPU_COUNT="${2:-${GPU_COUNT:-2}}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/qwen38_ud_q4k_xl_queue_profile_isolation_${STAMP}_tp${GPU_COUNT}}"
PORT="${PORT:-18080}"
SERVED="${SERVED:-hotschmoe-dd}"
IMG="${IMG:-qwen38-b70:quant-timing}"
IMG_ID_EXPECTED="${IMG_ID_EXPECTED:-}"
SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-ud-q4-k-xl/serve.sh"
OVERLAY="$REPO/llamacpp/qwen38_ud_q4k_xl_profile_entrypoint.sh"
ANALYZER="$REPO/llamacpp/analyze_qwen38_queue_profile_isolation.py"
LOCK_BASE="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"
TIMING_SKIP=18446744073709551615
NAMES=(
    "qwen38_xl_qprofile_tp${GPU_COUNT}_off"
    "qwen38_xl_qprofile_tp${GPU_COUNT}_on"
)
active_name=""
active_dir=""

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

static_checks() {
    bash -n "$0" "$SHELF" "$REPO/llamacpp/serve_qwen38_stock_q4km_tp2.sh"
    git -C "$REPO" apply --numstat \
        "$REPO/llamacpp/qwen38-b70/patches/quant-timing.patch" >/dev/null
    python3 -m unittest \
        llamacpp.tests.test_queue_profile_isolation \
        llamacpp.tests.test_queue_profile_isolation_static
    git -C "$REPO" diff --check -- \
        llamacpp/qwen38-b70/patches/quant-timing.patch \
        llamacpp/qwen38-b70/build_image.sh \
        llamacpp/serve_qwen38_stock_q4km_tp2.sh \
        llamacpp/04_qwen38_ud_q4k_xl_queue_profile_isolation.sh \
        llamacpp/analyze_qwen38_queue_profile_isolation.py \
        llamacpp/tests/test_queue_profile_isolation.py \
        llamacpp/tests/test_queue_profile_isolation_static.py
}

ancestor_has_pid() {
    local wanted="$1" pid="$$"
    while [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" -gt 1 ]; do
        [ "$pid" = "$wanted" ] && return 0
        [ -r "/proc/$pid/status" ] || return 1
        pid="$(awk '/^PPid:/ { print $2 }' "/proc/$pid/status")"
    done
    return 1
}

require_external_gpu_run() {
    local card owner owner_pid
    for ((card = 0; card < GPU_COUNT; card++)); do
        owner="$LOCK_BASE.$card.owner"
        [ -s "$owner" ] || { say "card $card has no gpu-run owner"; return 2; }
        [[ "$(<"$owner")" =~ pid=([0-9]+) ]] || return 2
        owner_pid="${BASH_REMATCH[1]}"
        kill -0 "$owner_pid" 2>/dev/null || return 2
        ancestor_has_pid "$owner_pid" || return 2
    done
}

write_endpoint_down() {
    local listener=0 running=() name
    ss -ltnH "sport = :$PORT" | rg -q . && listener=1
    for name in "${NAMES[@]}"; do
        docker ps --format '{{.Names}}' | rg -qx "$name" && running+=("$name")
    done
    python3 - "$OUT/endpoint_down.json" "$listener" "${running[*]}" <<'PY'
import json
import sys
from pathlib import Path

out, listener, names = sys.argv[1:]
running = names.split() if names else []
payload = {
    "passed": listener == "0" and not running,
    "port_listener": bool(int(listener)),
    "running_campaign_containers": running,
}
Path(out).write_text(
    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    encoding="ascii",
)
raise SystemExit(0 if payload["passed"] else 1)
PY
}

capture_active() {
    [ -n "$active_name" ] || return 0
    docker inspect "$active_name" >"$active_dir/container_inspect.json" 2>/dev/null || true
    docker inspect "$active_name" --format '{{range .Config.Env}}{{println .}}{{end}}' \
        | sort >"$active_dir/container_env.txt" 2>/dev/null || true
    docker logs "$active_name" >"$active_dir/server.log" 2>&1 || true
    docker stop --time 30 "$active_name" >"$active_dir/docker_stop.log" 2>&1 || true
    docker rm -f "$active_name" >"$active_dir/docker_rm.log" 2>&1 || true
    active_name=""
    active_dir=""
}

health_probe() {
    local label="$1" rc
    local health_args=()
    [ "$GPU_COUNT" = "1" ] && health_args=(--card 0)
    set +e
    env -u IMG "$REPO/bin/xpu-health" "${health_args[@]}" \
        > >(tee "$OUT/health_${label}.log") 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" >"$OUT/health_${label}.rc"
    return "$rc"
}

check_code_hashes() {
    local rc
    set +e
    sha256sum -c "$OUT/code_sha256.txt" >"$OUT/code_sha256_check.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" >"$OUT/code_sha256_check.rc"
    return "$rc"
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    capture_active || rc=1
    if [ -d "$OUT" ]; then
        write_endpoint_down || rc=1
    fi
    say "exit=$rc endpoint=down artifacts=$OUT"
    exit "$rc"
}

refuse_occupied() {
    local name
    ss -ltnH "sport = :$PORT" | rg -q . && {
        say "port $PORT occupied; refusing to stop an unrelated endpoint"
        return 2
    }
    for name in "${NAMES[@]}"; do
        docker ps -a --format '{{.Names}}' | rg -qx "$name" && {
            say "stale campaign container $name exists"
            return 2
        }
    done
}

write_manifest() {
    local image_id="$1" patch_sha="$2"
    python3 - "$OUT/manifest.json" "$image_id" "$GPU_COUNT" "$TIMING_SKIP" "$patch_sha" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out, image_id, gpu_count, timing_skip, patch_sha = sys.argv[1:]
payload = {
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "image_id": image_id,
    "quant_timing_patch_sha": patch_sha,
    "gpu_count": int(gpu_count),
    "timing_period": 64,
    "timing_skip": int(timing_skip),
    "restart_policy": "no",
    "arms": {"queue_profile_off": 0, "queue_profile_on": 1},
}
Path(out).write_text(
    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    encoding="ascii",
)
PY
    sha256sum \
        "$REPO/llamacpp/qwen38-b70/patches/quant-timing.patch" \
        "$REPO/llamacpp/qwen38-b70/build_image.sh" \
        "$REPO/llamacpp/serve_qwen38_stock_q4km_tp2.sh" \
        "$REPO/llamacpp/04_qwen38_ud_q4k_xl_queue_profile_isolation.sh" \
        "$ANALYZER" >"$OUT/code_sha256.txt"
}

run_arm() {
    local label="$1" queue_profile="$2"
    local name="qwen38_xl_qprofile_tp${GPU_COUNT}_${label}" directory="$OUT/queue_profile_$label"
    mkdir -p "$directory"
    active_name="$name"
    active_dir="$directory"
    say "arm=$label tp=$GPU_COUNT queue_profile=$queue_profile barriers=unreachable"
    set +e
    NAME="$name" PORT="$PORT" SERVED="$SERVED" IMG="$IMG" IMG_ID="$IMAGE_ID" \
        OVERLAY="$OVERLAY" GPU_COUNT="$GPU_COUNT" ENABLE_MTP=0 LAB_DOORS=0 \
        RESTART_POLICY=no GGML_SYCL_QUANT_CENSUS=1 \
        GGML_SYCL_QUANT_TIMING_SAMPLE=64 GGML_SYCL_QUANT_TIMING_SKIP="$TIMING_SKIP" \
        GGML_SYCL_QUANT_TIMING_MAX=65536 \
        GGML_SYCL_QUANT_TIMING_QUEUE_PROFILE="$queue_profile" \
        bash "$SHELF" start 2>&1 | tee "$directory/start.log"
    local start_rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$start_rc" >"$directory/start.rc"
    capture_active
    return "$start_rc"
}

run_full() {
    case "$GPU_COUNT" in 1|2) ;; *) say "GPU_COUNT must be 1 or 2"; return 2 ;; esac
    static_checks
    require_external_gpu_run
    refuse_occupied
    [ ! -e "$OUT" ] || { say "refusing existing output $OUT"; return 2; }
    mkdir -p "$OUT"
    exec > >(tee -a "$OUT/campaign.log") 2>&1
    trap on_exit EXIT INT TERM
    IMAGE_ID="$(docker image inspect "$IMG" --format '{{.Id}}')"
    [ -z "$IMG_ID_EXPECTED" ] || [ "$IMAGE_ID" = "$IMG_ID_EXPECTED" ] || {
        say "image id mismatch expected=$IMG_ID_EXPECTED actual=$IMAGE_ID"
        return 2
    }
    PATCH_SHA="$(sha256sum "$REPO/llamacpp/qwen38-b70/patches/quant-timing.patch" | awk '{ print $1 }')"
    IMAGE_PATCH_SHA="$(docker image inspect "$IMG" \
        --format '{{index .Config.Labels "ai.b70.quant_timing_patch_sha"}}')"
    [ "$IMAGE_PATCH_SHA" = "$PATCH_SHA" ] || {
        say "stale image: timing patch label=$IMAGE_PATCH_SHA source=$PATCH_SHA"
        return 2
    }
    write_manifest "$IMAGE_ID" "$PATCH_SHA"
    health_probe pre
    run_arm off 0 || {
        say "profile-off control failed; refusing the profile-on arm"
        return 1
    }
    if rg -i 'device_lost|out_of_resources|ur_result_error' \
        "$OUT/queue_profile_off/server.log"; then
        say "profile-off control contains a fatal marker; refusing the profile-on arm"
        return 1
    fi
    health_probe after_off || {
        say "profile-off teardown left GPU health degraded; refusing the profile-on arm"
        return 1
    }
    if run_arm on 1; then
        say "profile-on arm reached healthy startup"
    else
        say "profile-on arm exited nonzero; analyzer will require the expected failure signature"
    fi
    health_probe final || say "final GPU health failed; analyzer will reject the run"
    check_code_hashes || say "source hashes changed; analyzer will reject the run"
    write_endpoint_down
    python3 "$ANALYZER" "$OUT" --write "$OUT/summary.json"
    trap - EXIT INT TERM
    say "complete endpoint=down artifacts=$OUT"
}

case "$ACTION" in
    static) static_checks ;;
    full) run_full ;;
    *) echo "usage: $0 {static|full} [1|2]" >&2; exit 2 ;;
esac
