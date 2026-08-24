#!/usr/bin/env bash
# Fail-closed UD-Q4_K_XL quant timing qualification campaign.
# Run full only under: ./bin/gpu-run bash llamacpp/01_qwen38_ud_q4k_xl_quant_timing_campaign.sh full
# This script never restores production; the endpoint is left down.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-full}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/qwen38_ud_q4k_xl_quant_timing_$STAMP}"
PORT="${PORT:-18080}"
SERVED="${SERVED:-hotschmoe-dd}"
CURRENT_IMG="${CURRENT_IMG:-qwen38-b70:latest}"
CANDIDATE_IMG="${CANDIDATE_IMG:-qwen38-b70:quant-timing}"
CURRENT_IMG_ID_EXPECTED="${CURRENT_IMG_ID_EXPECTED:-sha256:8c6dc0462011e7d4596882009fc7fb1128fbe656cb17a998999cd1e720a2b4de}"
CANDIDATE_IMG_ID_EXPECTED="${CANDIDATE_IMG_ID_EXPECTED:-}"
CTX_SIZE=262144
MODEL="$REPO/models/files/qwen3.8-27b/ud-q4-k-xl-unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf"
MODEL_SIZE=17559178144
MODEL_SHA=3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e
SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-ud-q4-k-xl/serve.sh"
OVERLAY="$REPO/llamacpp/qwen38_ud_q4k_xl_profile_entrypoint.sh"
QUALIFY="$REPO/llamacpp/qualify_qwen38_obliterated_q4km.py"
PROFILE="$REPO/llamacpp/profile_quant_timing_api.py"
PARSE_CENSUS="$REPO/llamacpp/parse_quant_census.py"
PARSE_TIMING="$REPO/llamacpp/parse_quant_timing.py"
ANALYZER="$REPO/llamacpp/analyze_qwen38_ud_q4k_xl_quant_timing.py"
API_KEY_FILE="${API_KEY_FILE:-/mnt/vm_8tb/b70/secrets/dd_api_key}"
LOCK_BASE="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"
CAMPAIGN_NAMES=(
    qwen38_xl_qtiming_current_off
    qwen38_xl_qtiming_candidate_off
    qwen38_xl_qtiming_counts_only
    qwen38_xl_qtiming_timing_64
    qwen38_xl_qtiming_timing_128
    qwen38_xl_qtiming_timing_256
)

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

ancestor_has_pid() {
    local wanted="$1"
    local pid="$$"
    while [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" -gt 1 ]; do
        [ "$pid" = "$wanted" ] && return 0
        [ -r "/proc/$pid/status" ] || return 1
        pid="$(awk '/^PPid:/ { print $2 }' "/proc/$pid/status")"
    done
    return 1
}

require_external_gpu_run() {
    local card owner owner_pid first_pid=""
    for card in 0 1; do
        owner="$LOCK_BASE.$card.owner"
        [ -s "$owner" ] || {
            say "card $card has no gpu-run owner; invoke this action under ./bin/gpu-run"
            return 2
        }
        if [[ "$(<"$owner")" =~ pid=([0-9]+) ]]; then
            owner_pid="${BASH_REMATCH[1]}"
        else
            say "card $card gpu-run owner record is malformed"
            return 2
        fi
        kill -0 "$owner_pid" 2>/dev/null || {
            say "card $card gpu-run owner pid $owner_pid is not alive"
            return 2
        }
        ancestor_has_pid "$owner_pid" || {
            say "card $card lease belongs to pid $owner_pid, not this process tree"
            return 2
        }
        [ -z "$first_pid" ] && first_pid="$owner_pid"
        [ "$owner_pid" = "$first_pid" ] || {
            say "the two card leases have different owners"
            return 2
        }
    done
}

resolve_images() {
    CURRENT_IMG_ID="$(docker image inspect "$CURRENT_IMG" --format '{{.Id}}')"
    CANDIDATE_IMG_ID="$(docker image inspect "$CANDIDATE_IMG" --format '{{.Id}}')"
    [ "$CURRENT_IMG" != "$CANDIDATE_IMG" ] || {
        say "current and candidate tags must differ"
        return 2
    }
    [ "$CURRENT_IMG_ID" != "$CANDIDATE_IMG_ID" ] || {
        say "current and candidate resolve to the same image id"
        return 2
    }
    if [ -n "$CURRENT_IMG_ID_EXPECTED" ]; then
        [ "$CURRENT_IMG_ID" = "$CURRENT_IMG_ID_EXPECTED" ] || return 2
    fi
    if [ -n "$CANDIDATE_IMG_ID_EXPECTED" ]; then
        [ "$CANDIDATE_IMG_ID" = "$CANDIDATE_IMG_ID_EXPECTED" ] || return 2
    fi
}

prepare_output() {
    [ ! -e "$OUT" ] || {
        say "refusing to reuse output directory $OUT"
        return 2
    }
    mkdir -p "$OUT"
    exec > >(tee -a "$OUT/campaign.log") 2>&1
}

metadata() {
    resolve_images
    [ -s "$MODEL" ] || { say "missing model $MODEL"; return 2; }
    [ "$(stat -c %s "$MODEL")" = "$MODEL_SIZE" ] || {
        say "XL model size mismatch"
        return 2
    }
    docker image inspect "$CURRENT_IMG" >"$OUT/current_image_inspect.json"
    docker image inspect "$CANDIDATE_IMG" >"$OUT/candidate_image_inspect.json"
    sha256sum \
        "$REPO/llamacpp/qwen38-b70/patches/quant-census.patch" \
        "$REPO/llamacpp/qwen38-b70/patches/quant-timing.patch" \
        "$REPO/llamacpp/qwen38-b70/Dockerfile" \
        "$REPO/llamacpp/qwen38-b70/build_image.sh" \
        "$REPO/llamacpp/01_qwen38_ud_q4k_xl_quant_timing_campaign.sh" \
        "$OVERLAY" "$PROFILE" "$ANALYZER" >"$OUT/code_sha256.txt"
    python3 - "$OUT/manifest.json" "$CURRENT_IMG" "$CURRENT_IMG_ID" \
        "$CANDIDATE_IMG" "$CANDIDATE_IMG_ID" "$MODEL" "$MODEL_SIZE" "$MODEL_SHA" \
        "$PORT" "$SERVED" "$CTX_SIZE" "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

(out, current_tag, current_id, candidate_tag, candidate_id, model, model_size,
 model_sha, port, served, ctx, git_head) = sys.argv[1:]
payload = {
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "git_head": git_head,
    "host": platform.node(),
    "kernel": platform.release(),
    "images": {
        "current": {"tag": current_tag, "id": current_id},
        "candidate": {"tag": candidate_tag, "id": candidate_id},
    },
    "model": {"path": model, "size": int(model_size), "sha256": model_sha},
    "config": {
        "port": int(port), "served": served, "ctx_size": int(ctx),
        "gpu_count": 2, "mtp": 0, "lab_doors": 0, "p2p": 0,
        "temperature": 0, "decode_tokens": 256, "warmup_tokens": 32,
        "inert_repetitions": 5, "evidence_repetitions": 1,
        "timing_periods": [64, 128, 256], "timing_skip": 4,
        "timing_max_samples": 65536, "restore": False,
    },
}
Path(out).write_text(
    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    encoding="ascii",
)
PY
    say "metadata captured current=$CURRENT_IMG_ID candidate=$CANDIDATE_IMG_ID"
}

active_name=""
active_dir=""

write_endpoint_down() {
    local listener=0 containers=()
    ss -ltnH "sport = :$PORT" | rg -q . && listener=1
    local name
    for name in "${CAMPAIGN_NAMES[@]}"; do
        docker ps --format '{{.Names}}' | rg -qx "$name" && containers+=("$name")
    done
    python3 - "$OUT/endpoint_down.json" "$listener" "${containers[*]}" <<'PY'
import json
import sys
from pathlib import Path

out, listener, containers = sys.argv[1:]
running = containers.split() if containers else []
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

capture_and_stop_active() {
    [ -n "$active_name" ] || return 0
    local name="$active_name" directory="$active_dir"
    local stop_rc log_rc remove_rc
    say "graceful stop $name"
    docker inspect "$name" >"$directory/container_inspect_before_stop.json" 2>/dev/null || true
    set +e
    docker stop --time 60 "$name" >"$directory/docker_stop.log" 2>&1
    stop_rc=$?
    docker inspect "$name" >"$directory/container_inspect_after_stop.json" 2>/dev/null
    docker logs "$name" >"$directory/server.log" 2>&1
    log_rc=$?
    docker rm -f "$name" >"$directory/docker_rm.log" 2>&1
    remove_rc=$?
    set -e
    active_name=""
    active_dir=""
    python3 - "$directory/graceful_stop.json" "$stop_rc" "$log_rc" "$remove_rc" <<'PY'
import json
import sys
from pathlib import Path

out, stop_rc, log_rc, remove_rc = sys.argv[1:]
payload = {
    "passed": stop_rc == "0" and log_rc == "0" and remove_rc == "0",
    "docker_stop_rc": int(stop_rc),
    "docker_logs_rc": int(log_rc),
    "docker_rm_rc": int(remove_rc),
}
Path(out).write_text(
    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    encoding="ascii",
)
raise SystemExit(0 if payload["passed"] else 1)
PY
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    if [ -n "$active_name" ]; then
        capture_and_stop_active || rc=1
    fi
    if [ -d "$OUT" ]; then
        write_endpoint_down || rc=1
    fi
    say "campaign exit rc=$rc artifacts=$OUT endpoint=down restore=never"
    exit "$rc"
}
refuse_occupied_endpoint() {
    local name
    for name in "${CAMPAIGN_NAMES[@]}"; do
        if docker ps -a --format '{{.Names}}' | rg -qx "$name"; then
            say "refusing stale campaign container $name; inspect and remove it explicitly"
            return 2
        fi
    done
    if ss -ltnH "sport = :$PORT" | rg -q .; then
        say "port $PORT is occupied; this campaign never stops unrelated services"
        return 2
    fi
}

write_identity() {
    local directory="$1" expected_image_id="$2" expected_census="$3" expected_period="$4"
    curl -fsS --max-time 15 "${AUTH[@]}" \
        "http://127.0.0.1:$PORT/v1/models" >"$directory/models.json"
    curl -fsS --max-time 15 "${AUTH[@]}" \
        "http://127.0.0.1:$PORT/props" >"$directory/props.json" 2>/dev/null || true
    docker inspect "$active_name" >"$directory/container_inspect.json"
    docker inspect "$active_name" --format '{{range .Config.Env}}{{println .}}{{end}}' \
        | sort >"$directory/container_env.txt"
    local actual_image_id
    actual_image_id="$(docker inspect "$active_name" --format '{{.Image}}')"
    python3 - "$directory" "$SERVED" "$MODEL" "$MODEL_SHA" "$MODEL_SIZE" \
        "$CTX_SIZE" "$expected_image_id" "$actual_image_id" "$expected_census" \
        "$expected_period" <<'PY'
import json
import sys
from pathlib import Path

(directory, served, model, model_sha, model_size, ctx, expected_image_id,
 actual_image_id, census, period) = sys.argv[1:]
directory = Path(directory)
models = json.loads((directory / "models.json").read_text(encoding="utf-8"))
inspect = json.loads((directory / "container_inspect.json").read_text(encoding="utf-8"))[0]
env = {}
for line in (directory / "container_env.txt").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        env[key] = value
checks = {
    "served_id": served in [item.get("id") for item in models.get("data", [])],
    "model_file": env.get("MODEL_FILE") == Path(model).name,
    "model_sha": env.get("MODEL_SHA256") == model_sha,
    "model_size": Path(model).stat().st_size == int(model_size),
    "context": env.get("CTX_SIZE_OVERRIDE") == ctx,
    "tp2": env.get("GPU_COUNT") == "2",
    "mtp_off": env.get("ENABLE_MTP") == "0",
    "lab_doors_off": env.get("LAB_DOORS") == "0",
    "p2p_off": env.get("CCL_TOPO_P2P_ACCESS") == "0",
    "old_profiler_off": env.get("GGML_SYCL_PROFILE", "0") == "0",
    "census": env.get("GGML_SYCL_QUANT_CENSUS") == census,
    "timing_period": env.get("GGML_SYCL_QUANT_TIMING_SAMPLE") == period,
    "image_id": actual_image_id == expected_image_id,
    "restart_disabled": inspect.get("HostConfig", {}).get("RestartPolicy", {}).get("Name")
    == "no",
}
payload = {
    "passed": all(checks.values()),
    "checks": checks,
    "actual": {"image_id": actual_image_id, "env": env},
    "expected": {"image_id": expected_image_id, "census": int(census),
                 "timing_period": int(period)},
}
(directory / "identity.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    encoding="ascii",
)
print(f"IDENTITY passed={payload['passed']} checks={checks}")
raise SystemExit(0 if payload["passed"] else 1)
PY
}

run_deterministic() {
    local directory="$1" label="$2" reference="${3:-}"
    local args=(--base "http://127.0.0.1:$PORT" --model "$SERVED" --tag "$label"
        --out "$directory/deterministic.json")
    [ -n "$reference" ] && args+=(--reference "$reference")
    set +e
    python3 "$QUALIFY" "${args[@]}" 2>&1 | tee "$directory/deterministic.log"
    local rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$rc" >"$directory/deterministic.rc"
}

run_profile() {
    local directory="$1" label="$2" reps="$3"
    python3 "$PROFILE" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
        --tag "$label" --reps "$reps" --gen-tokens 256 --warmup-tokens 32 \
        --out "$directory/decode_profile.json" 2>&1 | tee "$directory/decode_profile.log"
}

start_arm() {
    local label="$1" image="$2" image_id="$3" census="$4" period="$5" reps="$6"
    local directory="$OUT/$label" name="qwen38_xl_qtiming_$label"
    mkdir -p "$directory"
    active_name="$name"
    active_dir="$directory"
    say "ARM $label image=$image census=$census timing_period=$period"
    NAME="$name" PORT="$PORT" SERVED="$SERVED" CTX_SIZE="$CTX_SIZE" \
        IMG="$image" IMG_ID="$image_id" OVERLAY="$OVERLAY" \
        RESTART_POLICY=no \
        ENABLE_MTP=0 LAB_DOORS=0 GGML_SYCL_QUANT_CENSUS="$census" \
        GGML_SYCL_QUANT_TIMING_SAMPLE="$period" GGML_SYCL_QUANT_TIMING_SKIP=4 \
        GGML_SYCL_QUANT_TIMING_MAX=65536 bash "$SHELF" start \
        2>&1 | tee "$directory/start.log"
    write_identity "$directory" "$image_id" "$census" "$period"
    if [ "$label" = "current_off" ]; then
        run_deterministic "$directory" "$label"
    elif [ "$label" = "candidate_off" ]; then
        run_deterministic "$directory" "$label" "$OUT/current_off/deterministic.json"
    fi
    run_profile "$directory" "$label" "$reps"
    capture_and_stop_active
    env -u IMG "$REPO/bin/xpu-health" >"$directory/health_post.log" 2>&1
    if rg -i 'device_lost|out_of_resources|ur_result_error|!!!!|(^|[^a-z])nan([^a-z]|$)' \
        "$directory/server.log"; then
        say "ARM $label contains a fatal marker"
        return 1
    fi
    if [ "$census" = "1" ]; then
        python3 "$PARSE_CENSUS" "$directory/server.log" --write "$directory/census.json" \
            >"$directory/census.stdout.json"
    fi
    if [ "$period" != "0" ]; then
        python3 "$PARSE_TIMING" "$directory/server.log" --write "$directory/timing.json" \
            >"$directory/timing.stdout.json"
    fi
}

run_campaign() {
    require_external_gpu_run
    metadata
    refuse_occupied_endpoint
    [ -s "$API_KEY_FILE" ] || { say "missing API key file $API_KEY_FILE"; return 2; }
    API_KEY="$(<"$API_KEY_FILE")"
    export API_KEY OPENAI_API_KEY="$API_KEY"
    AUTH=(-H "Authorization: Bearer $API_KEY")

    env -u IMG "$REPO/bin/xpu-health" >"$OUT/health_pre.log" 2>&1
    start_arm current_off "$CURRENT_IMG" "$CURRENT_IMG_ID" 0 0 5
    start_arm candidate_off "$CANDIDATE_IMG" "$CANDIDATE_IMG_ID" 0 0 5
    start_arm counts_only "$CANDIDATE_IMG" "$CANDIDATE_IMG_ID" 1 0 1
    start_arm timing_64 "$CANDIDATE_IMG" "$CANDIDATE_IMG_ID" 1 64 1
    start_arm timing_128 "$CANDIDATE_IMG" "$CANDIDATE_IMG_ID" 1 128 1
    start_arm timing_256 "$CANDIDATE_IMG" "$CANDIDATE_IMG_ID" 1 256 1
    write_endpoint_down
    python3 "$ANALYZER" --out-dir "$OUT" --write "$OUT/analysis.json" \
        | tee "$OUT/analysis.stdout.json"
}

case "$ACTION" in
    metadata)
        prepare_output
        metadata
        ;;
    full)
        prepare_output
        trap on_exit EXIT INT TERM
        run_campaign
        ;;
    *)
        echo "usage: $0 {metadata|full}"
        exit 2
        ;;
esac

trap - EXIT INT TERM
