#!/usr/bin/env bash
# Attach-after-load VTune gpu-offload mechanism gate for XL TP=2 decode.
# Run only as: ./bin/gpu-run bash llamacpp/03_qwen38_ud_q4k_xl_vtune_attach_gate.sh full
# This script never restores production and leaves the endpoint down.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-full}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/qwen38_ud_q4k_xl_vtune_attach_$STAMP}"
PORT="${PORT:-18080}"
SERVED="${SERVED:-hotschmoe-dd}"
IMG="${IMG:-qwen38-b70:quant-timing}"
IMG_ID_EXPECTED="${IMG_ID_EXPECTED:-sha256:5029a9d394eacd46b48686b564fcc93a410c27a6b1064630008eaec83ef748d1}"
MODEL_DIR="$REPO/models/files/qwen3.8-27b/ud-q4-k-xl-unsloth"
MODEL_FILE=Qwen3.8-27B-UD-Q4_K_XL.gguf
MODEL_SIZE=17559178144
MODEL_SHA=3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e
OVERLAY="$REPO/llamacpp/qwen38_ud_q4k_xl_profile_entrypoint.sh"
API_HELPER="$REPO/llamacpp/profile_vtune_decode_api.py"
PARSE_CENSUS="$REPO/llamacpp/parse_quant_census.py"
PARSE_TASKS="$REPO/llamacpp/parse_vtune_quant_tasks.py"
ANALYZER="$REPO/llamacpp/analyze_qwen38_ud_q4k_xl_vtune.py"
API_KEY_FILE="${API_KEY_FILE:-/mnt/vm_8tb/b70/secrets/dd_api_key}"
LOCK_BASE="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"
VTUNE_BIN=/opt/intel/oneapi/vtune/2025.10/bin64/vtune
NAMES=(qwen38_xl_vtune_attach_reference qwen38_xl_vtune_attach_trace)
active_name=""
active_dir=""

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

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
    local card owner owner_pid first_pid=""
    for card in 0 1; do
        owner="$LOCK_BASE.$card.owner"
        [ -s "$owner" ] || { say "card $card has no gpu-run owner"; return 2; }
        [[ "$(<"$owner")" =~ pid=([0-9]+) ]] || return 2
        owner_pid="${BASH_REMATCH[1]}"
        kill -0 "$owner_pid" 2>/dev/null || return 2
        ancestor_has_pid "$owner_pid" || return 2
        [ -z "$first_pid" ] && first_pid="$owner_pid"
        [ "$owner_pid" = "$first_pid" ] || return 2
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
payload = {"passed": listener == "0" and not running,
           "port_listener": bool(int(listener)), "running_campaign_containers": running}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
raise SystemExit(0 if payload["passed"] else 1)
PY
}

capture_container() {
    local name="$1" directory="$2"
    docker inspect "$name" >"$directory/container_inspect.json" 2>/dev/null || true
    docker inspect "$name" --format '{{range .Config.Env}}{{println .}}{{end}}' \
        | sort >"$directory/container_env.txt" 2>/dev/null || true
    docker logs "$name" >"$directory/server.log" 2>&1 || true
}

cleanup_active() {
    [ -n "$active_name" ] || return 0
    say "cleanup active container $active_name"
    if [ -d "$active_dir/profile/result" ]; then
        docker exec "$active_name" bash -lc \
            ". /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; $VTUNE_BIN -command stop -r /profile/result" \
            >"$active_dir/cleanup_vtune_stop.log" 2>&1 || true
    fi
    docker stop --time 60 "$active_name" >"$active_dir/cleanup_stop.log" 2>&1 || true
    capture_container "$active_name" "$active_dir"
    docker rm -f "$active_name" >"$active_dir/cleanup_rm.log" 2>&1 || true
    active_name=""
    active_dir=""
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    cleanup_active || rc=1
    [ -d "$OUT" ] && write_endpoint_down || rc=1
    if [ -d "$OUT" ] && [ ! -s "$OUT/health_final.log" ]; then
        env -u IMG "$REPO/bin/xpu-health" >"$OUT/health_final.log" 2>&1 || rc=1
    fi
    say "exit rc=$rc artifacts=$OUT endpoint=down restore=never"
    exit "$rc"
}

prepare() {
    [ ! -e "$OUT" ] || { say "refuse existing output $OUT"; return 2; }
    mkdir -p "$OUT"
    exec > >(tee -a "$OUT/campaign.log") 2>&1
    [ -s "$MODEL_DIR/$MODEL_FILE" ] || return 2
    [ "$(stat -c %s "$MODEL_DIR/$MODEL_FILE")" = "$MODEL_SIZE" ] || return 2
    [ -x "$OVERLAY" ] || return 2
    [ -s "$API_KEY_FILE" ] || return 2
    local image_id
    image_id="$(docker image inspect "$IMG" --format '{{.Id}}')"
    [ "$image_id" = "$IMG_ID_EXPECTED" ] || {
        say "image id mismatch actual=$image_id expected=$IMG_ID_EXPECTED"
        return 2
    }
    docker image inspect "$IMG" >"$OUT/image_inspect.json"
    sha256sum "$OVERLAY" "$API_HELPER" "$PARSE_CENSUS" "$PARSE_TASKS" \
        "$ANALYZER" "$REPO/llamacpp/03_qwen38_ud_q4k_xl_vtune_attach_gate.sh" \
        "$REPO/llamacpp/qwen38-b70/patches/quant-census.patch" \
        "$REPO/llamacpp/qwen38-b70/patches/quant-timing.patch" >"$OUT/code_sha256.txt"
    python3 - "$OUT/manifest.json" "$IMG" "$image_id" "$MODEL_FILE" "$MODEL_SIZE" \
        "$MODEL_SHA" "$PORT" "$SERVED" "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

out, tag, image_id, model_file, size, sha, port, served, git_head = sys.argv[1:]
payload = {
    "captured_utc": datetime.now(timezone.utc).isoformat(), "git_head": git_head,
    "host": platform.node(), "kernel": platform.release(),
    "image": {"tag": tag, "id": image_id},
    "model": {"file": model_file, "size": int(size), "sha256": sha},
    "config": {"port": int(port), "served": served, "tp": 2, "context": 262144,
               "mtp": False, "lab_doors": False, "p2p": False,
               "warmup_tokens": 32, "measure_tokens": 512,
               "vtune": "gpu-offload", "collection_mode": "attach_after_load",
               "target_pid": 1, "cap_add": ["SYS_PTRACE"],
               "seccomp": "default", "restore": False},
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
}

refuse_occupied() {
    local name
    ss -ltnH "sport = :$PORT" | rg -q . && { say "port $PORT occupied"; return 2; }
    for name in "${NAMES[@]}"; do
        docker ps -a --format '{{.Names}}' | rg -qx "$name" && {
            say "stale campaign container $name"; return 2;
        }
    done
    return 0
}

wait_healthy() {
    local deadline=$((SECONDS + 1200))
    while [ "$SECONDS" -lt "$deadline" ]; do
        docker ps --format '{{.Names}}' | rg -qx "$active_name" || return 1
        curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 5
    done
    return 1
}

start_arm() {
    local label="$1" attach="$2" name directory
    local profile_mount=() cap_args=()
    name="qwen38_xl_vtune_attach_$label"
    if [ "$attach" = "1" ]; then
        directory="$OUT/vtune"
        mkdir -p "$directory/profile"
        profile_mount=(-v "$directory/profile:/profile")
        cap_args=(--cap-add=SYS_PTRACE)
    else
        directory="$OUT/reference"
        mkdir -p "$directory"
    fi
    active_name="$name"
    active_dir="$directory"
    say "start arm=$label attach_after_load=$attach"
    docker run -d --name "$name" --restart no "${cap_args[@]}" \
        --device /dev/dri --ipc=host --shm-size 8g \
        -v /dev/dri/by-path:/dev/dri/by-path:ro \
        -v "$MODEL_DIR:/models:ro" -v "$OVERLAY:/entrypoint.sh:ro" \
        -v "$API_KEY_FILE:/run/secrets/dd_api_key:ro" "${profile_mount[@]}" \
        -e CCL_TOPO_P2P_ACCESS=0 -e MODELS_DIR=/models \
        -e MODEL_FILE="$MODEL_FILE" -e MODEL_SHA256="$MODEL_SHA" \
        -e SERVED="$SERVED" -e API_KEY_FILE=/run/secrets/dd_api_key \
        -e GPU_COUNT=2 -e CTX_SIZE_OVERRIDE=262144 -e PARALLEL=1 \
        -e BATCH=1024 -e UBATCH=256 -e ENABLE_MTP=0 -e LAB_DOORS=0 \
        -e GGML_SYCL_QUANT_CENSUS=1 -e GGML_SYCL_QUANT_TIMING_SAMPLE=0 \
        -e GGML_SYCL_PROFILE=0 -e GGML_SYCL_DEBUG=0 \
        -e PROFILE_VERBOSE=0 -e PROFILE_STATS=0 -e VTUNE_GPU_OFFLOAD=0 \
        -e VTUNE_ATTACH_MODE="$attach" -e VTUNE_TARGET_GPU=0:11:0.0,0:68:0.0 \
        -p "$PORT:8010" --entrypoint bash "$IMG" /entrypoint.sh \
        >"$directory/docker_run.id"
    wait_healthy || { docker logs --tail 200 "$name"; return 1; }
    curl -fsS --max-time 15 -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$PORT/v1/models" >"$directory/models.json"
    capture_container "$name" "$directory"
}

run_api() {
    local directory="$1" mode="$2" tokens="$3"
    python3 "$API_HELPER" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
        --mode "$mode" --tokens "$tokens" --out "$directory/$mode.json" \
        2>&1 | tee "$directory/$mode.log"
}

stop_server() {
    local name="$active_name" directory="$active_dir"
    docker stop --time 60 "$name" >"$directory/docker_stop.log" 2>&1
    capture_container "$name" "$directory"
    docker rm "$name" >"$directory/docker_rm.log" 2>&1
    active_name=""
    active_dir=""
}

start_attach() {
    local directory="$OUT/vtune" deadline
    [ "$active_name" = "qwen38_xl_vtune_attach_trace" ] || return 2
    docker exec "$active_name" bash -lc \
        'printf "pid=1 comm="; tr -d "\n" </proc/1/comm; printf "\n"' \
        >"$directory/target_pid.txt"
    rg -q '^pid=1 comm=llama-server$' "$directory/target_pid.txt" || return 2
    docker exec "$active_name" "$VTUNE_BIN" -version \
        >"$directory/profile/vtune_version.txt"
    docker exec -d "$active_name" bash -lc '
        set +e
        . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
        /opt/intel/oneapi/vtune/2025.10/bin64/vtune -collect gpu-offload \
          -knob collect-programming-api=true \
          -knob enable-tasks-stack-collection=false \
          -knob enable-stack-collection=false \
          -knob enable-characterization-insights=false \
          -knob dump-compute-task-binaries=false \
          -knob target-gpu="$VTUNE_TARGET_GPU" \
          -data-limit=1000 -finalization-mode=full \
          -target-pid 1 -result-dir /profile/result \
          > /profile/attach.log 2>&1
        rc=$?
        printf "%s\n" "$rc" > /profile/attach.rc
    '
    deadline=$((SECONDS + 60))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if [ -f "$directory/profile/attach.rc" ]; then
            say "VTune attach exited before collection-ready"
            return 1
        fi
        if [ -f "$directory/profile/attach.log" ] && \
                rg -q 'Collection started' "$directory/profile/attach.log"; then
            break
        fi
        sleep 1
    done
    [ -f "$directory/profile/attach.log" ] && \
        rg -q 'Collection started' "$directory/profile/attach.log" || return 1
    docker exec "$active_name" bash -lc \
        ". /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; $VTUNE_BIN -command status -r /profile/result" \
        >"$directory/vtune_status.log" 2>&1
    awk '$2 == 1 && $3 == "RESUME" && $4 == "llama-server" { found=1 }
         END { exit !found }' "$directory/vtune_status.log"
}

stop_attach() {
    local directory="$OUT/vtune" stop_rc attach_rc="" deadline server_alive=0 health_rc=0
    set +e
    docker exec "$active_name" bash -lc \
        ". /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; $VTUNE_BIN -command stop -r /profile/result" \
        >"$directory/vtune_stop.log" 2>&1
    stop_rc=$?
    set -e
    deadline=$((SECONDS + 300))
    while [ "$SECONDS" -lt "$deadline" ]; do
        [ -s "$directory/profile/attach.rc" ] && break
        docker inspect "$active_name" --format '{{.State.Running}}' 2>/dev/null \
            | rg -qx true || break
        sleep 2
    done
    [ -s "$directory/profile/attach.rc" ] && attach_rc="$(<"$directory/profile/attach.rc")"
    docker exec "$active_name" kill -0 1 >/dev/null 2>&1 && server_alive=1
    curl -fsS --max-time 10 "http://127.0.0.1:$PORT/health" \
        >"$directory/health_after_detach.json" 2>"$directory/health_after_detach.err" || health_rc=$?
    python3 - "$directory/stop_contract.json" "$stop_rc" "$attach_rc" \
        "$server_alive" "$health_rc" <<'PY'
import json
import sys
from pathlib import Path

out, stop_rc, attach_rc, alive, health_rc = sys.argv[1:]
payload = {
    "passed": stop_rc == "0" and attach_rc == "0" and alive == "1" and health_rc == "0",
    "vtune_stop_rc": int(stop_rc), "attach_rc": int(attach_rc) if attach_rc else None,
    "server_alive_after_detach": alive == "1", "health_after_detach_rc": int(health_rc),
    "target_owned_by_vtune": False,
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
raise SystemExit(0 if payload["passed"] else 1)
PY
}

generate_reports() {
    local directory="$OUT/vtune" profile="$OUT/vtune/profile"
    docker run --rm --entrypoint bash -v "$profile:/profile" "$IMG" -lc \
        ". /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
         $VTUNE_BIN -report summary -r /profile/result -format csv \
           -csv-delimiter comma -report-output /profile/summary.csv; \
         $VTUNE_BIN -report hotspots -r /profile/result \
           -group-by=gpu-adapter,computing-task-offload -format csv \
           -csv-delimiter comma -report-output /profile/tasks.csv" \
        >"$directory/vtune_report.log" 2>&1
    cp "$profile/summary.csv" "$directory/summary.csv"
    cp "$profile/tasks.csv" "$directory/tasks.csv"
    python3 "$PARSE_TASKS" "$directory/tasks.csv" \
        --require-family q3_K --require-family q4_K --require-family q5_K \
        --require-family q6_K --require-family q8_0 --require-family iq3_s \
        --require-family iq4_nl --require-family iq4_xs \
        --write "$directory/tasks.json" >"$directory/tasks.stdout.json"
}

run_gate() {
    require_external_gpu_run
    prepare
    trap on_exit EXIT INT TERM
    refuse_occupied
    API_KEY="$(<"$API_KEY_FILE")"
    export API_KEY OPENAI_API_KEY="$API_KEY"
    env -u IMG "$REPO/bin/xpu-health" >"$OUT/health_pre.log" 2>&1

    start_arm reference 0
    run_api "$OUT/reference" warmup 32
    run_api "$OUT/reference" measure 512
    stop_server
    write_endpoint_down
    env -u IMG "$REPO/bin/xpu-health" >"$OUT/reference/health_post.log" 2>&1

    start_arm trace 1
    run_api "$OUT/vtune" warmup 32
    start_attach
    run_api "$OUT/vtune" measure 512
    stop_attach
    stop_server
    write_endpoint_down

    python3 "$PARSE_CENSUS" "$OUT/vtune/server.log" \
        --write "$OUT/vtune/census.json" >"$OUT/vtune/census.stdout.json"
    generate_reports
    write_endpoint_down
    env -u IMG "$REPO/bin/xpu-health" >"$OUT/health_final.log" 2>&1
    cp "$OUT/health_final.log" "$OUT/vtune/health_post.log"
    python3 "$ANALYZER" --out-dir "$OUT" --write "$OUT/analysis.json" \
        | tee "$OUT/analysis.stdout.json"
}

case "$ACTION" in
    metadata)
        prepare
        ;;
    full)
        run_gate
        ;;
    *)
        echo "usage: $0 {metadata|full}"
        exit 2
        ;;
esac

trap - EXIT INT TERM
