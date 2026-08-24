#!/usr/bin/env bash
# Matched stock Q4_K_M versus Unsloth UD-Q4_K_XL TP=2 campaign.
# Run every action except metadata under: ./bin/gpu-run bash <this-script> ACTION
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-full}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/qwen38_ud_q4k_xl_campaign_$STAMP}"
PORT="${PORT:-18080}"
SERVED="${SERVED:-hotschmoe-dd}"
CTX_SIZE="${CTX_SIZE:-262144}"
REPS="${REPS:-5}"
GEN_TOKENS="${GEN_TOKENS:-256}"
RUN_MTP="${RUN_MTP:-1}"
RUN_HEPLUS="${RUN_HEPLUS:-0}"
RUN_EVIDENCE="${RUN_EVIDENCE:-1}"
FINAL_RESTORE="${FINAL_RESTORE:-xl}"

Q4_SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-q4km/serve.sh"
XL_SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-ud-q4-k-xl/serve.sh"
PROFILE_ENTRYPOINT="$REPO/llamacpp/qwen38_ud_q4k_xl_profile_entrypoint.sh"
PROFILE_CLIENT="$REPO/llamacpp/profile_qwen38_api.py"
QUALIFY="$REPO/llamacpp/qualify_qwen38_obliterated_q4km.py"
ANALYZE="$REPO/llamacpp/analyze_qwen38_ud_q4k_xl.py"
INSPECT_GGUF="$REPO/llamacpp/inspect_gguf_quant_mix.py"
API_KEY_FILE="${API_KEY_FILE:-/mnt/vm_8tb/b70/secrets/dd_api_key}"

Q4_MODEL="$REPO/models/files/qwen3.8-27b/q4km-ggml-org/Qwen3.8-27B-Q4_K_M.gguf"
Q4_SIZE=18973870432
Q4_SHA=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
XL_MODEL="$REPO/models/files/qwen3.8-27b/ud-q4-k-xl-unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf"
XL_SIZE=17559178144
XL_SHA=3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e

mkdir -p "$OUT"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

if [ -s "$API_KEY_FILE" ]; then
    API_KEY="$(<"$API_KEY_FILE")"
    export API_KEY OPENAI_API_KEY="$API_KEY"
    AUTH=(-H "Authorization: Bearer $API_KEY")
else
    API_KEY=""
    export API_KEY OPENAI_API_KEY="EMPTY"
    AUTH=()
fi

active_name=""
active_dir=""
campaign_touched=0
restore_complete=0

capture_and_stop_active() {
    if [ -z "$active_name" ]; then
        return 0
    fi
    say "graceful stop $active_name"
    docker inspect "$active_name" >"$active_dir/container_inspect_before_stop.json" 2>/dev/null || true
    docker stop --time 60 "$active_name" >"$active_dir/docker_stop.log" 2>&1 || true
    docker logs "$active_name" >"$active_dir/server.log" 2>&1 || true
    docker rm -f "$active_name" >/dev/null 2>&1 || true
    active_name=""
    active_dir=""
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    if [ -n "$active_name" ]; then
        capture_and_stop_active
    fi
    if [ "$rc" -ne 0 ]; then
        if [ "$restore_complete" = "1" ]; then
            say "campaign exited rc=$rc after completing FINAL_RESTORE=$FINAL_RESTORE"
        elif [ "$campaign_touched" = "1" ]; then
            say "campaign exited rc=$rc; endpoint intentionally left down"
        else
            say "campaign exited rc=$rc before any GPU or endpoint change"
        fi
    fi
    exit "$rc"
}
trap on_exit EXIT INT TERM

metadata() {
    say "inspect pinned GGUF identities and quant mixes"
    python3 "$INSPECT_GGUF" \
        --out "$OUT/gguf_quant_mix.json" \
        --expect "$Q4_MODEL:$Q4_SIZE:$Q4_SHA" \
        --expect "$XL_MODEL:$XL_SIZE:$XL_SHA" \
        "$Q4_MODEL" "$XL_MODEL"
    jq '{passed, models: [.models[] | {
        path, size_bytes, sha256,
        metadata: .gguf.metadata,
        types,
        q4k_dense_swiglu_coverage: {
          complete_gate_up_pairs: .q4k_dense_swiglu_coverage.complete_gate_up_pairs,
          non_q4k_or_partial_pairs: .q4k_dense_swiglu_coverage.non_q4k_or_partial_pairs
        },
        mtp_tensor_count: (.mtp_tensor_names | length)
    }]}' "$OUT/gguf_quant_mix.json" | tee "$OUT/gguf_quant_mix_summary.json"
}

stop_known_servers() {
    campaign_touched=1
    local names=(
        qwen38_stock_q4km_tp2
        qwen38_unsloth_ud_q4k_xl_tp2
        qwen38_xl_campaign_q4km_mtp0
        qwen38_xl_campaign_xl_mtp0
        qwen38_xl_campaign_xl_mtp1
        qwen38_xl_campaign_evidence
    )
    for name in "${names[@]}"; do
        if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
            say "stop prior $name"
            docker stop --time 60 "$name" >/dev/null 2>&1 || true
            docker rm -f "$name" >/dev/null 2>&1 || true
        fi
    done
    if ss -ltnH "sport = :$PORT" | grep -q .; then
        say "port $PORT remains occupied by an unknown service"
        return 2
    fi
}

write_identity() {
    local directory="$1"
    local expected_path="$2"
    local expected_file
    expected_file="$(basename "$expected_path")"
    local expected_sha="$3"
    local expected_size="$4"
    local expected_mtp="$5"
    curl -fsS --max-time 15 "${AUTH[@]}" \
        "http://127.0.0.1:$PORT/v1/models" >"$directory/models.json"
    curl -fsS --max-time 15 "${AUTH[@]}" \
        "http://127.0.0.1:$PORT/props" >"$directory/props.json" 2>/dev/null || true
    docker inspect "$active_name" >"$directory/container_inspect.json"
    docker inspect "$active_name" --format '{{range .Config.Env}}{{println .}}{{end}}' \
        | sort >"$directory/container_env.txt"
    python3 - "$directory" "$SERVED" "$expected_path" "$expected_sha" \
        "$expected_size" "$expected_mtp" "$CTX_SIZE" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
served, expected_path, expected_sha, expected_size, expected_mtp, expected_ctx = sys.argv[2:]
expected_file = Path(expected_path).name
models = json.loads((directory / "models.json").read_text())
env = {}
for line in (directory / "container_env.txt").read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        env[key] = value
model_ids = [item.get("id") for item in models.get("data", [])]
checks = {
    "served_id": served in model_ids,
    "model_file": env.get("MODEL_FILE") == expected_file,
    "model_sha256": env.get("MODEL_SHA256") == expected_sha,
    "model_size": Path(expected_path).stat().st_size == int(expected_size),
    "context": env.get("CTX_SIZE_OVERRIDE") == expected_ctx,
    "mtp": env.get("ENABLE_MTP") == expected_mtp,
    "tp2": env.get("GPU_COUNT") == "2",
    "p2p_off": env.get("CCL_TOPO_P2P_ACCESS") == "0",
    "lab_doors_off": env.get("LAB_DOORS") == "0",
}
output = {
    "passed": all(checks.values()),
    "checks": checks,
    "served_ids": model_ids,
    "expected": {
        "file": expected_file,
        "host_path": expected_path,
        "sha256": expected_sha,
        "size": int(expected_size),
        "context": int(expected_ctx),
        "mtp": int(expected_mtp),
    },
}
(directory / "identity.json").write_text(
    json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    encoding="ascii",
)
print(f"IDENTITY -> passed={output['passed']} checks={checks}")
raise SystemExit(0 if output["passed"] else 1)
PY
}

run_deterministic() {
    local directory="$1"
    local tag="$2"
    local reference="${3:-}"
    local args=(
        --base "http://127.0.0.1:$PORT"
        --model "$SERVED"
        --tag "$tag"
        --out "$directory/deterministic.json"
    )
    [ -n "$reference" ] && args+=(--reference "$reference")
    set +e
    python3 "$QUALIFY" "${args[@]}" 2>&1 | tee "$directory/deterministic.log"
    local rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$rc" >"$directory/deterministic.rc"
    say "deterministic tag=$tag rc=$rc (recorded; full quality policy is analyzed later)"
}

run_heplus() {
    local directory="$1"
    say "HumanEval+ 164 on XL MTP-off"
    "$REPO/evals/.venv/bin/python" "$REPO/evals/orchestrator/run_evals.py" \
        --endpoint "http://127.0.0.1:$PORT/v1" \
        --model "$SERVED" --quant ud-q4-k-xl-unsloth-tp2-mtp0 \
        --tiers 1 --tier1-dataset humaneval --limit 164 --max-tokens 2048 \
        2>&1 | tee "$directory/heplus.log"
    local summary_path
    summary_path="$(sed -n 's/^== done\. summary -> //p' "$directory/heplus.log" | tail -1)"
    [ -n "$summary_path" ] && [ -f "$summary_path" ] || {
        say "could not resolve HumanEval+ summary path"
        return 1
    }
    printf '%s\n' "$summary_path" >"$directory/heplus_summary_path.txt"
}

run_timed_arm() {
    local label="$1"
    local shelf="$2"
    local expected_file="$3"
    local expected_sha="$4"
    local expected_size="$5"
    local mtp="$6"
    local reference="${7:-}"
    local directory="$OUT/$label"
    local name="qwen38_xl_campaign_${label}"
    mkdir -p "$directory"
    active_name="$name"
    active_dir="$directory"

    say "ARM $label start mtp=$mtp"
    "$REPO/bin/xpu-health" | tee "$directory/health_pre.log"
    NAME="$name" PORT="$PORT" SERVED="$SERVED" CTX_SIZE="$CTX_SIZE" \
        OVERLAY="$PROFILE_ENTRYPOINT" ENABLE_MTP="$mtp" \
        LAB_DOORS=0 bash "$shelf" start 2>&1 | tee "$directory/start.log"

    write_identity "$directory" "$expected_file" "$expected_sha" "$expected_size" "$mtp"
    run_deterministic "$directory" "$label" "$reference"

    python3 "$PROFILE_CLIENT" \
        --base "http://127.0.0.1:$PORT" --model "$SERVED" --tag "$label" \
        --reps "$REPS" --gen-tokens "$GEN_TOKENS" --out "$directory/profile.json" \
        2>&1 | tee "$directory/profile.log"

    API_KEY="$API_KEY" python3 "$REPO/vllm/nvfp4/bench_code.py" \
        "http://127.0.0.1:$PORT/v1" "$SERVED" 1 "$GEN_TOKENS" "$REPS" \
        2>&1 | tee "$directory/bench_code_c1.log"

    if [ "$label" = "xl_mtp0" ] && [ "$RUN_HEPLUS" = "1" ]; then
        run_heplus "$directory"
    fi

    capture_and_stop_active
    "$REPO/bin/xpu-health" | tee "$directory/health_post.log"
    if rg -i 'device_lost|out_of_resources|ur_result_error|!!!!|(^|[^a-z])nan([^a-z]|$)' \
        "$directory/server.log"; then
        say "ARM $label fatal marker"
        return 1
    fi
    say "ARM $label complete"
}

run_evidence_arm() {
    local directory="$OUT/xl_mtp0"
    local name="qwen38_xl_campaign_evidence"
    mkdir -p "$directory"
    active_name="$name"
    active_dir="$directory/evidence_tmp"
    mkdir -p "$active_dir"
    say "EVIDENCE XL start verbose+census, excluded from timing"
    NAME="$name" PORT="$PORT" SERVED="$SERVED" CTX_SIZE="$CTX_SIZE" \
        OVERLAY="$PROFILE_ENTRYPOINT" ENABLE_MTP=0 LAB_DOORS=2 \
        bash "$XL_SHELF" start 2>&1 | tee "$directory/evidence_start.log"
    curl -fsS --max-time 600 "${AUTH[@]}" -H 'content-type: application/json' \
        -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that merges two sorted lists.\"}],\"max_tokens\":128,\"temperature\":0,\"ignore_eos\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" >"$directory/evidence_response.json"
    docker stop --time 60 "$active_name" >"$directory/evidence_stop.log" 2>&1 || true
    docker logs "$active_name" >"$directory/evidence_server.log" 2>&1 || true
    docker rm -f "$active_name" >/dev/null 2>&1 || true
    active_name=""
    active_dir=""
    "$REPO/bin/xpu-health" | tee "$directory/evidence_health_post.log"
    say "EVIDENCE XL complete"
}

restore() {
    case "$FINAL_RESTORE" in
        xl)
            say "final restore XL MTP-off production"
            PORT="$PORT" SERVED="$SERVED" ENABLE_MTP=0 LAB_DOORS=0 \
                bash "$XL_SHELF" start 2>&1 | tee "$OUT/final_restore_xl.log"
            ;;
        q4km)
            say "explicit rollback restore Q4_K_M"
            PORT="$PORT" SERVED="$SERVED" ENABLE_MTP=0 LAB_DOORS=0 \
                bash "$Q4_SHELF" start 2>&1 | tee "$OUT/final_restore_q4km.log"
            ;;
        none)
            say "FINAL_RESTORE=none; endpoint remains down"
            ;;
        *)
            say "invalid FINAL_RESTORE=$FINAL_RESTORE (xl|q4km|none)"
            return 2
            ;;
    esac
    restore_complete=1
}

analyze() {
    local args=(--out-dir "$OUT" --write "$OUT/analysis.json")
    [ "$RUN_MTP" = "1" ] && args+=(--run-mtp)
    [ "$RUN_HEPLUS" = "1" ] && args+=(--run-heplus)
    python3 "$ANALYZE" "${args[@]}" | tee "$OUT/analysis.stdout.json"
}

run_campaign() {
    metadata
    stop_known_servers
    "$REPO/bin/xpu-health" | tee "$OUT/health_pre_campaign.log"
    printf '%s\n' \
        "$REPO/evals/results/20260817T231333Z___models_Qwen3.8-27B-Q4_K_M.gguf__q4km-0xsero-sycl-tp2/summary.json" \
        >"$OUT/q4km_pinned_heplus_summary_path.txt"

    run_timed_arm q4km_mtp0 "$Q4_SHELF" "$Q4_MODEL" "$Q4_SHA" "$Q4_SIZE" 0
    run_timed_arm xl_mtp0 "$XL_SHELF" "$XL_MODEL" "$XL_SHA" "$XL_SIZE" 0
    if [ "$RUN_MTP" = "1" ]; then
        run_timed_arm xl_mtp1 "$XL_SHELF" "$XL_MODEL" "$XL_SHA" "$XL_SIZE" 1 \
            "$OUT/xl_mtp0/deterministic.json"
    fi
    if [ "$RUN_EVIDENCE" = "1" ]; then
        run_evidence_arm
    fi

    set +e
    analyze
    local analysis_rc=$?
    set -e
    restore
    return "$analysis_rc"
}

case "$ACTION" in
    metadata)
        metadata
        ;;
    full)
        run_campaign
        ;;
    restore-xl)
        FINAL_RESTORE=xl
        stop_known_servers
        restore
        ;;
    restore-q4km)
        FINAL_RESTORE=q4km
        stop_known_servers
        restore
        ;;
    stop)
        stop_known_servers
        ;;
    *)
        echo "usage: $0 {metadata|full|restore-xl|restore-q4km|stop}"
        exit 2
        ;;
esac

trap - EXIT INT TERM
