#!/usr/bin/env bash
# agentic-eval/configs.sh -- CANONICAL config registry for the 4-way quant comparison.
#
# The whole campaign compares 4 served configs of the Qwen3.6 family, two architectures x two
# quant schemes, so the load-bearing question (does int4 hurt multi-step agentic trajectories more
# than single-shot codegen?) is answered WITHIN architecture:
#     dense:  27b-int4  vs  27b-w8a8
#     moe:    35b-int4  vs  35b-w8a8
# Do NOT read a dense-vs-moe gap as a quant effect; that is a different (model-choice) question.
#
# Each config dispatches to an EXISTING verified shelf recipe under rdy_to_serve/<dir>/serve.sh
# (no new serve logic here). We only override a few env knobs for comparability (see EVAL_* below).
# Served-model-ids encode method+scheme per the CLAUDE.md model-identity rule.
set -uo pipefail

# Order matters for the README table (dense pair, then moe pair).
EVAL_CONFIG_LIST=(27b-int4 27b-w8a8 35b-int4 35b-w8a8)

# eval_config <label> -> sets EVAL_LABEL/ARCH/SCHEME/SERVED/SERVE_DIR/CARDS for that config.
eval_config() {
  case "$1" in
    27b-int4)  EVAL_LABEL=27b-int4;  EVAL_ARCH=dense; EVAL_SCHEME=int4; EVAL_CARDS=1
               EVAL_SERVED=qwen36-27b-int4;                  EVAL_SERVE_DIR=qwen36-27b-int4 ;;
    27b-w8a8)  EVAL_LABEL=27b-w8a8;  EVAL_ARCH=dense; EVAL_SCHEME=w8a8; EVAL_CARDS=2
               EVAL_SERVED=qwen36-27b-w8a8-sqgptq-mtp;       EVAL_SERVE_DIR=qwen36-27b-w8a8-sqgptq-mtp ;;
    35b-int4)  EVAL_LABEL=35b-int4;  EVAL_ARCH=moe;   EVAL_SCHEME=int4; EVAL_CARDS=1
               EVAL_SERVED=qwen36-35b-a3b-int4;              EVAL_SERVE_DIR=qwen36-35b-a3b-int4 ;;
    35b-w8a8)  EVAL_LABEL=35b-w8a8;  EVAL_ARCH=moe;   EVAL_SCHEME=w8a8; EVAL_CARDS=2
               EVAL_SERVED=qwen36-35b-a3b-quark-w8a8-int8;   EVAL_SERVE_DIR=qwen36-35b-a3b-quark-w8a8-int8 ;;
    *) echo "configs.sh: unknown config '$1' (valid: ${EVAL_CONFIG_LIST[*]})" >&2; return 2 ;;
  esac
}

# ---- shared serve overrides applied to ALL four configs (held constant for comparability) --------
# Rationale (validate in smoke; see docs/DESIGN.md):
#  - PORT 18080 matches the project endpoint convention (models.yaml / daily driver).
#  - MAXLEN 16384 + MAXSEQS 4 is the largest context that fits ALL four with their default KV dtype
#    (27B dense int4 single-card fp16-KV is the tightest: ~2 GB/seq at 16k -> 8 GB <= ~14 GiB free).
#    Longer SWE trajectories may truncate, but the setting is identical across configs so the A/B is fair.
#  - TOOLCALL on for all four so BFCL/tau2 native tool-calls work (the int4 recipes set it; the W8A8
#    recipes do not -> we force it). qwen3_coder is the Qwen3.6 XML tool parser.
#  - We deliberately do NOT override KVDTYPE/GRAPH/MTP/PUSH_AR: each recipe keeps its shipped, verified
#    defaults (the realistic "what you would actually serve" config), so speed numbers are honest.
EVAL_PORT="${EVAL_PORT:-18080}"
EVAL_MAXLEN="${EVAL_MAXLEN:-16384}"
EVAL_MAXSEQS="${EVAL_MAXSEQS:-4}"
EVAL_TOOLCALL="${EVAL_TOOLCALL:-1}"
EVAL_TOOLPARSER="${EVAL_TOOLPARSER:-qwen3_coder}"
EVAL_REASONPARSER="${EVAL_REASONPARSER:-qwen3}"

# ---- determinism knobs handed to every harness (mirror evals/configs/models.yaml defaults) -------
AE_TEMPERATURE="${AE_TEMPERATURE:-0.0}"
AE_TOP_P="${AE_TOP_P:-1.0}"
AE_SEED="${AE_SEED:-1234}"
AE_MAX_TOKENS="${AE_MAX_TOKENS:-2048}"
AE_CONCURRENCY="${AE_CONCURRENCY:-4}"   # greedy -> per-request output is concurrency-invariant; this only
                                        # affects wall-clock/throughput, held constant across configs.
