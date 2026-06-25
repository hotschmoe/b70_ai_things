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
# Rationale (see docs/DESIGN.md):
#  - PORT 18080 matches the project endpoint convention (models.yaml / daily driver).
#  - TOOLCALL on for all four so BFCL/tau2 native tool-calls work (the int4 recipes set it; the W8A8
#    recipes do not -> we force it). qwen3_coder is the Qwen3.6 XML tool parser.
#  - We deliberately do NOT override KVDTYPE/GRAPH/MTP/PUSH_AR: each recipe keeps its shipped, verified
#    defaults (the realistic "what you would actually serve" config), so speed numbers are honest.
EVAL_PORT="${EVAL_PORT:-18080}"
EVAL_TOOLCALL="${EVAL_TOOLCALL:-1}"
EVAL_TOOLPARSER="${EVAL_TOOLPARSER:-qwen3_coder}"
# Prefix caching ON for the agentic eval (multi-turn agents re-send a growing shared prefix every turn;
# APC skips re-prefill -> big speedup, and greedy output is identical so scores are unchanged). The repo
# default is OFF (clean perf baselines); this is the correct agentic setting. Drives lib.sh PREFIXCACHE.
# Per-config validation: int4 is low-risk; the W8A8 MTP+capture path is the one to confirm in smoke.
EVAL_PREFIX_CACHE="${EVAL_PREFIX_CACHE:-1}"

# ---- THINKING mode (the primary axis) ------------------------------------------------------------
# Qwen3.6 is a hybrid reasoner. DEFAULT = on, because thinking-on is the real agentic-coding workload
# (the model emits <think> traces before tool calls / edits). THINKING=off measures no-think behavior
# (faster, far fewer tokens; some models hold up well without it) -- a deliberate second axis, not the
# default. The mode drives BOTH the serve (reasoning parser) AND the context/token budget: thinking
# needs a much bigger window. The 2026-06-25 smoke proved it -- aider at MAXLEN 16384 / max_tokens 2048
# hit exhausted_context_windows mid-think and scored an artifactual 0.0. So thinking-on sizes up.
#
# KV note: MAXSEQS 2 is chosen so MAXLEN 32768 fits the TIGHTEST config (27B-int4 single-card, fp16 KV
# ~4 GB/seq at 32k -> ~8 GB <= ~12 GiB free at UTIL 0.92). All other configs (TP=2 dense, tiny-KV MoE)
# have ample headroom. We keep each recipe's default KV dtype (no fp8-KV override on the custom W8A8
# kernels). AE_CONCURRENCY tracks MAXSEQS; greedy => scores are concurrency-invariant, so this only
# sets the wall-clock/throughput operating point (held constant across configs).
EVAL_THINKING="${EVAL_THINKING:-on}"
if [ "$EVAL_THINKING" = on ]; then
  EVAL_REASONPARSER="${EVAL_REASONPARSER:-qwen3}"   # split <think> into reasoning_content
  EVAL_NO_THINK=0
  EVAL_MAXLEN="${EVAL_MAXLEN:-65536}"              # 64k, fp16 KV (no fp8 hack). SAFE on the tight config
  EVAL_MAXSEQS="${EVAL_MAXSEQS:-1}"                # (27B-int4 single-card): 64k x 1 seq has the IDENTICAL
                                                   # KV footprint as 32k x 2 seqs, which the 2026-06-25
                                                   # smoke already fit. Doubles context headroom (kills the
                                                   # 32k exhausted_context_windows) at concurrency 1; prefix
                                                   # caching recovers most of the lost concurrency on multi-turn.
  AE_MAX_TOKENS="${AE_MAX_TOKENS:-8192}"           # room for a full think + answer per turn
  AE_CONCURRENCY="${AE_CONCURRENCY:-1}"
else
  # no-think: drop the reasoning parser and signal suppression to harnesses (EVAL_NO_THINK=1). NOTE:
  # full thinking suppression on Qwen3.6 needs the no-think switch (enable_thinking=false / "/no_think");
  # this off-path is EXPERIMENTAL until a live no-think run is validated (docs/DESIGN.md "thinking off").
  EVAL_REASONPARSER="${EVAL_REASONPARSER:-}"
  EVAL_NO_THINK=1
  EVAL_MAXLEN="${EVAL_MAXLEN:-16384}"
  EVAL_MAXSEQS="${EVAL_MAXSEQS:-4}"
  AE_MAX_TOKENS="${AE_MAX_TOKENS:-2048}"
  AE_CONCURRENCY="${AE_CONCURRENCY:-4}"
fi

# ---- determinism knobs handed to every harness (mirror evals/configs/models.yaml defaults) -------
AE_TEMPERATURE="${AE_TEMPERATURE:-0.0}"
AE_TOP_P="${AE_TOP_P:-1.0}"
AE_SEED="${AE_SEED:-1234}"
