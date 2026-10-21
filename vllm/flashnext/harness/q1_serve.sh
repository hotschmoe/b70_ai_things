#!/bin/bash
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
export B70_AGENT=Q1 X6_PRIORITY_WAIT=0 X6_F1_DIAGNOSTICS=0 X6_REQUIRE_REFERENCES=1
export X6_QSA=${Q1_QSA:-1}
export X6_REFERENCES=$B/q1/strata-references.json
export SKIP_HEALTH_BEFORE=0
export TAG=${Q1_TAG:-q1-r1} PORT=18156 MAX_LEN=65536 MAX_BATCHED=12288 DECODE=64 PREFILL=1024
export O4_RECIPE=$R/vllm/flashnext/harness/o4_recipe.env
export F1_DELTA=$R/vllm/flashnext/harness/f1_prefill_delta.env
export O4_SERVE_EXTRA="B70_GDN_BA_DETERMINISTIC=0 B70_D1_FA_PLAN=0 B70_ONEDNN_DET=0 B70_PREFILL_QSA=$X6_QSA"
unset SERVE_ARGS_EXTRA O4_FA_OVERLAY F1_PYPATH_PRE F1_API_WRAPPER F1_POST_PROBE_SCRIPT
exec bash "$R/vllm/flashnext/harness/x6_serve.sh"
