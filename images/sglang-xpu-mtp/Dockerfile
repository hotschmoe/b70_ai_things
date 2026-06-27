# sglang-xpu:mtp = sglang-xpu:woq + the FINAL (4-gate) NEXTN/MTP XPU enablement baked in.
#
# Why a dedicated tag: sglang-xpu:woq already bakes an EARLIER mtp_tree_xpu.py (built before the
# gate-3 top_p_renorm and gate-4 eagle_sample greedy-branch fixes) and does NOT bake the
# memory_pool.py XPU device fix. scripts/128 worked around this by runtime-mounting the fresh
# files. This image bakes them so the rdy_to_serve/qwen36-27b-int4-mtp recipe needs ZERO mounts.
#
# Contents over :woq:
#   - mtp_tree_xpu.py  : all 4 XPU gates (out_cache_loc, mamba-scatter is_cuda, top_p_renorm,
#                        eagle_sample greedy-verify branch). Opt-in via B70_XPU_MTP=1 (woq_shim installs it).
#   - memory_pool.py   : 3-line device="cuda" -> device=device fix for the spec-decode mamba state cache.
# woq_shim.py + the woqgemm int4 path are inherited unchanged from :woq.
# See sglang/PERF.md and JOURNAL 2026-06-27 (MTP COMPLETE).
FROM sglang-xpu:woq
COPY mtp_tree_xpu.py /opt/venv/lib/python3.12/site-packages/mtp_tree_xpu.py
COPY memory_pool.py  /opt/venv/lib/python3.12/site-packages/sglang/srt/mem_cache/memory_pool.py
