# sglang-xpu:mtp-0515 = sglang-xpu:woq-0515 + the FINAL (4-gate) NEXTN/MTP XPU enablement, re-grafted
# onto v0.5.15.post1. 0.5.15 re-graft of sglang-xpu:mtp. See sglang/SGLANG_0515_UPGRADE.md.
#
# Contents over :woq-0515:
#   - mtp_tree_xpu.py  : canonical MTP shim. NOTE re 0.5.15 drift -- upstream v0.5.15 NATIVELY added XPU to
#                        BOTH the eagle_sample greedy branch AND assign_extend_cache_locs_func, so DOMINO 1
#                        and DOMINO 4 are now no-ops (the greedy-force re-exec prints "SKIPPED (condition not
#                        found)" because upstream changed the string -- benign; upstream already does our fix).
#                        The remaining gates (build_tree/verify torch fallbacks, mamba-scatter is_cuda strip,
#                        top_p_renorm, extra_buffer un-gate) still apply. Opt-in via B70_XPU_MTP=1.
#   - memory_pool.py   : the v0.5.15.post1 upstream memory_pool.py (3749 lines) RE-DERIVED with the 3-site
#                        device="cuda" -> device=device fix for the spec-decode mamba state cache. This is
#                        NOT the stale 0.5.6 file (which was 2959 lines) -- baking that would revert ~790
#                        lines of upstream mem_cache changes (dedup sliding-window conv layout, etc).
# woq_shim.py + the woqgemm int4 path are inherited from :woq-0515. xpu_cudagraph.py is baked but INERT for
# the W8A8/MTP drivers (only imported under B70_XPU_CUDAGRAPH=1).
FROM sglang-xpu:woq-0515
COPY mtp_tree_xpu.py /opt/venv/lib/python3.12/site-packages/mtp_tree_xpu.py
COPY memory_pool.py  /opt/venv/lib/python3.12/site-packages/sglang/srt/mem_cache/memory_pool.py
COPY woq_shim.py      /opt/venv/lib/python3.12/site-packages/woq_shim.py
COPY xpu_cudagraph.py /opt/venv/lib/python3.12/site-packages/xpu_cudagraph.py
