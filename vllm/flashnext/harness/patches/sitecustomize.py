# Runtime overlay imported because xpu-run.sh puts this directory on
# PYTHONPATH (site.py imports sitecustomize from sys.path).
# Add small XPU workarounds here. Keep this file import-safe: a failure
# here breaks every Python process in the container.
#
# B70_GGUF=1 loads the fork quant config, the b70_gguf model loader, and
# the b70_gguf_xpu ops. Unset, this file does not import vllm.

import os

if os.environ.get("B70_PREFILL_PROFILE"):
    from x3_prefill_profile import install as _x3_prefill_install

    _x3_prefill_install()

# The loader check runs /usr/bin/python3.12, which has no torch. Skip
# there. site.py swallows sitecustomize errors, so a failed register in
# the real interpreter would also be silent; re-raise only when torch
# imported and registration itself failed.

# D2: copy the Shisa MTP head onto the drafter after load.
if os.environ.get("B70_D2_SHISA_MTP") == "1":
    try:
        from d2_mtp_shisa import install as _d2_shisa_install

        _d2_shisa_install()
    except Exception:
        import traceback

        traceback.print_exc()

# D2: int4 GEMV for fp16 trunk projections and lm_head. Decode M only.
if os.environ.get("B70_D2_INT4_LINEAR") == "1":
    try:
        from d2_int4_linear import install as _d2_int4_install

        _d2_int4_install()
    except Exception:
        import traceback

        traceback.print_exc()

# D3: fused GDN decode (conv + delta + RMSNorm). Off unless B70_D3_GDN=1.
if os.environ.get("B70_D3_GDN") == "1":
    try:
        from d3_gdn import install as _d3_gdn_install

        _d3_gdn_install()
    except Exception:
        import traceback

        traceback.print_exc()

# D3: paged full-attention decode. Off unless B70_D3_ATTN=1.
if os.environ.get("B70_D3_ATTN") == "1":
    try:
        from d3_attn import install as _d3_attn_install

        _d3_attn_install()
    except Exception:
        import traceback

        traceback.print_exc()

# D2: per-region XPU-graph device times inside one decode step.
if os.environ.get("B70_D2_PROFILE") == "1":
    try:
        from d2_profile import install as _d2_profile_install

        _d2_profile_install()
    except Exception:
        import traceback

        traceback.print_exc()

# C7: collective count and time. Off unless B70_COLL_LOG is a path.
if os.environ.get("B70_COLL_LOG"):
    try:
        from c7_collectives import install as _c7_coll_install

        _c7_coll_install()
    except Exception:
        import traceback

        traceback.print_exc()

# D1: eager-step timer. The finder does not import torch or vLLM.
if os.environ.get("B70_STEP_TIMER") == "1":
    try:
        from d1_step_timer import install as _d1_timer_install

        _d1_timer_install()
    except Exception:
        import traceback

        traceback.print_exc()

# D1: XPUGraph replay reclaim. Off unless B70_XPU_CG_RECLAIM>0.
if os.environ.get("B70_XPU_CG_RECLAIM", "0") not in ("", "0"):
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None
    if torch is not None:
        try:
            from d1_graph import install_reclaim

            install_reclaim()
        except Exception:
            import traceback

            traceback.print_exc()
            raise
if os.environ.get("B70_GGUF") == "1":
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None
    if torch is not None:
        try:
            from vllm.models.qwen4_exp.gguf import register

            register()
            import b70_gguf_xpu

            b70_gguf_xpu.register()
        except Exception:
            import traceback

            traceback.print_exc()
            raise
# H1 EP sidecar. After the GGUF loader so B70EPMoEMethod can subclass it.
# vLLM stays world size 1 on card 0. The plugin owns card 1.
if os.environ.get("B70_EP") == "1":
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None
    if torch is not None:
        try:
            import b70_ep_xpu

            b70_ep_xpu.register()
        except Exception:
            import traceback

            traceback.print_exc()
            raise
