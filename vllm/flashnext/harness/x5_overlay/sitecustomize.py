"""X5-only startup overlay; also imported by spawned vLLM workers."""
import os
import runpy
import sys
from pathlib import Path

_patches = Path(__file__).resolve().parents[1] / "patches"
runpy.run_path(str(_patches / "sitecustomize.py"))
if os.environ.get("B70_X5_EVENTLESS_LAZY") == "1":
    import ctypes
    import torch

    if not hasattr(torch, "_x5_eventless_helper"):
        torch._x5_eventless_helper = ctypes.CDLL(
            "/mnt/vm_8tb/b70/build/vllm-flashnext/x5/eventless-lazy/libx5_eventless_helper.so",
            mode=ctypes.RTLD_GLOBAL,
        )
        print("X5_EVENTLESS_HELPER_LOADED_AFTER_TORCH", file=sys.stderr, flush=True)
if os.environ.get("B70_X5_GDN_BF16") == "1":
    # Same D3 wrapper and geometry, with BF16 guards matching the isolated
    # BF16 build. B70_D3_GDN=0 prevents loading the FP16 library first.
    import types

    module = types.ModuleType("x5_d3_bf16")
    source = (_patches / "d3_gdn.py").read_text()
    source = source.replace("torch.float16", "torch.bfloat16")
    source = source.replace("b70_d3", "b70_x5_d3")
    if os.environ.get("B70_X5_GDN_RAGGED") == "1":
        # Only the matching ragged binary uses compact cu-based indexing.
        assert 'or ntok != nseq * nwidth' in source
        source = source.replace('or ntok != nseq * nwidth', 'or ntok > nseq * nwidth')
        source = source.replace('or token_indx.numel() < nseq * nwidth',
                                'or token_indx.numel() < ntok')
    sys.modules[module.__name__] = module
    exec(compile(source, str(_patches / "d3_gdn.py"), "exec"), module.__dict__)
    module.install()
    if os.environ.get("B70_X5_GDN_MIXED") == "1":
        from x5_gdn_mixed import install

        install(module)
if os.environ.get("B70_X5_Q8_LINEAR") == "1":
    from x5_q8_linear import install

    install()
if os.environ.get("B70_X5_Q8_FAST") == "1":
    from x5_q8_fast import install

    install()
if os.environ.get("B70_X5_PACK_LINEAR") == "1":
    from x5_pack_linear import install

    install()
if os.environ.get("B70_X5_CANONICAL") == "1":
    from x5_canonical_moe import install

    install()
if os.environ.get("B70_X5_BATCH_LINEAR") == "1":
    from x5_batch_linear import install

    install()
