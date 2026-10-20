"""X5-only startup overlay; also imported by spawned vLLM workers."""
import os
import runpy
import sys
from pathlib import Path

_patches = Path(__file__).resolve().parents[1] / "patches"
runpy.run_path(str(_patches / "sitecustomize.py"))
if os.environ.get("B70_X5_GDN_BF16") == "1":
    # Same D3 wrapper and geometry, with BF16 guards matching the isolated
    # BF16 build. B70_D3_GDN=0 prevents loading the FP16 library first.
    import types

    module = types.ModuleType("x5_d3_bf16")
    source = (_patches / "d3_gdn.py").read_text()
    source = source.replace("torch.float16", "torch.bfloat16")
    source = source.replace("b70_d3", "b70_x5_d3")
    sys.modules[module.__name__] = module
    exec(compile(source, str(_patches / "d3_gdn.py"), "exec"), module.__dict__)
    module.install()
if os.environ.get("B70_X5_PACK_LINEAR") == "1":
    from x5_pack_linear import install

    install()
