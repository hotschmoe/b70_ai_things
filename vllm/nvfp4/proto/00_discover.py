# 00_discover.py -- map the sub-int8 capability surface of the int8g-v0240 image.
# CHEAP introspection only: import versions, list torch.ops, probe triton presence.
# No GPU kernels executed here (safe without holding a heavy lease, but run under it anyway).
import sys, importlib

def line(*a): print(*a, flush=True)

line("=== python/torch/ipex/oneDNN versions ===")
import torch
line("torch", torch.__version__)
try:
    import intel_extension_for_pytorch as ipex
    line("ipex", ipex.__version__)
except Exception as e:
    line("ipex import FAIL:", type(e).__name__, str(e)[:120])
line("xpu available:", torch.xpu.is_available() if hasattr(torch, "xpu") else "no xpu attr")
try:
    line("oneDNN (torch backend) version:", torch.backends.mkldnn.__dict__.get("version", "n/a"))
except Exception as e:
    line("mkldnn ver fail", e)

line("\n=== torch.ops namespaces ===")
for ns in dir(torch.ops):
    if ns.startswith("_"):
        continue
    line("  ns:", ns)

line("\n=== search torch.ops for int4/woq/gemm/fp4/awq/gptq ===")
needles = ["int4", "woq", "gemm", "fp4", "awq", "gptq", "int8", "dequant", "quant"]
for ns_name in dir(torch.ops):
    try:
        ns = getattr(torch.ops, ns_name)
    except Exception:
        continue
    for attr in dir(ns):
        low = attr.lower()
        if any(n in low for n in needles):
            line(f"  torch.ops.{ns_name}.{attr}")

line("\n=== _xpu_C ops (import the kernels pkg) ===")
try:
    import vllm_xpu_kernels._xpu_C  # noqa
    for attr in dir(torch.ops._xpu_C):
        if not attr.startswith("__"):
            line("  torch.ops._xpu_C." + attr)
except Exception as e:
    line("_xpu_C import FAIL:", type(e).__name__, str(e)[:160])

line("\n=== triton presence ===")
try:
    import triton, triton.language as tl
    line("triton", triton.__version__)
    line("triton file", triton.__file__)
    # backends
    try:
        from triton.runtime import driver
        line("triton active driver:", driver.active)
    except Exception as e:
        line("driver probe fail", str(e)[:120])
except Exception as e:
    line("triton import FAIL:", type(e).__name__, str(e)[:160])

line("\n=== oneDNN direct (ctypes find lib) ===")
import glob, os
for pat in ["/opt/intel/oneapi/**/libdnnl*.so*", "/usr/**/libdnnl*.so*", "/opt/venv/**/libdnnl*.so*"]:
    for f in glob.glob(pat, recursive=True)[:5]:
        line("  dnnl:", f)
line("DONE")
