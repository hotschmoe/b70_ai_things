# 05_ops_probe.py -- EXPERIMENT A.1/A.2: which quant GEMM ops actually resolve on XPU?
# torch binds torch.ops.<ns>.<op> lazily, so we ACCESS each candidate to force-load it.
import torch
import vllm_xpu_kernels._xpu_C  # noqa
def line(*a): print(*a, flush=True)

CANDIDATES = [
    ("_xpu_C", "int8_gemm_w8a16"),   # our production decode op
    ("_xpu_C", "int8_gemm_w8a8"),    # prefill
    ("_xpu_C", "int4_gemm_w4a16"),   # upstream int4 (twos-complement) decode
    ("_xpu_C", "int4_gemm_w4a8"),    # upstream int4 prefill
    ("_xpu_C", "fp4_gemm"),          # MXFP4-only per NVFP4_XPU.md
    ("_xpu_C", "dynamic_per_token_int8_quant"),
    ("aten", "_weight_int4pack_mm"),
    ("aten", "_weight_int8pack_mm"),
]
line("=== op resolution probe (force lazy-load each) ===")
for ns, op in CANDIDATES:
    try:
        obj = getattr(getattr(torch.ops, ns), op)
        # try to get its schema
        try:
            sch = str(obj.default._schema) if hasattr(obj, "default") else "(overloadpacket)"
        except Exception:
            sch = "(no schema readable)"
        line(f"  RESOLVED  torch.ops.{ns}.{op}   {sch[:150]}")
    except Exception as e:
        line(f"  MISSING   torch.ops.{ns}.{op}   {type(e).__name__}: {str(e)[:80]}")
line("DONE")
