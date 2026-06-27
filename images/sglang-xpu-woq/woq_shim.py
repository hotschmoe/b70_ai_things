# woq_shim.py -- wire auto_round_kernel.woqgemm (the proven-fast XPU int4 GEMM, vLLM's 30 t/s path)
# into sglang's GPTQ/AutoRound path on XPU. sglang's GPTQLinearScheme delegates to self.kernel via
# _init_kernel(); we patch _init_kernel to return an XPU WOQ kernel (mirrors the NPU backend pattern).
# The AutoRound int4 ckpt (auto_round:auto_gptq) -> AutoRoundConfig -> GPTQLinearMethod -> GPTQLinearScheme
# -> (patched) WOQ kernel -> woqgemm. GDN/vision/lm_head stay bf16 (AutoRoundConfig.get_layer_config).
# Auto-imported at interpreter startup via a .pth file; safe no-op off-XPU or if deps are missing.
import os


def _install():
    try:
        from sglang.srt.utils import is_xpu
        if not is_xpu():
            return
    except Exception:
        return
    if os.environ.get("WOQ_SHIM_DISABLE") == "1":
        return
    try:
        import torch
        import auto_round_kernel  # noqa: F401  (ensures the XPU .so loads)
        from auto_round_kernel.qlinear import QuantLinearGPTQ
        from sglang.srt.layers.quantization.gptq.schemes.gptq_linear import (
            GPTQLinearScheme,
        )
    except Exception as e:
        print(f"[woq-shim] not installing (import failed): {e}", flush=True)
        return

    class _XpuWoqGptqKernel:
        """Replaces the CUDA gptq_gemm kernel with auto_round_kernel.woqgemm on XPU."""

        def __init__(self, quant_config):
            self.qc = quant_config

        def process_weights_after_loading(self, layer):
            qw = layer.qweight.data          # [in//8, out] int32 (gptq pack)
            qz = layer.qzeros.data           # [in//g, out//8] int32
            sc = layer.scales.data           # [in//g, out]
            dev = qw.device
            in_f = qw.shape[0] * 8
            out_f = qw.shape[1]
            bits = int(getattr(self.qc, "weight_bits", getattr(self.qc, "bits", 4)))
            gs = int(getattr(self.qc, "group_size", 128))
            # AutoRound int is symmetric (sym=True) for our ckpts; desc_act None -> trivial g_idx.
            sym = bool(getattr(self.qc, "sym", True))
            ql = QuantLinearGPTQ(bits, gs, sym, in_f, out_f, False,
                                 weight_dtype=torch.bfloat16)
            ql.qweight.data = qw
            ql.qzeros.data = qz
            ql.scales.data = sc.to(torch.float16)
            ql = ql.to(dev)
            ql.post_init()
            layer._woq = ql
            # free the now-redundant gptq buffers (the ARK blob holds the packed weight)
            empty = torch.nn.Parameter(torch.empty(0, device=dev), requires_grad=False)
            for n in ("qweight", "qzeros", "scales", "g_idx"):
                if hasattr(layer, n):
                    setattr(layer, n, empty)
            torch.xpu.empty_cache() if hasattr(torch, "xpu") else None
            print(f"[woq-shim] WOQ layer ready in={in_f} out={out_f} bits={bits} g={gs}", flush=True)

        def apply(self, layer, x, bias=None):
            out = layer._woq(x)
            if bias is not None:
                out = out + bias
            return out

    def _patched_init_kernel(self, quant_config):
        return _XpuWoqGptqKernel(quant_config)

    GPTQLinearScheme._init_kernel = _patched_init_kernel

    # AutoRound's gptq/awq dispatch calls check_marlin_supported(..., device_capability=None) on XPU,
    # which does `None * int` -> TypeError before reaching the scheme. Guard it: no marlin on XPU.
    try:
        import sglang.srt.layers.quantization.marlin_utils as mu
        _orig_cms = mu.check_marlin_supported

        def _safe_cms(*a, **k):
            dc = k.get("device_capability", a[3] if len(a) > 3 else None)
            if dc is None:
                return False
            return _orig_cms(*a, **k)

        mu.check_marlin_supported = _safe_cms
    except Exception as e:
        print(f"[woq-shim] marlin guard failed: {e}", flush=True)

    # sglang's spec-decode (EAGLE/NEXTN) path hardcodes torch.cuda.{synchronize,Stream,Event,...} which
    # assert "Torch not compiled with CUDA". On XPU, redirect them to the torch.xpu equivalents so MTP runs.
    try:
        if hasattr(torch, "xpu"):
            torch.cuda.synchronize = lambda *a, **k: torch.xpu.synchronize()
            torch.cuda.current_stream = lambda *a, **k: torch.xpu.current_stream(*a, **k)
            torch.cuda.stream = torch.xpu.stream
            torch.cuda.Stream = torch.xpu.Stream
            torch.cuda.Event = torch.xpu.Event
            torch.cuda.empty_cache = torch.xpu.empty_cache
            print("[woq-shim] torch.cuda.{synchronize,Stream,Event,...} -> torch.xpu (for spec-decode)", flush=True)
    except Exception as e:
        print(f"[woq-shim] cuda->xpu redirect failed: {e}", flush=True)

    print("[woq-shim] installed: GPTQLinearScheme -> auto_round_kernel.woqgemm (XPU int4)", flush=True)


_install()
