"""Scoped upstream XPU batch-invariant dense linears for a GGUF model.

The global vLLM mode rejects quantized models. Reuse only its existing
dense linear implementation here; packed GGUF ops and attention keep
their selected implementations. Needs numeric/full-model graph gates.
"""
import sys


def install():
    import torch
    import triton
    from vllm.model_executor.determinism import batch_invariant as bi
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.qwen4_exp.gguf import linear
    from vllm.utils.platform_utils import num_compute_units

    if getattr(UnquantizedLinearMethod, '_x5_batch_linear', False):
        return
    original = UnquantizedLinearMethod.apply
    old_dense = linear._dense
    ready = False

    def dense(x, weight, bias=None):
        nonlocal ready
        if not ready:
            assert not torch.xpu.is_current_stream_capturing(), 'X5 BI setup needs eager warmup'
            triton.set_allocator(lambda size, alignment, stream: torch.empty(
                size, device=x.device, dtype=torch.int8))
            bi._NUM_SMS = num_compute_units(x.device.index)
            ready = True
        return bi.linear_batch_invariant(x, weight, bias)

    def supported(x, weight):
        return (x.device.type == 'xpu' and weight.ndim == 2
                and x.dtype == weight.dtype
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32))

    def apply(self, layer, x, bias=None):
        if supported(x, layer.weight):
            return dense(x, layer.weight, bias)
        return original(self, layer, x, bias)

    def packed_dense(x, weight):
        return dense(x, weight) if supported(x, weight) else old_dense(x, weight)

    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._x5_batch_linear = True
    linear._dense = packed_dense
    print('X5_BATCH_LINEAR scoped native XPU invariant dense path', file=sys.stderr, flush=True)
