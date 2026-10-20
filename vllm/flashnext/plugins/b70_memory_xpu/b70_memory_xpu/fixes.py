"""Alias dense GGUF shards and reuse ordered-stream dequant storage.

No quantization changes. The shared dequant buffer is overwritten only after
the prior GEMM reading it has been submitted on the same stream. Callers must
consume a dequant view before requesting the next one (the existing API).
"""
import math


def alias_dense_shards(layer):
    import torch
    from torch import nn

    shards = getattr(layer, "b70_shards", {})
    if not shards or any(s.packed is not None for s in shards.values()):
        return
    if len(shards) == 1 and "full" in shards:
        shard = shards["full"]
        if shard.dense is not None:
            # Full dense loads had copied into weight but kept the source
            # device tensor alive as well. Both forward paths now share it.
            layer.weight = nn.Parameter(shard.dense, requires_grad=False)
    elif layer.weight.numel():
        # Merged b/a and indexer q/k call F.linear on each shard; their
        # init placeholder is never read after loading.
        layer.weight = nn.Parameter(
            torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device),
            requires_grad=False,
        )


def install():
    import torch
    from vllm.models.qwen4_exp.gguf.linear import B70GGUFLinearMethod
    from b70_gguf_xpu import prefill_dq

    original = B70GGUFLinearMethod.process_weights_after_loading

    def process(self, layer):
        original(self, layer)
        alias_dense_shards(layer)

    B70GGUFLinearMethod.process_weights_after_loading = process
    old_scratch = prefill_dq._scratch

    def scratch(key, shape, dtype, device):
        if key[0] != "dq":
            return old_scratch(key, shape, dtype, device)
        # Shape/quant type do not change the lifetime. Distinct streams need
        # distinct storage. No new stream or synchronization is introduced.
        stream = torch.xpu.current_stream(device).sycl_queue if device.type == "xpu" else None
        shared_key = ("dq_shared", dtype, device.type, device.index, stream)
        size = math.prod(shape)
        buf = prefill_dq._SCRATCH.get(shared_key)
        if buf is None or buf.numel() < size:
            buf = torch.empty(size, dtype=dtype, device=device)
            prefill_dq._SCRATCH[shared_key] = buf
        return buf[:size].view(shape)

    prefill_dq._scratch = scratch
    print("b70_memory: dense aliases and per-stream dequant arena enabled", flush=True)
