"""Use the aligned DP4A Q8 kernel, preserving original GGUF codes/scales."""
import sys


def install():
    import torch
    from vllm.models.qwen4_exp.gguf import linear
    import b70_x5_xpu

    if getattr(linear, '_x5_q8_fast', False):
        return
    b70_x5_xpu.load()
    original_process = linear.B70GGUFLinearMethod.process_weights_after_loading
    original_apply = linear._apply_one

    def process(self, layer):
        # Install BEFORE pack_linear so this sees its merged shard.
        for shard in layer.b70_shards.values():
            if shard.qtype != 8:
                continue
            n, k = shard.logical_shape
            raw = shard.packed.reshape(n, k // 32, 34)
            codes = raw[:, :, 2:].contiguous().view(torch.int8).reshape(n, k)
            scales = raw[:, :, :2].contiguous().view(torch.float16).reshape(n, k // 32)
            device = layer.weight.device
            shard._x5_codes = codes.to(device)
            shard._x5_scales = scales.to(device)
        return original_process(self, layer)

    def apply(shard, x):
        if shard.qtype != 8 or x.device.type != 'xpu':
            return original_apply(shard, x)
        assert hasattr(shard, '_x5_codes'), 'X5 Q8 needs load-time relayout'
        k = x.shape[-1]
        rows = x.reshape(-1, k).to(torch.bfloat16).contiguous()
        outputs = [torch.ops.b70_x5.q8_linear(rows[i:i+8], shard._x5_codes, shard._x5_scales)
                   for i in range(0, rows.shape[0], 8)]
        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        return out.reshape(*x.shape[:-1], shard.out_features).to(x.dtype)

    linear.B70GGUFLinearMethod.process_weights_after_loading = process
    linear._apply_one = apply
    linear._x5_q8_fast = True
    print('X5_Q8_FAST aligned payload + DP4A + M-row reuse installed', file=sys.stderr, flush=True)
