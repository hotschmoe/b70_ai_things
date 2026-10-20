"""Make opt-in proxy MoE prefill/mixed rows use decode arithmetic.

X1's B70_MTP_CANONICAL_MOE fixes routed arithmetic. Chunk the complete
TP1 apply as well so shared experts use the same fused path above eight
rows. Otherwise a running decode row mixed with a prefill still changes
shared-expert math. This is a correctness control; measure its TTFT cost.
"""
import os
import sys


def install():
    import torch
    from vllm.models.qwen4_exp.gguf import moe

    cls = moe.B70GGUFMoEMethod
    if getattr(cls, '_x5_canonical_moe', False):
        return
    assert os.environ.get('B70_MTP_CANONICAL_MOE') == '1'
    original = cls.apply

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        if x.shape[0] <= 8 or getattr(layer, 'expert_map', None) is not None:
            return original(self, layer, x, topk_weights, topk_ids,
                            shared_experts, shared_experts_input)
        routed, shared = [], []
        for start in range(0, x.shape[0], 8):
            stop = min(start + 8, x.shape[0])
            sx = None if shared_experts_input is None else shared_experts_input[start:stop]
            y = original(self, layer, x[start:stop], topk_weights[start:stop],
                         topk_ids[start:stop], shared_experts, sx)
            assert isinstance(y, torch.Tensor), 'X5 canonical TP1 requires finalized output'
            routed.append(y)
            if shared_experts is not None and moe._fuse_shared_on():
                shared.append(shared_experts._output[shared_experts._output_idx])
        if shared:
            moe._store_shared(shared_experts, torch.cat(shared, dim=0))
        return torch.cat(routed, dim=0)

    cls.apply = apply
    cls._x5_canonical_moe = True
    print('X5_CANONICAL_MOE routed+shared chunk8 installed', file=sys.stderr, flush=True)
