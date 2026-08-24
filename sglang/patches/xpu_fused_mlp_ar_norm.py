"""Small-row XPU push all-reduce + residual + Gemma RMSNorm fusion.

This installs only with B70_XPU_FUSED_MLP_AR_NORM=1 and complements the
delayed dense-MLP route in xpu_delayed_mlp_ar. It advertises fusion only after
the push communicator has initialized the exact fused ABI, and only for BF16
contiguous [M,5120] tensors at the bit-exact measured M=1..8,10,11 decode
shapes. M=9 and all larger shapes retain SGLang's generic all-reduce plus
Gemma RMSNorm path.
"""

import hashlib
import inspect
import os
from functools import wraps


_ENV = "B70_XPU_FUSED_MLP_AR_NORM"
_DELAY_ENV = "B70_XPU_DELAY_MLP_AR"
_MARKER = "_sglang_needs_allreduce_fusion"
_APPLY_SHA256 = "3903deeaf8b701e301d0f425bf023d55e4e1c54890186d3e9e79c039225d2a64"
_FORWARD_SHA256 = "7882232f4d9eb91aba4b033cf51736f39674b7f6356b40787cfb8e8c69246057"
_INSTALLED = False
_CALLS = 0
_FUSED_ROWS = frozenset((*range(1, 9), 10, 11))


def _eligible_tensor(tensor, torch, push_ar_xpu):
    return (
        getattr(tensor, _MARKER, None) is True
        and tensor.device.type == "xpu"
        and tensor.dtype == torch.bfloat16
        and tensor.ndim == 2
        and tensor.shape[0] in _FUSED_ROWS
        and tensor.shape[1] == 5120
        and tensor.is_contiguous()
        and push_ar_xpu.fused_boundary_ready()
    )


def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get(_ENV) != "1":
        return False
    if os.environ.get(_DELAY_ENV) != "1":
        raise RuntimeError(f"{_ENV}=1 requires {_DELAY_ENV}=1")

    import torch

    import push_ar_xpu
    from sglang.srt.layers import communicator as comm_module
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.utils import is_xpu

    if not is_xpu():
        return False

    original_apply = comm_module.apply_aiter_all_reduce_fusion
    original_forward = GemmaRMSNorm.forward_with_allreduce_fusion
    apply_sha = hashlib.sha256(inspect.getsource(original_apply).encode()).hexdigest()
    forward_sha = hashlib.sha256(inspect.getsource(original_forward).encode()).hexdigest()
    if apply_sha != _APPLY_SHA256 or forward_sha != _FORWARD_SHA256:
        raise RuntimeError(
            "C3b fused boundary refusing unknown SGLang sources: "
            f"apply={apply_sha} forward={forward_sha}"
        )

    @wraps(original_apply)
    def apply_fused_boundary(input_tensor):
        if _eligible_tensor(input_tensor, torch, push_ar_xpu):
            return True
        return original_apply(input_tensor)

    @wraps(original_forward)
    def forward_fused_boundary(
        self,
        x,
        residual=None,
        post_residual_addition=None,
        use_attn_tp_group=True,
    ):
        global _CALLS
        if not _eligible_tensor(x, torch, push_ar_xpu):
            return original_forward(
                self,
                x,
                residual,
                post_residual_addition,
                use_attn_tp_group=use_attn_tp_group,
            )

        assert use_attn_tp_group is False
        assert residual is not None and post_residual_addition is None
        assert residual.device == x.device and residual.dtype == torch.bfloat16
        assert residual.shape == x.shape and residual.is_contiguous()
        raw_weight = self.weight.data
        assert raw_weight.device == x.device
        assert raw_weight.dtype == torch.bfloat16
        assert raw_weight.shape == (5120,) and raw_weight.is_contiguous()

        engaged = push_ar_xpu.fused_ar_residual_gemma_rmsnorm_bf16(
            x, residual, raw_weight, self.variance_epsilon
        )
        if not engaged:
            raise RuntimeError("C3b fused boundary readiness changed during dispatch")
        _CALLS += 1
        if _CALLS == 1 or _CALLS % 4096 == 0:
            print(
                f"[c3b-fused] calls={_CALLS} rows={x.shape[0]} hidden={x.shape[1]}",
                flush=True,
            )
        return x, residual

    comm_module.apply_aiter_all_reduce_fusion = apply_fused_boundary
    GemmaRMSNorm.forward_with_allreduce_fusion = forward_fused_boundary
    _INSTALLED = True
    print(
        "[c3b-fused] installed push-AR + residual + Gemma RMSNorm "
        "for BF16 M=1..8/10/11 H=5120 (M=9 exact fallback)",
        flush=True,
    )
    return True
