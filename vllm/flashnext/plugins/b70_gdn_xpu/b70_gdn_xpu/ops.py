"""Loader for libb70_gdn_xpu.so.

Decode calls are XPU-graph safe: the schema lives in the SYCL library,
fake kernels cover Dynamo, and the wrappers do not sync or allocate.
"""

from __future__ import annotations

import os

import torch

_LOADED = False
_FAKES = False

DEFAULT_SO = "/mnt/vm_8tb/b70/build/vllm-flashnext/b70_gdn_xpu/libb70_gdn_xpu.so"


def _register_fakes() -> None:
    global _FAKES
    if _FAKES:
        return
    from torch.library import register_fake

    @register_fake("b70_gdn::decode_fused")
    def _decode_fake(
        core,
        out,
        qkvz,
        ba,
        conv_state,
        ssm_state,
        conv_w,
        A_log,
        dt_bias,
        norm_w,
        cu_seqlens,
        slot,
        has_initial,
        token_indx,
        num_accepted,
        eps,
    ):
        del core, out, qkvz, ba, conv_state, ssm_state, conv_w, A_log, dt_bias
        del norm_w, cu_seqlens, slot, has_initial, token_indx, num_accepted, eps
        return None

    @register_fake("b70_gdn::paged_decode")
    def _paged_fake(
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seqused,
        cu_seqlens,
        scale,
        splits,
        partial_o,
        partial_m,
        partial_l,
    ):
        del out, q, k_cache, v_cache, block_table, seqused, cu_seqlens
        del scale, splits, partial_o, partial_m, partial_l
        return None

    _FAKES = True


def register() -> None:
    global _LOADED
    if _LOADED:
        return
    so = os.environ.get("B70_GDN_SO", DEFAULT_SO)
    if not os.path.isfile(so):
        raise FileNotFoundError(so)
    torch.ops.load_library(so)
    _register_fakes()
    _LOADED = True
