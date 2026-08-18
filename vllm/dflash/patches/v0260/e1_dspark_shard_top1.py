#!/usr/bin/env python3
# D6 hook: DSpark greedy sequential sample uses sharded logits +
# xpu_shard_top1, then a tiny (val, idx) all-gather. Markov bias stays.
# Probabilistic draft_logits path is unchanged (bench_code is greedy).
# Loaded as PUSH_AR_CHAIN_SITECUSTOMIZE; execs the original MTP shim first.
from __future__ import annotations

import os
import sys

_chain = "/opt/mtp_shim/sitecustomize.py"
if os.path.exists(_chain):
    import importlib.util

    _spec = importlib.util.spec_from_file_location("_mtp_shim", _chain)
    _m = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_m)
    print("[e1-dspark] chained", _chain, file=sys.stderr, flush=True)

try:
    import torch
    from torch.library import register_fake
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
except Exception as e:
    print("[e1-dspark] import failed:", repr(e), file=sys.stderr, flush=True)
    raise

if hasattr(torch.ops._xpu_C, "xpu_shard_top1"):

    def _fake_xpu_shard_top1(logits):
        out = logits.shape[:-1]
        return (
            logits.new_empty(out, dtype=torch.float32),
            logits.new_empty(out, dtype=torch.int64),
        )

    try:
        register_fake("_xpu_C::xpu_shard_top1", _fake_xpu_shard_top1)
        print("[e1-dspark] registered fake xpu_shard_top1", file=sys.stderr, flush=True)
    except (RuntimeError, ValueError) as e:
        print("[e1-dspark] fake skipped:", e, file=sys.stderr, flush=True)
else:
    print("[e1-dspark] xpu_shard_top1 MISSING", file=sys.stderr, flush=True)

_orig = DSparkSpeculator._sample_sequential


def _sample_sequential_shard(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
    if self.draft_logits is not None or not hasattr(torch.ops._xpu_C, "xpu_shard_top1"):
        return _orig(self, num_reqs, head_hidden)

    n_spec = self.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[self.sample_indices[:num_sample]]
    lp = self.model.logits_processor
    lm_head = self.model.lm_head
    markov_w2 = self.model.model.markov_head.markov_w2

    base_local = lp._apply_head(lm_head, sample_hidden, None)
    if lp.soft_cap is not None:
        base_local = torch.tanh(base_local / lp.soft_cap) * lp.soft_cap
    if lp.scale != 1.0:
        base_local = base_local * lp.scale
    num_pad = lm_head.shard_indices.num_org_vocab_padding
    if num_pad > 0:
        base_local[..., -num_pad:] = float("-inf")
    vocab_start = int(lm_head.shard_indices.org_vocab_start_index)
    shard_v = base_local.shape[-1]
    base_local = base_local.view(num_reqs, n_spec, shard_v)

    prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
    tp_size = get_tensor_model_parallel_world_size()

    for i in range(n_spec):
        markov_embed = self.model.markov_embed(prev)
        bias_local = lp._apply_head(markov_w2, markov_embed, None)
        if lp.scale != 1.0:
            bias_local = bias_local * lp.scale
        logits_i = base_local[:, i] + bias_local
        local_vals, local_idx = torch.ops._xpu_C.xpu_shard_top1(logits_i)
        global_idx = local_idx + vocab_start
        if tp_size == 1:
            draft_ids = global_idx
        else:
            local_pair = torch.stack(
                [local_vals.float(), global_idx.float()], dim=-1
            )
            gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
            gathered = gathered.view(num_reqs, tp_size, 2)
            max_rank = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
            draft_ids = (
                gathered[:, :, 1].gather(dim=-1, index=max_rank).squeeze(-1).to(torch.int64)
            )
        sampled = self.model.map_draft_to_target(draft_ids)
        self.draft_tokens[:num_reqs, i] = sampled
        prev = sampled


DSparkSpeculator._sample_sequential = _sample_sequential_shard
print("[e1-dspark] ENGAGED DSparkSpeculator._sample_sequential shard-top1", file=sys.stderr, flush=True)
