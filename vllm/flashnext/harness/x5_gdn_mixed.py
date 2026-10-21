"""Keep fused decode arithmetic when another request is prefilling (TP1).

Opt-in, SYCL prefill only. FULL_DECODE_ONLY capture stays on the existing
pure-decode path. Mixed eager batches split metadata and rows, without
copying or rolling back recurrent state. Different requests own disjoint
state slots. This module is not enabled until its model gates pass.
"""
import copy
import os
import sys


def _metadata(md, **updates):
    out = copy.copy(md)
    for key, value in updates.items():
        setattr(out, key, value)
    return out


def split(md, torch, device, *, ragged=False):
    """Return (prefill rows/metadata, decode rows/metadata), or None."""
    np, nd, ns = md.num_prefills, md.num_decodes, md.num_spec_decodes
    if not np or not (nd or ns):
        return None
    assert not (nd and ns), 'GDN metadata reclassifies this combination'
    common = dict(spec_sequence_masks=None, spec_sequence_masks_cpu=None,
                  checkpoint=None, flashinfer_checkpoint_plan=None)
    pf = _metadata(md, **common, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=0, num_spec_decode_tokens=0,
        num_actual_tokens=md.num_prefill_tokens,
        non_spec_query_start_loc=md.prefill_query_start_loc,
        non_spec_state_indices_tensor=md.prefill_state_indices,
        has_initial_state=md.prefill_has_initial_state,
        non_spec_token_indx=None, spec_query_start_loc=None,
        spec_state_indices_tensor=None, spec_token_indx=None,
        num_accepted_tokens=None, uniform_spec_sequence_length=None)
    if pf.non_spec_query_start_loc is None or pf.non_spec_state_indices_tensor is None:
        return None
    if ns:
        width = md.spec_state_indices_tensor.shape[1]
        # D3's kernel requires uniform verification. Keep its original safe
        # fallback for ragged batches; never index past a short token array.
        if md.num_spec_decode_tokens != ns * width and not ragged:
            return None
        pr = md.non_spec_token_indx.to(dtype=torch.long)
        dr = md.spec_token_indx.to(dtype=torch.long)
        dm = _metadata(md, **common, num_prefills=0, num_prefill_tokens=0,
            num_actual_tokens=md.num_spec_decode_tokens, has_initial_state=None,
            non_spec_query_start_loc=None, non_spec_token_indx=None,
            non_spec_state_indices_tensor=None,
            spec_token_indx=torch.arange(dr.numel(), dtype=torch.int32, device=device))
        # The builder relabels known one-token decodes as prefills when a
        # speculative request is present. They still need decode arithmetic.
        # This tiny host read occurs only in mixed eager batches, once per
        # forward context; it never enters the steady decode graph.
        if (md.num_prefill_tokens == np and pf.has_initial_state is not None
                and bool(torch.all(pf.has_initial_state[:np]).item())):
            pf = _metadata(pf, num_prefills=0, num_prefill_tokens=0,
                num_decodes=np, num_decode_tokens=np,
                non_spec_query_start_loc=torch.arange(np + 1, dtype=torch.int32, device=device),
                non_spec_state_indices_tensor=pf.non_spec_state_indices_tensor[:np].contiguous())
    else:
        dr = torch.arange(nd, dtype=torch.long, device=device)
        pr = torch.arange(nd, nd + md.num_prefill_tokens, dtype=torch.long, device=device)
        dm = _metadata(md, **common, num_prefills=0, num_prefill_tokens=0,
            num_actual_tokens=nd, has_initial_state=None,
            non_spec_query_start_loc=torch.arange(nd + 1, dtype=torch.int32, device=device),
            non_spec_state_indices_tensor=md.non_spec_state_indices_tensor[:nd].contiguous(),
            non_spec_token_indx=None)
    assert pr.numel() == md.num_prefill_tokens
    assert pr.numel() + dr.numel() == md.num_actual_tokens
    return pr, pf, dr, dm


def install(fused_module):
    import torch
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as cls

    if getattr(cls, '_x5_gdn_mixed', False):
        return
    pure = cls.forward_xpu
    ragged = os.environ.get('B70_X5_GDN_RAGGED') == '1'
    seen = set()

    def forward(self, hidden_states):
        ctx = get_forward_context()
        raw = ctx.attn_metadata
        md = raw.get(self.prefix) if isinstance(raw, dict) else None
        if (md is None or self.tp_size != 1 or self.gqa_interleaved_layout
                or self.gdn_xpu_backend != 'sycl' or not md.num_prefills
                or not (md.num_decodes or md.num_spec_decodes)):
            return pure(self, hidden_states)
        assert not torch.xpu.is_current_stream_capturing(), 'mixed GDN is eager only'
        cache = getattr(ctx, '_x5_gdn_mixed_plans', None)
        if cache is None:
            cache = ctx._x5_gdn_mixed_plans = {}
        key = id(md)
        if key not in cache:
            cache[key] = split(md, torch, hidden_states.device, ragged=ragged)
        plan = cache[key]
        if plan is None:
            return pure(self, hidden_states)
        pr, pf, dr, dm = plan
        signature = (md.num_prefills, md.num_decodes, md.num_spec_decodes)
        if signature not in seen:
            seen.add(signature)
            print('X5_GDN_MIXED_SPLIT', signature, file=sys.stderr, flush=True)
        result = torch.zeros_like(hidden_states)
        # Never catch and replay a partially completed state update. Restore
        # metadata even when an operation raises, and propagate that error.
        try:
            raw[self.prefix] = pf
            yp = pure(self, hidden_states.index_select(0, pr))
            raw[self.prefix] = dm
            yd = pure(self, hidden_states.index_select(0, dr))
            result.index_copy_(0, pr, yp)
            result.index_copy_(0, dr, yd)
        finally:
            raw[self.prefix] = md
        return result

    cls.forward_xpu = forward
    cls._x5_gdn_mixed = True
    print('X5_GDN_MIXED installed, pure graph path unchanged', file=sys.stderr, flush=True)
