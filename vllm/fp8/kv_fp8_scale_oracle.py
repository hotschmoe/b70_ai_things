#!/usr/bin/env python3
"""Prove scaled E4M3 cache write/read on the installed XPU attention kernel."""
import json
import torch
from vllm import _custom_ops as ops
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


for card in range(torch.xpu.device_count()):
    torch.xpu.set_device(card)
    device = f'xpu:{card}'
    torch.manual_seed(7103)
    length, heads, kvheads, dim = 31, 12, 2, 256
    q = (torch.randn(1, heads, dim) * .2).half().to(device)
    k = (torch.randn(length, kvheads, dim) * 8).half().to(device)
    v = (torch.randn(length, kvheads, dim) * 5).half().to(device)
    kc = torch.zeros((1, 64, kvheads, dim), dtype=torch.uint8, device=device)
    vc = torch.zeros_like(kc)
    ks = torch.tensor(.08, device=device); vs = torch.tensor(.06, device=device)
    ops.reshape_and_cache_flash(k, v, kc, vc, torch.arange(length, device=device), 'fp8_e4m3', ks, vs)
    torch.xpu.synchronize()
    dk = kc.view(torch.float8_e4m3fn)[0, :length].float().cpu() * ks.cpu()
    dv = vc.view(torch.float8_e4m3fn)[0, :length].float().cpu() * vs.cpu()
    write_k_error = float((dk - k.float().cpu()).norm() / k.float().cpu().norm())
    write_v_error = float((dv - v.float().cpu()).norm() / v.float().cpu().norm())
    assert write_k_error < .04 and write_v_error < .04
    def attend(v_multiplier=1., k_multiplier=1.):
        out = torch.empty_like(q)
        flash_attn_varlen_func(q=q, k=kc.view(torch.float8_e4m3fn), v=vc.view(torch.float8_e4m3fn), out=out,
                              cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
                              max_seqlen_q=1, max_seqlen_k=length,
                              seqused_k=torch.tensor([length], dtype=torch.int32, device=device),
                              block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
                              softmax_scale=dim**-.5, causal=True,
                              k_descale=(ks*k_multiplier).expand(1, kvheads), v_descale=(vs*v_multiplier).expand(1, kvheads))
        torch.xpu.synchronize()
        return out.float().cpu()
    actual = attend()
    keys = dk.repeat_interleave(heads//kvheads, dim=1).permute(1, 0, 2)
    values = dv.repeat_interleave(heads//kvheads, dim=1).permute(1, 0, 2)
    queries = q.float().cpu().permute(1, 0, 2)
    reference = (torch.softmax(queries @ keys.transpose(-1, -2) * dim**-.5, dim=-1) @ values).permute(1, 0, 2)
    attention_error = float((actual-reference).norm()/reference.norm())
    assert attention_error < .015, attention_error
    doubled = attend(v_multiplier=2.)
    assert float((doubled-actual*2).norm()/doubled.norm()) < .015
    wrong_k = attend(k_multiplier=2.)
    assert float((wrong_k-actual).norm()/actual.norm()) > .1
    print(json.dumps({'card':card, 'write_k_relative_l2':write_k_error, 'write_v_relative_l2':write_v_error,
                      'attention_relative_l2':attention_error, 'k_and_v_read_scales_consumed':True, 'q_dtype':str(q.dtype)}), flush=True)
