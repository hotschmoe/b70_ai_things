#!/usr/bin/env python3
"""Prove scaled E4M3 cache write/read on the installed XPU attention kernel."""
import json
import argparse
import torch
from vllm import _custom_ops as ops
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


p = argparse.ArgumentParser()
p.add_argument('--length', type=int, default=31)
p.add_argument('--block-size', type=int, default=64)
p.add_argument('--permute', action='store_true')
p.add_argument('--hybrid-layout', action='store_true')
p.add_argument('--query-length', type=int, default=1)
args = p.parse_args()
assert args.length > 0 and args.block_size > 0
assert 1 <= args.query_length <= args.length

for card in range(torch.xpu.device_count()):
    torch.xpu.set_device(card)
    device = f'xpu:{card}'
    torch.manual_seed(7103)
    length, heads, kvheads, dim = args.length, 12, 2, 256
    q = (torch.randn(args.query_length, heads, dim) * .2).half().to(device)
    k = (torch.randn(length, kvheads, dim) * 8).half().to(device)
    v = (torch.randn(length, kvheads, dim) * 5).half().to(device)
    blocks = (length + args.block_size - 1) // args.block_size
    table = list(range(blocks)) if not args.permute else list(range(blocks + 2))[::-1][:blocks]
    if args.hybrid_layout:
        joint = torch.zeros((max(table) + 1, 2, args.block_size, kvheads, dim), dtype=torch.uint8, device=device)
        kc, vc = joint[:, 0], joint[:, 1]
    else:
        kc = torch.zeros((max(table) + 1, args.block_size, kvheads, dim), dtype=torch.uint8, device=device)
        vc = torch.zeros_like(kc)
    ks = torch.tensor(.08, device=device); vs = torch.tensor(.06, device=device)
    positions = torch.arange(length, device=device)
    slots = torch.tensor(table, device=device)[positions // args.block_size] * args.block_size + positions % args.block_size
    ops.reshape_and_cache_flash(k, v, kc, vc, slots, 'fp8_e4m3', ks, vs)
    torch.xpu.synchronize()
    dk = kc.view(torch.float8_e4m3fn).float().cpu()[table].reshape(-1, kvheads, dim)[:length] * ks.cpu()
    dv = vc.view(torch.float8_e4m3fn).float().cpu()[table].reshape(-1, kvheads, dim)[:length] * vs.cpu()
    write_k_error = float((dk - k.float().cpu()).norm() / k.float().cpu().norm())
    write_v_error = float((dv - v.float().cpu()).norm() / v.float().cpu().norm())
    assert write_k_error < .04 and write_v_error < .04
    def attend(v_multiplier=1., k_multiplier=1.):
        out = torch.empty_like(q)
        flash_attn_varlen_func(q=q, k=kc.view(torch.float8_e4m3fn), v=vc.view(torch.float8_e4m3fn), out=out,
                              cu_seqlens_q=torch.tensor([0, args.query_length], dtype=torch.int32, device=device),
                              max_seqlen_q=args.query_length, max_seqlen_k=length,
                              seqused_k=torch.tensor([length], dtype=torch.int32, device=device),
                              block_table=torch.tensor([table], dtype=torch.int32, device=device),
                              softmax_scale=dim**-.5, causal=True,
                              k_descale=(ks*k_multiplier).expand(1, kvheads), v_descale=(vs*v_multiplier).expand(1, kvheads))
        torch.xpu.synchronize()
        return out.float().cpu()
    actual = attend()
    keys = dk.repeat_interleave(heads//kvheads, dim=1).permute(1, 0, 2)
    values = dv.repeat_interleave(heads//kvheads, dim=1).permute(1, 0, 2)
    queries = q.float().cpu().permute(1, 0, 2)
    logits = queries @ keys.transpose(-1, -2) * dim**-.5
    causal_mask = torch.arange(length)[None, :] > torch.arange(args.query_length)[:, None] + length - args.query_length
    logits.masked_fill_(causal_mask[None], float('-inf'))
    reference = (torch.softmax(logits, dim=-1) @ values).permute(1, 0, 2)
    attention_error = float((actual-reference).norm()/reference.norm())
    print(json.dumps({'card':card,'hybrid_layout':args.hybrid_layout,'cache_stride':kc.stride(),
                      'length':length,'query_length':args.query_length,'attention_relative_l2':attention_error}),flush=True)
    assert attention_error < .015, attention_error
    doubled = attend(v_multiplier=2.)
    assert float((doubled-actual*2).norm()/doubled.norm()) < .015
    wrong_k = attend(k_multiplier=2.)
    assert float((wrong_k-actual).norm()/actual.norm()) > .1
    print(json.dumps({'card':card, 'length': length, 'block_size': args.block_size, 'block_table': table,
                      'write_k_relative_l2':write_k_error, 'write_v_relative_l2':write_v_error,
                      'attention_relative_l2':attention_error, 'k_and_v_read_scales_consumed':True, 'q_dtype':str(q.dtype)}), flush=True)
