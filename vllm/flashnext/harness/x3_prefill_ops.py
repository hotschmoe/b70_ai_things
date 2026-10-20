"""X3: one-card prefill budget at real Flash-Next shapes and packed weights.

Invoke inside xpu-run under cpu-run --quiet. Events include host gaps on the
queue. Wall times also include CPU PLE hashes, gathers, and synchronization.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gguf import GGUFReader

torch.set_grad_enabled(False)

from b70_gguf_xpu import prefill, prefill_dq
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

_CONFIG_CONTEXT = set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(mode=0)))
_CONFIG_CONTEXT.__enter__()

ROOT = Path('/mnt/vm_8tb/b70/build/vllm-flashnext/x3')
SHARD = Path('/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS')
ROWS = []
P = argparse.ArgumentParser()
P.add_argument('--tokens', type=int, default=12288)
P.add_argument('--part', default='gdn,attention,hc,ple,dense,expert,qsa')
P.add_argument('--out', default='ops.json')
A = P.parse_args()
T = A.tokens
D = 'xpu'
PARTS = A.part.split(',')


def bench(name, fn, count=1, warm=3, iters=5):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    event, wall = [], []
    for _ in range(iters):
        a, b = (torch.xpu.Event(enable_timing=True) for _ in range(2))
        t = time.perf_counter()
        a.record()
        out = fn()
        b.record()
        b.synchronize()
        event.append(a.elapsed_time(b))
        wall.append((time.perf_counter() - t) * 1000)
    if isinstance(out, torch.Tensor):
        assert torch.isfinite(out).all().item(), name
    row = dict(name=name, T=T, count=count, event_ms=statistics.median(event),
               wall_ms=statistics.median(wall), event_samples=event, wall_samples=wall)
    ROWS.append(row)
    print(json.dumps(row), flush=True)
    (ROOT / A.out).write_text(json.dumps(ROWS, indent=2))
    return out


def cleanup():
    prefill_dq._SCRATCH.clear()
    gc.collect()
    torch.xpu.empty_cache()


print('CONFIG', torch.__version__, 'T', T, 'SO', os.environ.get('B70_GGUF_SO'), flush=True)
prefill.load()
R = GGUFReader(str(SHARD / 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'))
TS = {x.name: x for x in R.tensors}


def raw(name):
    return torch.from_numpy(np.array(TS[name].data, copy=True)).to(D)


def dense(name):
    tt = TS[name]
    arr = torch.from_numpy(np.array(tt.data, copy=True))
    if int(tt.tensor_type) == 30:
        arr = arr.view(torch.bfloat16)
    elif int(tt.tensor_type) not in (0, 1):
        raise RuntimeError('dense called on packed ' + name)
    return arr.reshape(tuple(int(z) for z in reversed(tt.shape))).to(D, torch.bfloat16)


def proj(name, x):
    tt = TS[name]
    n, k = map(int, reversed(tt.shape))
    q = raw(name)
    return bench(name, lambda: prefill.prefill_dense(x, q, int(tt.tensor_type), n, k))


if 'gdn' in PARTS:
    import vllm_xpu_kernels._xpu_C  # registers _xpu_C ops
    for nv, label, count in ((32, '35b', 30), (48, 'flashnext', 36)):
        nk, dh = 16, 128
        ch = (2 * nk + nv) * dh
        qkvz = torch.randn(T, ch + nv * dh, dtype=torch.bfloat16, device=D) * .1
        ba = torch.randn(T, 2 * nv, dtype=torch.bfloat16, device=D) * .1
        conv = torch.zeros(1, 3, ch, dtype=torch.bfloat16, device=D)
        ssm = torch.zeros(1, nv, dh, dh, dtype=torch.float32, device=D)
        cw = torch.randn(ch, 4, device=D, dtype=torch.bfloat16) * .1
        aa = torch.zeros(nv, device=D)
        dt = torch.zeros(nv, device=D, dtype=torch.bfloat16)
        core = torch.empty(T, nv, dh, dtype=torch.bfloat16, device=D)
        z = torch.empty_like(core)
        has = torch.tensor([False], device=D)
        cu = torch.tensor([0, T], dtype=torch.int32, device=D)
        slot = torch.tensor([0], dtype=torch.int32, device=D)
        def gdn():
            torch.ops._xpu_C.gdn_attention(core, z, qkvz, ba, nk, nv, dh, dh,
                conv, ssm, cw, None, 'silu', aa, dt, 1, 0, 0, has, cu,
                None, slot, None, None, None, None, T, 1, True)
            return core
        bench(label + '.gdn_conv_chunk', gdn, count)
        from vllm.model_executor.layers.layernorm import RMSNormGated
        norm = RMSNormGated(dh, eps=1e-6, group_size=None,
                            norm_before_gate=True, device=D, dtype=torch.bfloat16)
        norm.weight.data.fill_(1)
        print('NORM_DISPATCH', label, norm._forward_method.__name__, flush=True)
        bench(label + '.gdn_norm_gate', lambda: norm(core, z), count)
        del qkvz, ba, conv, ssm, cw, aa, dt, core, z, norm
        cleanup()

if 'attention' in PARTS:
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
    for nh, label, count in ((16, '35b', 10), (24, 'flashnext', 12)):
        q = torch.randn(T, nh, 256, dtype=torch.bfloat16, device=D) * .1
        k = torch.randn(T, 2, 256, dtype=torch.bfloat16, device=D) * .1
        v = torch.randn_like(k)
        cu = torch.tensor([0, T], dtype=torch.int32, device=D)
        bench(label + '.full_attention', lambda: flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=T, max_seqlen_k=T, causal=True), count)
        # Q/K head norm, rope, and output gate are measured separately.
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        norm = GemmaRMSNorm(256, eps=1e-6).to(D, torch.bfloat16)
        norm.weight.data.zero_()
        bench(label + '.q_norm', lambda: norm(q), count)
        bench(label + '.k_norm', lambda: norm(k), count)
        from vllm.model_executor.layers.rotary_embedding import get_rope
        rope = get_rope(256, 262144, rope_parameters={
            'rope_theta': 1e7, 'partial_rotary_factor': .25, 'rope_type': 'default'},
            dtype=torch.bfloat16).to(D)
        pos = torch.arange(T, device=D)
        bench(label + '.rope', lambda: rope(pos, q.reshape(T, -1), k.reshape(T, -1)), count)
        gate = torch.randn_like(q)
        bench(label + '.attention_gate', lambda: q * gate.sigmoid(), count)
        del q, k, v, gate
        cleanup()

if 'hc' in PARTS or 'ple' in PARTS:
    from transformers import AutoConfig
    import vllm.models.qwen4_exp.xpu.model as model
    cfg = AutoConfig.from_pretrained('/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext').text_config
    xhc = torch.randn(T, 4, 2560, device=D, dtype=torch.bfloat16) * .1

if 'hc' in PARTS:
    hc = model.HyperConnection(cfg).to(D, torch.bfloat16)
    mapping = {'hc_norm': 'norm', 'input_mix_weight_down': 'down',
               'input_mix_weight_up': 'up', 'block_inject_weight': 'inject'}
    for attr, src in mapping.items():
        val = dense('blk.0.hc_attn_' + src + '.weight')
        if src == 'norm':
            val = val - 1
        getattr(hc, attr).weight.data = val
    mix, inject = bench('flashnext.hc_mix', lambda: hc.mix(xhc), 97)
    bench('flashnext.hc_combine', lambda: hc.combine(xhc, mix, inject), 96)
    del hc, mix, inject
    cleanup()

if 'ple' in PARTS:
    from vllm.models.qwen4_exp.gguf.ple import host_table_from_packed, assign_ple_hash
    rr = GGUFReader(str(SHARD / 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf'))
    table = next(tt for tt in rr.tensors if tt.name == 'per_layer_token_embd.weight')
    ple = model.Qwen4ExpPLE(cfg).to(D, torch.bfloat16)
    ple.host_table = host_table_from_packed(torch.from_numpy(table.data), int(table.tensor_type), 160, owner=rr)
    keys = ('layer_multipliers', 'head_offsets', 'head_vocab_sizes')
    assign_ple_hash(ple.ple_embedding, *(R.fields['qwen4exp.ple.' + key].contents() for key in keys))
    for attr, src in (('key_proj', 'key'), ('value_proj', 'value'), ('conv1d', 'conv1d'),
                      ('norm_key', 'norm_key'), ('norm_query', 'norm_query'), ('norm_conv', 'norm_conv')):
        val = dense('blk.1.ple_' + src + '.weight')
        if attr.startswith('norm_'):
            val = val - 1
        if attr == 'conv1d':
            val = val.unsqueeze(1)
        getattr(ple, attr).weight.data = val
    ids = torch.randint(0, 248000, (T,), device=D)
    cu = torch.tensor([0, T], dtype=torch.int32, device=D)
    slot = torch.tensor([0], dtype=torch.int32, device=D)
    has = torch.tensor([False], device=D)
    metadata = SimpleNamespace(non_spec_query_start_loc=cu, non_spec_state_indices_tensor=slot, has_initial_state=has)
    model.get_forward_context = lambda: SimpleNamespace(attn_metadata={'x3': metadata})
    ple._ensure_state(1, torch.device(D))
    cpu_ids, cpu_cu, cpu_slot, cpu_has = (v.cpu() for v in (ids, cu, slot, has))
    bench('flashnext.ple_hash', lambda: ple._compute_rows(cpu_ids, cpu_cu, cpu_slot, cpu_has), iters=2)
    bench('flashnext.ple_gather', lambda: ple._gather_emb(cpu_ids, cpu_cu, cpu_slot, cpu_has, torch.device(D)), iters=2)
    bench('flashnext.ple_whole', lambda: ple(xhc, ids, 'x3', 1), iters=2)
    os.environ['B70_PREFILL_PLE_HASH'] = 'numpy'
    bench('flashnext.ple_hash_numpy', lambda: ple._compute_rows(cpu_ids, cpu_cu, cpu_slot, cpu_has), iters=2)
    bench('flashnext.ple_gather_numpy', lambda: ple._gather_emb(cpu_ids, cpu_cu, cpu_slot, cpu_has, torch.device(D)), iters=2)
    bench('flashnext.ple_whole_numpy', lambda: ple(xhc, ids, 'x3', 1), iters=2)
    del ple, rr, ids, cu, slot, has, metadata
    cleanup()

if 'dense' in PARTS:
    x = torch.randn(T, 2560, device=D, dtype=torch.bfloat16) * .1
    for name in ('blk.0.attn_qkv.weight', 'blk.0.attn_gate.weight',
                 'blk.3.attn_q.weight', 'blk.3.attn_k.weight', 'blk.3.attn_v.weight',
                 'blk.0.ffn_gate_shexp.weight', 'blk.0.ffn_up_shexp.weight',
                 'blk.0.ffn_gate_inp.weight', 'blk.0.ssm_alpha.weight',
                 'blk.3.indexer.q_proj.weight', 'blk.3.indexer.k_proj.weight'):
        tt = TS[name]
        if int(tt.tensor_type) in (0, 1, 30):
            w = dense(name)
            bench(name, lambda: torch.nn.functional.linear(x, w))
        else:
            proj(name, x)
    y = torch.randn(T, 6144, device=D, dtype=torch.bfloat16)
    proj('blk.0.ssm_out.weight', y)
    proj('blk.3.attn_output.weight', y)
    y = torch.randn(T, 640, device=D, dtype=torch.bfloat16) * .1
    proj('blk.0.ffn_down_shexp.weight', y)
    router = torch.randn(T, 512, device=D)
    bench('flashnext.router_softmax_top10', lambda: router.softmax(-1).topk(10, dim=-1), 48)
    del x, y, router
    cleanup()

if 'expert' in PARTS:
    from b70_gguf_xpu.serve_prefill import moe_prefill
    x = torch.randn(T, 2560, device=D, dtype=torch.bfloat16) * .1
    ids = torch.randint(0, 512, (T, 10), device=D, dtype=torch.int32)
    weights = torch.rand(T, 10, device=D)
    weights /= weights.sum(-1, keepdim=True)
    for li in (0, 3, 36):
        names = [f'blk.{li}.ffn_{p}_exps.weight' for p in ('gate', 'up', 'down')]
        packs = [raw(n) for n in names]
        types = [int(TS[n].tensor_type) for n in names]
        bench(f'flashnext.experts.{li}.types_{types}', lambda: moe_prefill(
            x, ids, *packs, weights, *types, 2560, 640), 48)
        os.environ['B70_PREFILL_REDUCE'] = 'triton'
        bench(f'flashnext.experts.{li}.triton_reduce', lambda: moe_prefill(
            x, ids, *packs, weights, *types, 2560, 640), 48)
        os.environ.pop('B70_PREFILL_REDUCE', None)
        del packs
        cleanup()
    del x, ids, weights
    cleanup()

if 'reduce' in PARTS:
    from b70_gguf_xpu.prefill_reduce import reduce_slots
    slots = torch.randn(T * 10, 2560, device=D, dtype=torch.bfloat16)
    router = torch.rand(T, 10, device=D)
    def reference():
        return (slots.view(T, 10, 2560).float() * router[:, :, None]).sum(1)
    ref = bench('flashnext.reduce_torch', reference, 48)
    got = bench('flashnext.reduce_triton', lambda: reduce_slots(slots, router, T, 10, 2560), 48)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5)
    print('REDUCE_OK', (ref-got).abs().max().item(), flush=True)
    del slots, router, ref, got
    cleanup()

if 'qsa' in PARTS:
    from b70_qsa_xpu import ops as qsops
    from b70_qsa_xpu.batch import QsaBatch
    from b70_qsa_xpu.state import Geom
    qsops.load()
    geom = Geom(n_head=24, n_kv=2, head_dim=256, n_rot=64, idx_n_head=4,
                idx_dim=128, idx_block=4, idx_top_k=2048, rope_base=1e7,
                eps=1e-6, page_size=64)
    batch = QsaBatch(geom, np.arange((T+63)//64, dtype=np.int32)[None], 64, torch.device(D))
    pos = torch.arange(T, device=D, dtype=torch.int32)
    q = torch.randn(T, 24, 256, device=D) * .1
    qi = torch.randn(T, 4, 128, device=D) * .1
    qfull = torch.cat((q, torch.zeros_like(q)), -1)
    def append():
        for t in range(T):
            batch.pos.fill_(t)
            batch.append_cells()
        return batch.length
    bench('flashnext.qsa_append_token_loop', append, 12, warm=0, iters=1)
    bench('flashnext.qsa_prefill_chunk', lambda: batch.prefill_chunk(0, q, qi, qfull, pos), 12, warm=0, iters=1)

print('DONE', ROOT / A.out, flush=True)
