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
torch.manual_seed(303)

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


def bench(name, fn, count=1, warm=3, iters=5, tokens=None):
    torch.xpu.reset_peak_memory_stats()
    base_bytes = torch.xpu.memory_allocated()
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
    row = dict(name=name, T=T if tokens is None else tokens, count=count, event_ms=statistics.median(event),
               wall_ms=statistics.median(wall), event_samples=event, wall_samples=wall,
               base_allocated=base_bytes, peak_allocated=torch.xpu.max_memory_allocated())
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


if 'gdn' in PARTS or 'gdnsplit' in PARTS:
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
        if 'gdnsplit' in PARTS:
            def conv_only():
                return torch.ops._xpu_C.causal_conv1d_non_spec(
                    z,qkvz,ba,nk,nv,dh,dh,conv,cw,None,'silu',1,0,0,
                    has,cu,None,slot,T,1,True)
            pieces = bench(label + '.gdn_conv_only', conv_only, count)
            def delta_only():
                torch.ops._xpu_C.gated_delta_rule_non_spec(
                    core,*pieces,nv,dh,aa,dt,ssm,1,0,0,has,cu,None,slot,T,1)
                return core
            bench(label + '.gdn_delta_only', delta_only, count)
            del pieces
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

if 'tile' in PARTS:
    from b70_gguf_xpu.serve_prefill import moe_prefill
    x = torch.randn(T, 2560, device=D, dtype=torch.bfloat16) * .1
    ids = torch.randint(0,512,(T,10),device=D,dtype=torch.int32)
    weights = torch.rand(T,10,device=D)
    weights /= weights.sum(-1,keepdim=True)
    packs = [raw(f'blk.36.ffn_{p}_exps.weight') for p in ('gate','up','down')]
    for mode,tile in (('triton',0),('grouped',0),('grouped',128),('grouped',64),('grouped',32)):
        cleanup()
        os.environ['B70_PREFILL_REDUCE'] = mode
        os.environ['B70_PREFILL_EXPERT_TILE'] = str(tile)
        bench(f'tile.mode_{mode}.E{tile}',lambda:moe_prefill(
            x,ids,*packs,weights,18,18,42,2560,640),48,warm=2,iters=3)
    os.environ['B70_PREFILL_EXPERT_TILE'] = '0'
    del x,ids,weights,packs
    cleanup()

if 'census' in PARTS:
    import re
    groups = {}
    kinds = ('attn_qkv', 'attn_gate', 'ssm_out', 'ssm_alpha', 'ssm_beta',
             'attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_shexp',
             'ffn_up_shexp', 'ffn_down_shexp', 'ffn_gate_inp', 'ffn_gate_inp_shexp',
             'indexer.q_proj', 'indexer.k_proj')
    for name, tt in TS.items():
        match = re.match(r'blk\.\d+\.(.*)\.weight$', name)
        if not match or match[1] not in kinds:
            continue
        shape = tuple(int(z) for z in reversed(tt.shape))
        if len(shape) == 1:
            shape = (1, shape[0])
        key = (match[1], int(tt.tensor_type), shape)
        groups.setdefault(key, []).append(name)
    for (kind, typ, (n, k)), names in groups.items():
        x = torch.randn(T, k, device=D, dtype=torch.bfloat16) * .1
        if typ in (0, 1, 30):
            w = dense(names[0]).reshape(n, k)
            fn = lambda: torch.nn.functional.linear(x, w)
        else:
            w = raw(names[0])
            fn = lambda: prefill.prefill_dense(x, w, typ, n, k)
        bench(f'census.{kind}.type{typ}.{n}x{k}', fn, len(names))
        del x, w
        cleanup()
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
    x=torch.empty(T,2560,device=D,dtype=torch.bfloat16)
    logits=torch.randn(T,512,device=D)
    rw,ri,_=bench('flashnext.router_fused_top10',lambda:fused_topk(x,logits,10,True),48)
    assert torch.allclose(rw.sum(-1),torch.ones(T,device=D),atol=1e-5,rtol=1e-5)
    del x,logits,rw,ri
    cleanup()
    x = torch.randn(T, 640, device=D, dtype=torch.bfloat16) * .1
    u = torch.randn_like(x)
    bench('flashnext.shared_silu_mul', lambda: torch.nn.functional.silu(x) * u, 48)
    del x, u
    cleanup()
    # Normal serving projects just the sampled position into the vocabulary.
    name = 'output.weight'
    tt = TS[name]
    n, k = map(int, reversed(tt.shape))
    w = raw(name)
    x = torch.randn(1, k, device=D, dtype=torch.bfloat16)
    bench('flashnext.head_one_position', lambda: torch.ops.b70_gguf.gemv(x, w, int(tt.tensor_type), n, k))
    del x, w
    cleanup()
    from vllm.models.qwen4_exp.gguf.ops_interface import gather_rows
    tt = TS['token_embd.weight']
    table = raw('token_embd.weight')
    ids = torch.randint(0, 248320, (T,), device=D)
    bench('flashnext.token_embedding', lambda: gather_rows(table, ids, int(tt.tensor_type), 2560))
    del table, ids
    cleanup()

if 'proxy' in PARTS or 'census' in PARTS:
    from b70_gguf_xpu.serve_prefill import moe_prefill
    rr = GGUFReader('/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf')
    tt = {v.name:v for v in rr.tensors}
    names = [f'blk.0.ffn_{p}_exps.weight' for p in ('gate','up','down')]
    packs = [torch.from_numpy(np.array(tt[n].data,copy=True)).to(D) for n in names]
    types = [int(tt[n].tensor_type) for n in names]
    os.environ['B70_PREFILL_REDUCE'] = 'grouped'
    for n in (1024,8192,12288):
        x = torch.randn(n,2048,device=D,dtype=torch.bfloat16) * .1
        ids = torch.randint(0,256,(n,8),device=D,dtype=torch.int32)
        w = torch.rand(n,8,device=D);w/=w.sum(-1,keepdim=True)
        for how in ('off','0'):
            cleanup()
            os.environ['B70_PREFILL_DQ_T'] = how
            bench(f'35b.experts.T{n}.dq_{how}',lambda:moe_prefill(
                x,ids,*packs,w,*types,2048,512),40,warm=2,iters=3,tokens=n)
        del x,ids,w
        cleanup()
    os.environ.pop('B70_PREFILL_DQ_T')
    del rr,tt,packs
    cleanup()

if 'misc' in PARTS:
    from vllm.models.qwen4_exp.gguf.mapping import ValueHeadLayout
    from b70_gguf_xpu.prefill_layout import restore_value_axis
    for nv,hpg,label,count in ((32,2,'35b',30),(48,3,'flashnext',36)):
        x = torch.randn(T,nv*128,device=D,dtype=torch.bfloat16)
        layout = ValueHeadLayout(hpg,128)
        old = bench(label+'.gdn_value_retile_old',lambda:layout.restore_last_dim(x),count)
        new = bench(label+'.gdn_value_retile_direct',lambda:restore_value_axis(x,layout),count)
        assert torch.equal(old,new)
        print('VALUE_RETILE_EXACT',label,flush=True)
        qkv = torch.randn(T,(32+nv)*128,device=D,dtype=torch.bfloat16)
        bench(label+'.gdn_qkvz_cat',lambda:torch.cat((qkv,x),-1),count)
        ba = torch.randn(T,nv,device=D,dtype=torch.bfloat16)
        bench(label+'.gdn_ba_cat',lambda:torch.cat((ba,ba),-1),count)
        del x,qkv,ba,old,new
        cleanup()
    x = torch.randn(T,640,device=D,dtype=torch.bfloat16)
    u = torch.randn_like(x)
    bench('flashnext.shared_silu_mul',lambda:torch.nn.functional.silu(x)*u,48)
    y = torch.randn(T,2560,device=D,dtype=torch.bfloat16)
    gate = torch.randn(T,1,device=D,dtype=torch.bfloat16)
    bench('flashnext.shared_gate_scale',lambda:y*gate.sigmoid(),48)
    residual = torch.randn(T,4,2560,device=D,dtype=torch.bfloat16)
    bench('flashnext.hc_expand',lambda:y[:,None,:].expand(-1,4,-1).contiguous())
    del x,u,y,gate,residual
    cleanup()
    from transformers import AutoConfig
    from vllm.models.qwen4_exp.xpu.model import HyperConnection
    cfg=AutoConfig.from_pretrained('/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext').text_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    rope=get_rope(256,262144,rope_parameters=cfg.rope_parameters,dtype=torch.bfloat16).to(D)
    positions=torch.arange(T,device=D).expand(3,-1).contiguous()
    query=torch.randn(T,24*256,device=D,dtype=torch.bfloat16)
    key=torch.randn(T,2*256,device=D,dtype=torch.bfloat16)
    bench('flashnext.mrope_text',lambda:rope(positions,query,key),12)
    del rope,positions,query,key
    cleanup()
    hc=HyperConnection(cfg).to(D,torch.bfloat16)
    for attr,suffix in (('hc_norm','norm'),('input_mix_weight_down','down'),
                         ('input_mix_weight_up','up'),('block_inject_weight','inject')):
        name='blk.0.hc_attn_'+suffix+'.weight'
        if suffix=='norm':
            value=(torch.from_numpy(np.array(TS[name].data,copy=True)).float()-1).to(D,torch.bfloat16)
        else:value=dense(name)
        getattr(hc,attr).weight.data=value
    xhc=torch.randn(T,4,2560,device=D,dtype=torch.bfloat16)*.1
    os.environ.pop('B70_PREFILL_HC',None)
    old,oldinj=bench('flashnext.hc_mix_reference',lambda:hc.mix(xhc),97)
    os.environ['B70_PREFILL_HC']='triton'
    new,newinj=bench('flashnext.hc_mix_fused',lambda:hc.mix(xhc),97)
    torch.testing.assert_close(new,old,atol=5e-2,rtol=5e-2)
    torch.testing.assert_close(newinj,oldinj,atol=5e-2,rtol=5e-2)
    cos=torch.nn.functional.cosine_similarity(new.flatten().float(),old.flatten().float(),dim=0)
    assert float(cos)>.9999
    print('HC_MIX_OK','max_abs',float((new-old).abs().max()),'cos',float(cos),flush=True)
    os.environ.pop('B70_PREFILL_HC')
    co=bench('flashnext.hc_combine_reference',lambda:hc.combine(xhc,old,oldinj),96)
    os.environ['B70_PREFILL_HC']='triton'
    cn=bench('flashnext.hc_combine_fused',lambda:hc.combine(xhc,old,oldinj),96)
    torch.testing.assert_close(cn,co,atol=5e-2,rtol=5e-2)
    cos=torch.nn.functional.cosine_similarity(cn.flatten().float(),co.flatten().float(),dim=0)
    assert float(cos)>.9999
    print('HC_COMBINE_OK','max_abs',float((cn-co).abs().max()),'cos',float(cos),flush=True)
    os.environ.pop('B70_PREFILL_HC')
    del xhc,hc,old,oldinj,new,newinj,co,cn
    cleanup()

if 'skew' in PARTS:
    from b70_gguf_xpu.serve_prefill import moe_prefill
    from b70_gguf_xpu.prefill_fast import LAST
    os.environ['B70_PREFILL_REDUCE'] = 'grouped'
    groups = {}
    for li in range(48):
        types = tuple(int(TS[f'blk.{li}.ffn_{p}_exps.weight'].tensor_type) for p in ('gate', 'up', 'down'))
        groups.setdefault(types, []).append(li)
    for types, layers in groups.items():
        li = layers[0]
        x = torch.randn(T, 2560, device=D, dtype=torch.bfloat16)
        packs = [raw(f'blk.{li}.ffn_{p}_exps.weight') for p in ('gate', 'up', 'down')]
        w = dense(f'blk.{li}.ffn_gate_inp.weight')
        logits = torch.nn.functional.linear(x, w).float()
        weights, ids = logits.softmax(-1).topk(10, dim=-1)
        weights /= weights.sum(-1, keepdim=True)
        ids = ids.to(torch.int32)
        bench(f'skew.expert_layer{li}.types{types}', lambda: moe_prefill(
            x, ids, *packs, weights, *types, 2560, 640), len(layers), warm=2, iters=3)
        print('ROUTER_CAP', li, types, LAST, flush=True)
        ROWS[-1]['router_cap'] = dict(LAST)
        (ROOT / A.out).write_text(json.dumps(ROWS, indent=2))
        del x, packs, w, logits, weights, ids
        cleanup()

if 'bucket' in PARTS:
    from b70_gguf_xpu import serve_prefill
    os.environ['B70_PREFILL_REDUCE'] = 'grouped'
    os.environ['B70_PREFILL_DQ_T'] = '0'
    for li in (0, 3, 36):
        types=tuple(int(TS[f'blk.{li}.ffn_{p}_exps.weight'].tensor_type) for p in ('gate','up','down'))
        packs=[raw(f'blk.{li}.ffn_{p}_exps.weight') for p in ('gate','up','down')]
        x=torch.randn(T,2560,device=D,dtype=torch.bfloat16)
        w=dense(f'blk.{li}.ffn_gate_inp.weight')
        weights,ids=torch.nn.functional.linear(x,w).float().softmax(-1).topk(10,dim=-1)
        weights/=weights.sum(-1,keepdim=True)
        ids=ids.to(torch.int32)
        reference=None
        for bucket,tile in (('0','0'),('1','0'),('1','64')):
            cleanup()
            os.environ['B70_PREFILL_BUCKET']=bucket
            os.environ['B70_PREFILL_EXPERT_TILE']=tile
            got=bench(f'bucket.layer{li}.bucket{bucket}.tile{tile}',lambda:serve_prefill.moe_prefill(
                x,ids,*packs,weights,*types,2560,640),warm=1,iters=3)
            if reference is None: reference=got.clone()
            else:
                torch.testing.assert_close(got,reference,atol=5e-2,rtol=5e-2)
                cos=torch.nn.functional.cosine_similarity(got.flatten().float(),reference.flatten().float(),dim=0)
                assert float(cos)>.999
                print('BUCKET_NUMERIC_OK',li,bucket,tile,'max_abs',float((got-reference).abs().max()),'cos',float(cos),flush=True)
            ROWS[-1]['buckets']=dict(serve_prefill.LAST)
            (ROOT/A.out).write_text(json.dumps(ROWS,indent=2))
        if li==36:
            os.environ['B70_PREFILL_DQ_PROFILE']='1'
            serve_prefill.moe_prefill(x,ids,*packs,weights,*types,2560,640)
            ROWS[-1]['diagnostic_stages']=dict(serve_prefill.LAST)
            print('BUCKET_STAGES',json.dumps(serve_prefill.LAST),flush=True)
            (ROOT/A.out).write_text(json.dumps(ROWS,indent=2))
            os.environ.pop('B70_PREFILL_DQ_PROFILE')
        del x,w,weights,ids,packs,got,reference
        cleanup()
    os.environ.pop('B70_PREFILL_BUCKET',None)
    os.environ['B70_PREFILL_EXPERT_TILE']='0'

if 'qsa' in PARTS:
    from b70_qsa_xpu import ops as qsops
    from b70_qsa_xpu.batch import QsaBatch
    from b70_qsa_xpu.state import Geom
    qsops._load()
    geom = Geom(n_head=24, n_kv=2, head_dim=256, n_rot=64, idx_n_head=4,
                idx_dim=128, idx_block=4, idx_top_k=2048, rope_base=1e7,
                eps=1e-6, page_size=64)
    batch = QsaBatch(geom, np.arange((T+63)//64, dtype=np.int32)[None], 64, torch.device(D))
    batch.k_norm.copy_(dense('blk.3.indexer.k_norm.weight').float())
    pos = torch.arange(T, device=D, dtype=torch.int32)
    q = torch.randn(T, 24, 256, device=D) * .1
    qi = torch.randn(T, 4, 128, device=D) * .1
    keys = torch.randn(T,2,256,device=D) * .1
    values = torch.randn_like(keys)
    raw_keys = torch.randn(T,128,device=D) * .1
    qfull = torch.cat((q, torch.zeros_like(q)), -1)
    for t in range(min(T,16)):
        batch.pos.fill_(t)
        batch.append_cells()
    batch.prefill_chunk(0,q[:min(T,16)],qi[:min(T,16)],qfull[:min(T,16)],pos[:min(T,16)])
    torch.xpu.synchronize()
    batch.zero_seq(0)
    torch.xpu.synchronize()
    def append():
        for t in range(T):
            batch.pos.fill_(t)
            batch.k.copy_(keys[t:t+1])
            batch.v.copy_(values[t:t+1])
            batch.raw.copy_(raw_keys[t:t+1])
            batch.append_cells()
            if (t+1) % 2048 == 0:
                print('QSA_APPEND_PROGRESS',t+1,flush=True)
        return batch.length
    bench('flashnext.qsa_append_token_loop', append, 12, warm=0, iters=1)
    bench('flashnext.qsa_prefill_chunk', lambda: batch.prefill_chunk(0, q, qi, qfull, pos), 12, warm=0, iters=1)
    # The serving wrapper also normalizes/rotates unused queries during
    # append and round-trips side state each token. Time that exact path.
    from b70_qsa_xpu.model_hook import QsaRunner
    runner=QsaRunner.__new__(QsaRunner)
    runner.batch=batch
    runner.indexer=SimpleNamespace(index_kv_heads=1)
    runner._q_gamma=dense('blk.3.indexer.q_norm.weight').float()
    runner._row_sink=torch.ones(1,device=D,dtype=torch.int32)
    runner._index=torch.zeros(1,device=D,dtype=torch.int32)
    runner.side_tail=torch.zeros(2,3,128,device=D)
    runner.side_dead=torch.zeros(2,128,device=D)
    runner.side_block_pos=torch.zeros(2,device=D,dtype=torch.int32)
    runner.side_length=torch.zeros(2,device=D,dtype=torch.int32)
    batch.mrope=torch.zeros(1,T,3,device=D,dtype=torch.int32)
    positions=pos.long().expand(3,-1).contiguous()
    qb,kb,vb,ib,rb=[a.to(torch.bfloat16) for a in (q,keys,values,qi,raw_keys)]
    gates=torch.zeros_like(qb)
    pos_cpu=list(range(T))
    batch.zero_seq(0)
    torch.xpu.synchronize()
    def actual_append():
        for start in range(0,T,1024):
            runner._append_request(0,start,min(start+1024,T),qb,kb,vb,gates,ib,rb,positions,pos_cpu,1)
            print('QSA_MODEL_APPEND_PROGRESS',min(start+1024,T),flush=True)
        return batch.length
    bench('flashnext.qsa_model_hook_append',actual_append,12,warm=0,iters=1)
    before=tuple(a.clone() for a in (batch.pages,batch.tail,batch.dead,batch.block_pos,batch.length))
    iq_reference=bench('flashnext.qsa_model_indexer_norm_mrope',lambda:runner._indexer_queries(ib,positions,0,T),12,warm=0,iters=1)
    batch.zero_seq(0)
    for a in (runner.side_tail,runner.side_dead,runner.side_block_pos,runner.side_length):a.zero_()
    os.environ['B70_PREFILL_QSA']='1'
    bench('flashnext.qsa_model_hook_append_fast',actual_append,12,warm=0,iters=1)
    for old,new in zip(before,(batch.pages,batch.tail,batch.dead,batch.block_pos,batch.length)):
        torch.testing.assert_close(new,old,atol=0,rtol=0)
    iq_fast=bench('flashnext.qsa_model_indexer_norm_mrope_fast',lambda:runner._indexer_queries(ib,positions,0,T),12,warm=1,iters=3)
    torch.testing.assert_close(iq_fast,iq_reference,atol=1e-6,rtol=1e-6)
    print('QSA_PREFILL_NUMERIC_OK','index_max_abs',float((iq_fast-iq_reference).abs().max()),flush=True)
    os.environ.pop('B70_PREFILL_QSA')

# QSA is also the last short window before the serving candidate. Recheck
# mixed expert types with the bounded bucket dequant arena in that window.
if 'check' in PARTS or 'qsa' in PARTS:
    from b70_gguf_xpu.serve_prefill import moe_prefill
    os.environ['B70_PREFILL_DQ_T'] = '0'
    os.environ['B70_PREFILL_REDUCE'] = 'triton'
    def small_pack(layer, which):
        tt = TS[f'blk.{layer}.ffn_{which}_exps.weight']
        return torch.from_numpy(np.array(tt.data[:4], copy=True)).to(D), int(tt.tensor_type)
    x = torch.randn(64, 2560, device=D, dtype=torch.bfloat16)
    ids = (torch.arange(128, device=D, dtype=torch.int32) % 4).view(64, 2)
    ids[0, 0] = -1
    router = torch.rand(64, 2, device=D)
    for up_layer in (0, 3):
        ids.copy_((torch.arange(128, device=D, dtype=torch.int32) % 4).view(64,2))
        ids[0,0] = -1
        g, gt = small_pack(0, 'gate')
        u, ut = small_pack(up_layer, 'up')
        d, dt = small_pack(0, 'down')
        wg = prefill_dq.dequant(g, gt, 4*640, 2560).view(4,640,2560).clone()
        wu = prefill_dq.dequant(u, ut, 4*640, 2560).view(4,640,2560).clone()
        wd = prefill_dq.dequant(d, dt, 4*2560, 640).view(4,2560,640).clone()
        def ref():
            slots = torch.zeros(64,2,2560,device=D,dtype=torch.bfloat16)
            for ti in range(64):
                for si in range(2):
                    ei = int(ids[ti,si])
                    if ei < 0:
                        continue
                    gu = torch.nn.functional.linear(x[ti:ti+1], wg[ei])
                    uu = torch.nn.functional.linear(x[ti:ti+1], wu[ei])
                    hidden = torch.nn.functional.silu(gu) * uu
                    slots[ti,si] = torch.nn.functional.linear(hidden,wd[ei])[0]
            return (slots.float()*router[:,:,None]).sum(1)
        expected = ref()
        got = moe_prefill(x,ids,g,u,d,router,gt,ut,dt,2560,640)
        torch.testing.assert_close(got, expected, atol=5e-2, rtol=5e-2)
        print('MOE_REFERENCE_OK',gt,ut,dt,'max_abs',float((got-expected).abs().max()),flush=True)
        original_fast = got.clone()
        for reduce_mode, tile in (('grouped','0'), ('grouped','2')):
            os.environ['B70_PREFILL_REDUCE'] = reduce_mode
            os.environ['B70_PREFILL_EXPERT_TILE'] = tile
            got = moe_prefill(x,ids,g,u,d,router,gt,ut,dt,2560,640)
            torch.testing.assert_close(got, expected, atol=5e-2, rtol=5e-2)
            if tile == '0':
                torch.testing.assert_close(got, original_fast, atol=1e-7, rtol=1e-5)
            else:
                cos = torch.nn.functional.cosine_similarity(got.flatten().float(), original_fast.flatten().float(),dim=0)
                assert float(cos) > .999
                print('TILE_GEOMETRY_DIFF',gt,ut,'max_abs',float((got-original_fast).abs().max()),'cos',float(cos),flush=True)
            print('MOE_GROUPED_REFERENCE_OK',gt,ut,dt,'tile',tile,flush=True)
        os.environ['B70_PREFILL_EXPERT_TILE'] = '0'
        # Identical routing reproduces the pathological profile geometry.
        ids.fill_(1)
        expected = ref()
        got = moe_prefill(x,ids,g,u,d,router,gt,ut,dt,2560,640)
        torch.testing.assert_close(got, expected, atol=5e-2, rtol=5e-2)
        print('MOE_SKEW_REFERENCE_OK',gt,ut,dt,flush=True)
        for bucket_ids in (ids, (torch.arange(128,device=D,dtype=torch.int32)%4).reshape(64,2)):
            bucket_ids[0,0]=-1
            ids.copy_(bucket_ids)
            expected=ref()
            os.environ['B70_PREFILL_BUCKET']='1'
            os.environ['B70_PREFILL_EXPERT_TILE']='2'
            got=moe_prefill(x,ids,g,u,d,router,gt,ut,dt,2560,640)
            torch.testing.assert_close(got,expected,atol=5e-2,rtol=5e-2)
            print('MOE_BUCKET_REFERENCE_OK',gt,ut,dt,flush=True)
            os.environ.pop('B70_PREFILL_BUCKET')
        os.environ['B70_PREFILL_EXPERT_TILE']='0'
        del g,u,d,wg,wu,wd,expected,got,original_fast
        cleanup()
    os.environ.pop('B70_PREFILL_DQ_T')
    del x,ids,router
    cleanup()


print('DONE', ROOT / A.out, flush=True)
