#!/usr/bin/env python3
"""Actual Mamba align postprocess byte oracle; plan-only by default."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

WORKER_SHA = '936d65f7e9e85e0c67ada8210ce26108cf7868c4f991f76d0ad3bc125c07dc3a'
MODEL_SHA = 'fb15e26fcfe4f7ab72a61a64fee8b8ea0c49b7f28a23b6a9dede099e41489231'
BLOCK_IDS = [3, 5, 1, 6, 2, 7, 0, 4]


def cases():
    out = [dict(name='self_bias3', computed=4796, scheduled=4, draft=3, accepted=4, src=2, dst=2, bias=3, accepted_out=1, packed=False)]
    out += [dict(name='backward_accept'+str(a), computed=4798, scheduled=4,
                 draft=3, accepted=a, src=3, dst=2 if a>=2 else None,
                 bias=1 if a>=2 else None, accepted_out=a, packed=False) for a in [1,2,3,4]]
    out += [dict(name='preboundary_accept'+str(a), computed=4796, scheduled=4,
                 draft=3, accepted=a, src=2, dst=None, bias=None,
                 accepted_out=a, packed=False) for a in [1,2,3]]
    out += [dict(name='self_bias'+str(a-1), computed=4800-a, scheduled=a,
                 draft=a-1, accepted=a, src=2, dst=2, bias=a-1,
                 accepted_out=1, packed=False) for a in [2,3]]
    out += [dict(out[0],name='packed_self_bias3',packed=True),
            dict(out[4],name='packed_backward_accept4',packed=True)]
    return out


def geometry(case,tp,kv):
    conv_dim=10240//tp;vheads=48//tp
    conv=6*conv_dim*2;temporal=vheads*128*128*4
    page=(832*4096 if kv=='fp16' else 1600*2048)//tp
    return dict(conv_dim=conv_dim,vheads=vheads,conv_bytes=conv,
                temporal_bytes=temporal,page_bytes=conv+temporal if case['packed'] else page)


def regions(case,g):
    if case['dst'] is None or (case['src']==case['dst'] and case['bias']==0):return []
    stride=g['page_bytes'];bias=case['bias']
    src=BLOCK_IDS[case['src']];dst=BLOCK_IDS[case['dst']]
    ssm_src=BLOCK_IDS[case['src']+bias]
    return [(src*stride+bias*g['conv_dim']*2,dst*stride,(6-bias)*g['conv_dim']*2),
            (ssm_src*stride+g['conv_bytes'],dst*stride+g['conv_bytes'],g['temporal_bytes'])]


def run(args):
    assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True", "production allocator required"
    import torch
    from vllm.v1.worker import mamba_utils as worker
    from vllm.model_executor.layers.mamba import mamba_utils as model
    torch.set_num_threads(2)
    assert hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest()==WORKER_SHA
    assert hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()==MODEL_SHA
    assert not model.is_conv_state_dim_first()
    assert Path(inspect.getsourcefile(worker.postprocess_mamba_fused_kernel.fn)).resolve()==Path(worker.__file__).resolve()
    results=[]
    for idx,case in enumerate(cases()):
        g=geometry(case,args.tp_size,args.kv_layout);page=g['page_bytes']
        gen=torch.Generator(device='cpu').manual_seed(17640+idx)
        before=torch.randint(0,256,(8*page,),dtype=torch.uint8,generator=gen)
        expected=before.clone();target_mask=torch.zeros_like(before,dtype=torch.bool)
        # Snapshot-based memmove reference: self-copy regions may overlap.
        for src,dst,size in regions(case,g):
            expected[dst:dst+size]=before[src:src+size]
            target_mask[dst:dst+size]=True
        raw=before.to('xpu');pages=raw.view(8,page)
        conv=pages[:,:g['conv_bytes']].view(torch.float16).view(8,6,g['conv_dim'])
        ssm=pages[:,g['conv_bytes']:g['conv_bytes']+g['temporal_bytes']].view(torch.float32).view(8,g['vheads'],128,128)
        assert conv.stride(0)*2==ssm.stride(0)*4==page
        assert ssm.data_ptr()-conv.data_ptr()==g['conv_bytes']
        bt=torch.tensor([BLOCK_IDS],dtype=torch.int32,device='xpu')
        pointers={'conv':conv.data_ptr(),'temporal':ssm.data_ptr(),'block_table':bt.data_ptr()}
        print(json.dumps(dict(event='pointer_range',index=idx,allocator=os.environ['PYTORCH_ALLOC_CONF'],pointers_hex={k:hex(v) for k,v in pointers.items()},int64_representable=all(0<=v<2**63 for v in pointers.values()))),flush=True)
        assert all(0<=v<2**63 for v in pointers.values()), 'actual int64 pointer metadata cannot represent allocation'
        dtypes=dict(state_base_addrs=torch.int64,state_block_strides=torch.int64,
                    state_elem_sizes=torch.int32,state_inner_sizes=torch.int64,
                    state_conv_widths=torch.int32,state_group_indices=torch.int32,
                    state_dim_row_count=torch.int32,state_dim_row_stride=torch.int64)
        ctx=NS(**{k:torch.zeros(2,dtype=v,device='xpu') for k,v in dtypes.items()},
               mamba_group_ids=[0],num_groups=1,num_layers=1,num_state_types=2,
               is_initialized=False,block_table_ptrs=torch.zeros(1,dtype=torch.int64,device='xpu'),
               block_size=1600,num_accepted_tokens_out=torch.full((4,),-777,dtype=torch.int32,device='xpu'))
        worker.MambaSpecDecodeGPUContext.initialize_from_forward_context(ctx,
            NS(kv_cache_groups=[NS(layer_names=['fixture'])]),
            {'fixture':NS(kv_cache=[conv,ssm])},
            (model.get_conv_copy_spec,model.get_temporal_copy_spec),[bt])
        metadata={k:getattr(ctx,k).cpu().tolist() for k in dtypes}
        assert metadata['state_block_strides']==[page,page]
        assert metadata['state_inner_sizes']==[g['conv_dim'],g['vheads']*128*128]
        tensor=lambda value:torch.tensor([value],dtype=torch.int32,device='xpu')
        accepted=tensor(case['accepted'])
        assert accepted.data_ptr()!=ctx.num_accepted_tokens_out.data_ptr()
        worker.MambaSpecDecodeGPUContext.run_fused_postprocess(ctx,1,accepted,
            tensor(case['src']),tensor(case['scheduled']),tensor(case['computed']),tensor(case['draft']))
        torch.xpu.synchronize()
        actual=raw.cpu();accepted_result=ctx.num_accepted_tokens_out.cpu().tolist()
        mismatches=int((actual!=expected).sum())
        untouched=torch.equal(actual[~target_mask],before[~target_mask])
        accepted_input_untouched=accepted.cpu().tolist()==[case['accepted']]
        accepted_correct=accepted_result==[case['accepted_out'],-777,-777,-777]
        row=dict(case=case,geometry=g,regions=regions(case,g),metadata=metadata,
            mismatched_bytes=mismatches,untargeted_bytes_untouched=untouched,
            accepted_result=accepted_result,accepted_correct=accepted_correct,
            accepted_input_untouched=accepted_input_untouched,
            accepted_buffers_distinct=True,
            passed=mismatches==0 and untouched and accepted_correct and accepted_input_untouched)
        results.append(row);print(json.dumps(dict(event='case',index=idx,**row)),flush=True)
    result=dict(execution='actual installed run_fused_postprocess -> postprocess_mamba_fused_kernel',
        worker_source_sha256=WORKER_SHA,model_source_sha256=MODEL_SHA,
        tp_size=args.tp_size,kv_layout=args.kv_layout,block_size=1600,cases=results,
        passed=all(r['passed'] for r in results),
        scope='Actual postprocess decision/byte copy/accepted reset only; no nativeGDN/math/model/graph/cancellation claim')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-xpu',action='store_true')
    p.add_argument('--tp-size',type=int,choices=[1,2],default=2)
    p.add_argument('--kv-layout',choices=['fp16','fp8'],default='fp8')
    args=p.parse_args()
    if not args.run_xpu:
        print(json.dumps(dict(status='PREPARED_NOT_RUN',tp_size=args.tp_size,kv_layout=args.kv_layout,
            cases=[dict(case=c,geometry=geometry(c,args.tp_size,args.kv_layout),regions=regions(c,geometry(c,args.tp_size,args.kv_layout))) for c in cases()],
            requirement='Owned selected-card lifecycle/devicepin/strictpreposthealth/timeout; no single-lease reset'),indent=2));return 0
    return run(args)


if __name__=='__main__':raise SystemExit(main())
