#!/usr/bin/env python3
"""Confirmed per-prefix GDN -> actual POST -> actual PRE -> GDN composition.

Plan only by default. Real execution requires the external selected lease.
Unsynced means no host readback/wait between those four operators; mandatory
metadata operations within actual helpers remain part of the tested chain.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

IMAGE='sha256:09fc6b75041566c9248d76b985f2a30a4d32e5a847389f637d2a6043436f3bce'
NATIVE_SHA='271db0d4882124e21ac6a4d080bfeab303fbb08b9ec10e11f21d10fb0723998f'
MODEL_SHA='25062fb6166498d77ec0e4efc0df5952e6384d16b3c0fdd8bcc2113448bb4fb5'
WORKER_SHA='318d958ed133dbe82081ea188721d14f7be1a9c1d40b1b05f393df5a8309c372'
BASE_ORACLE_SHA='22ee62b3495d46cc8fa72bb4c291d8e36abd317814ee23e0f13fd5073bf125e0'
BLOCKS=[3,5,1,6,2,7,0,4]


def cases():
    return [dict(name='self4_then_move',computed=4796,src=2,accepted=4,post_dst=2,post_bias=3,accepted_out=1,next_col=3)]+[
        dict(name='backward_accept'+str(a),computed=4798,src=3,accepted=a,post_dst=2 if a>=2 else None,post_bias=1 if a>=2 else None,accepted_out=a,next_col=3) for a in [1,2,3,4]]


def reference_step(torch,F,base,g,params,storage,qkv,ba,table,accepted):
    """Use confirmed per-prefix read and write with independent CPU recurrence."""
    raw,conv,ssm=storage;selected=table[accepted-1]
    window=conv[selected,:3].clone()
    out,z,unused,states=base.cpu_reference(torch,F,qkv,ba,*params,conv[selected],ssm[selected],1,g)
    kh,vh,d=g['kh'],g['vh'],g['dim'];projected=qkv[:,:2*kh*d+vh*d]
    joined=torch.cat([window,projected],dim=0)
    for i,slot in enumerate(table):
        conv[slot,:3]=joined[i+1:i+4]
        ssm[slot]=states[i]
    return out,z


def reference_copy(storage,src,dst,bias):
    _,conv,ssm=storage;source=BLOCKS[src+bias];dest=BLOCKS[dst]
    conv[dest,:3]=conv[source,:3].clone();ssm[dest]=ssm[source].clone()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-xpu',action='store_true');p.add_argument('--base-oracle',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    assert hashlib.sha256(a.base_oracle.read_bytes()).hexdigest()==BASE_ORACLE_SHA
    if not a.run_xpu:
        print(json.dumps(dict(status='PREPARED_NOT_RUN',image=IMAGE,cases=cases(),modes=['unsynced','synced'],contracts='per-prefix3rows; actual adapter metadata; logical no-move native; independent CPU; final readback only')));return 0
    assert a.output and not a.output.exists();assert os.environ.get('B70_XPU_GDN_PREFIX_CONV_COPY')=='1';assert os.environ.get('PYTORCH_ALLOC_CONF')=='expandable_segments:True';assert os.environ.get('ZE_AFFINITY_MASK') in ('0','1')
    import torch
    import torch.nn.functional as F
    from vllm.v1.worker import mamba_utils as worker
    from vllm.model_executor.layers.mamba import mamba_utils as model
    import vllm_xpu_kernels._xpu_C as native
    for path,digest in [(worker.__file__,WORKER_SHA),(model.__file__,MODEL_SHA),(native.__file__,NATIVE_SHA)]:assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    spec=importlib.util.spec_from_file_location('base_native_oracle',a.base_oracle);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    funcs=model.MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func();assert funcs==(model.get_xpu_gdn_conv_copy_spec,model.get_temporal_copy_spec)
    torch.set_num_threads(2);torch.xpu.set_device(0);g=base.geometry(1,'fp8');page=g['page'];rows=[];receipts=[]
    def storage(conv_data,ssm_data,device):
        raw=torch.full((8,page),0x5a,dtype=torch.uint8,device=device)
        conv=raw[:,:g['conv_bytes']].view(torch.float16).view(8,6,g['conv_dim'])
        ssm=raw[:,g['conv_bytes']:g['conv_bytes']+g['ssm_bytes']].view(torch.float32).view(8,g['vh'],128,128)
        conv.copy_(conv_data);ssm.copy_(ssm_data);assert conv.stride(0)*2==ssm.stride(0)*4==page
        return raw,conv,ssm
    def tensor(value):return torch.tensor([value],dtype=torch.int32,device='xpu')
    def err(x,y):
        x=x.float();y=y.float();diff=x-y
        return dict(max_abs=float(diff.abs().max()),relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(y).clamp_min(1e-8)))
    def math_pass(e):return e['max_abs']<=.005 and e['relative_l2']<=.03
    for ci,case in enumerate(cases()):
        rng=torch.Generator().manual_seed(20260910+ci)
        rand=lambda shape,scale:.5*(torch.randn(shape,generator=rng)*scale).half()
        conv0=rand((8,6,g['conv_dim']),.2);ssm0=rand((8,g['vh'],128,128),.02).float()
        params=base.make_parameters(torch,rng,g)
        qkvs=[rand((4,(2*g['kh']+2*g['vh'])*128),.2) for _ in range(2)];bas=[rand((4,2*g['vh']),.2) for _ in range(2)]
        first_table=BLOCKS[case['src']:case['src']+4];second_table=BLOCKS[case['next_col']:case['next_col']+4]
        # All CPU reference work precedes all tested GPU chains.
        expected=storage(conv0,ssm0,'cpu');rfirst=reference_step(torch,F,base,g,params,expected,qkvs[0],bas[0],first_table,1)
        if case['post_dst'] is not None:reference_copy(expected,case['src'],case['post_dst'],case['post_bias'])
        moved=case['src']!=case['next_col'];next_accepted=1 if moved else case['accepted_out']
        if moved:reference_copy(expected,case['src'],case['next_col'],case['accepted_out']-1)
        rsecond=reference_step(torch,F,base,g,params,expected,qkvs[1],bas[1],second_table,next_accepted)
        mode_results={}
        for synced in [False,True]:
            active=storage(conv0,ssm0,'xpu');nomove=storage(conv0,ssm0,'xpu')
            all_bt=torch.tensor([BLOCKS],dtype=torch.int32,device='xpu')
            bt1=torch.tensor([first_table],dtype=torch.int32,device='xpu');bt2=torch.tensor([second_table],dtype=torch.int32,device='xpu')
            params_x=[v.to('xpu') for v in params];qkv_x=[v.to('xpu') for v in qkvs];ba_x=[v.to('xpu') for v in bas]
            qs=torch.tensor([0,4],dtype=torch.int32,device='xpu');tokens=torch.arange(4,dtype=torch.int32,device='xpu');empty=torch.empty(0,dtype=torch.int32,device='xpu')
            outputs=[(torch.full((4,g['vh'],128),float('nan'),dtype=torch.float16,device='xpu'),torch.full((4,g['vh'],128),float('nan'),dtype=torch.float16,device='xpu')) for _ in range(4)]
            dtypes=dict(state_base_addrs=torch.int64,state_block_strides=torch.int64,state_elem_sizes=torch.int32,state_inner_sizes=torch.int64,state_conv_widths=torch.int32,state_group_indices=torch.int32,state_dim_row_count=torch.int32,state_dim_row_stride=torch.int64)
            ctx=NS(**{k:torch.zeros(2,dtype=v,device='xpu') for k,v in dtypes.items()},mamba_group_ids=[0],num_groups=1,num_layers=1,num_state_types=2,is_initialized=False,block_table_ptrs=torch.zeros(1,dtype=torch.int64,device='xpu'),block_size=1600,num_accepted_tokens_out=torch.full((4,),-777,dtype=torch.int32,device='xpu'))
            assert all(0<=v.data_ptr()<2**63 for v in [active[1],active[2],all_bt])
            worker.MambaSpecDecodeGPUContext.initialize_from_forward_context(ctx,NS(kv_cache_groups=[NS(layer_names=['gdn'])]),{'gdn':NS(kv_cache=list(active[1:]))},funcs,[all_bt])
            receipt=dict(copy_functions=[f.__name__ for f in funcs],conv_widths=ctx.state_conv_widths.cpu().tolist(),inner_sizes=ctx.state_inner_sizes.cpu().tolist(),block_strides=ctx.state_block_strides.cpu().tolist())
            assert receipt['conv_widths']==[0,0] and receipt['inner_sizes']==[3*g['conv_dim'],g['vh']*128*128] and receipt['block_strides']==[page,page]
            receipts.append(receipt)
            accepted=tensor(case['accepted']);one=tensor(1);computed=tensor(case['computed']);src=tensor(case['src']);dst=tensor(case['next_col']);scheduled=tensor(4);draft=tensor(3);bias=torch.empty(1,dtype=torch.int32,device='xpu')
            second_acc=one if moved else ctx.num_accepted_tokens_out[:1]
            def invoke(store,which,table,acc,outindex):
                out,z=outputs[outindex]
                torch.ops._xpu_C.gdn_attention(out,z,qkv_x[which],ba_x[which],16,48,128,128,store[1],store[2],params_x[0],None,'silu',params_x[1],params_x[2],0,0,1,None,None,empty,None,qs,tokens,table,acc,4,1,True)
            # Start with all allocations/uploads/context metadata complete.
            torch.xpu.synchronize()
            invoke(active,0,bt1,one,0)
            if synced:torch.xpu.synchronize()
            worker.MambaSpecDecodeGPUContext.run_fused_postprocess(ctx,1,accepted,src,scheduled,computed,draft)
            if synced:torch.xpu.synchronize()
            bias.copy_(ctx.num_accepted_tokens_out[:1]);bias.sub_(1)
            worker.MambaSpecDecodeGPUContext.run_fused_precopy(ctx,1,dst,src,bias,None)
            if synced:torch.xpu.synchronize()
            invoke(active,1,bt2,second_acc,1)
            # First readback/synchronization after the second native call.
            torch.xpu.synchronize()
            ar=[v.cpu() for v in active];aout=[(x.cpu(),z.cpu()) for x,z in outputs[:2]];accepted_out=ctx.num_accepted_tokens_out.cpu().tolist();accepted_in=accepted.cpu().tolist()
            # Independent native logical control has no publication/copy helpers.
            invoke(nomove,0,bt1,one,2)
            if synced:torch.xpu.synchronize()
            invoke(nomove,1,bt1,accepted,3)
            torch.xpu.synchronize();nr=[v.cpu() for v in nomove];nout=[(x.cpu(),z.cpu()) for x,z in outputs[2:]]
            touched=set(first_table+second_table)
            if case['post_dst'] is not None:touched.add(BLOCKS[case['post_dst']])
            if moved:touched.add(BLOCKS[case['next_col']])
            inactive=[i for i in range(8) if i not in touched]
            errors=dict(first=err(aout[0][0],rfirst[0]),second=err(aout[1][0],rsecond[0]),state=err(ar[2][sorted(touched)],expected[2][sorted(touched)]))
            checks=dict(finite=bool(torch.isfinite(aout[0][0]).all() and torch.isfinite(aout[1][0]).all() and torch.isfinite(ar[2]).all()),math=all(math_pass(e) for e in errors.values()),z_exact=torch.equal(aout[0][1],rfirst[1]) and torch.equal(aout[1][1],rsecond[1]),conv_exact=torch.equal(ar[1],expected[1]),padding_exact=torch.equal(ar[0][:,g['conv_bytes']+g['ssm_bytes']:],expected[0][:,g['conv_bytes']+g['ssm_bytes']:]),inactive_ssm_exact=torch.equal(ar[2][inactive],expected[2][inactive]),accepted_exact=accepted_out==[case['accepted_out'],-777,-777,-777] and accepted_in==[case['accepted']],nomove_output_exact=torch.equal(aout[1][0],nout[1][0]),nomove_z_exact=torch.equal(aout[1][1],nout[1][1]),nomove_conv_exact=torch.equal(ar[1][second_table,:3],nr[1][first_table,:3]),nomove_ssm_exact=torch.equal(ar[2][second_table],nr[2][first_table]))
            fingerprint=hashlib.sha256(ar[0].numpy().tobytes()+aout[1][0].numpy().tobytes()+aout[1][1].numpy().tobytes()).hexdigest();mode='synced' if synced else 'unsynced';mode_results[mode]=fingerprint
            row=dict(case=case,mode=mode,checks=checks,errors=errors,accepted_output=accepted_out,metadata=receipt,fingerprint=fingerprint,passed=all(checks.values()));rows.append(row);print(json.dumps(row),flush=True)
        equal=mode_results['synced']==mode_results['unsynced']
        for row in rows[-2:]:row['synced_unsynced_exact']=equal;row['passed'] &= equal
    result=dict(image=IMAGE,native_sha256=NATIVE_SHA,worker_sha256=WORKER_SHA,model_sha256=MODEL_SHA,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows,passed=len(rows)==10 and all(r['passed'] for r in rows),scope=__doc__)
    a.output.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
