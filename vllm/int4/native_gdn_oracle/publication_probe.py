#!/usr/bin/env python3
"""Observe actual native conv publication/read contract; never promotes a backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import oracle as base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-xpu',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not args.run_xpu:
        print(json.dumps(dict(mode='CPU plan only',image=base.IMAGE,native_sha256=base.NATIVE_SHA,
            tp_size=1,kv_layout='fp8',physical_state_columns=[7,3,9,2],calls='one publication + four independent accepted-count read controls')));return 0
    assert args.output is not None and not args.output.exists()
    assert os.environ.get('PYTORCH_ALLOC_CONF')=='expandable_segments:True'
    assert os.environ.get('ZE_AFFINITY_MASK') in ('0','1')
    import torch
    import torch.nn.functional as F
    import vllm_xpu_kernels._xpu_C as native
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()==base.NATIVE_SHA
    torch.set_num_threads(2);torch.xpu.set_device(0)
    g=base.geometry(1,'fp8');slots=12;page=g['page'];columns=[7,3,9,2]
    rng=torch.Generator(device='cpu').manual_seed(172335)
    rand=lambda shape,scale=.1:(torch.randn(shape,generator=rng)*scale).half()
    weights,alog,bias=base.make_parameters(torch,rng,g)
    weights_x=weights.to('xpu');alog_x=alog.to('xpu');bias_x=bias.to('xpu')
    initial_conv=rand((slots,6,g['conv_dim']));initial_ssm=rand((slots,g['vh'],128,128),.01).float()
    def views(raw):
        pages=raw.view(slots,page)
        conv=pages[:,:g['conv_bytes']].view(torch.float16).view(slots,6,g['conv_dim'])
        ssm=pages[:,g['conv_bytes']:g['conv_bytes']+g['ssm_bytes']].view(torch.float32).view(slots,g['vh'],128,128)
        assert conv.stride(0)*2==ssm.stride(0)*4==page
        return conv,ssm
    raw=torch.full((slots*page,),0x5a,dtype=torch.uint8,device='xpu');conv,ssm=views(raw)
    conv.copy_(initial_conv);ssm.copy_(initial_ssm)
    qs=torch.tensor([0,4],dtype=torch.int32,device='xpu');tokens=torch.arange(4,dtype=torch.int32,device='xpu')
    table=torch.tensor([columns],dtype=torch.int32,device='xpu')
    pointers=dict(conv=conv.data_ptr(),ssm=ssm.data_ptr(),block_table=table.data_ptr())
    print(json.dumps(dict(event='pointer_range',pointers_hex={k:hex(v) for k,v in pointers.items()},int64_representable=all(0<=v<2**63 for v in pointers.values()))),flush=True)
    assert all(0<=v<2**63 for v in pointers.values())
    def inputs():return rand((4,(2*g['kh']+2*g['vh'])*128)),rand((4,2*g['vh']))
    def call(raw,qkv,ba,accepted):
        cv,st=views(raw);out=torch.full((4,g['vh'],128),float('nan'),dtype=torch.float16,device='xpu');z=torch.full_like(out,float('nan'))
        torch.ops._xpu_C.gdn_attention(out,z,qkv.to('xpu'),ba.to('xpu'),16,48,128,128,
            cv,st,weights_x,None,'silu',alog_x,bias_x,0,0,1,None,None,
            torch.empty(0,dtype=torch.int32,device='xpu'),None,qs,tokens,table,
            torch.tensor([accepted],dtype=torch.int32,device='xpu'),4,1,True)
        torch.xpu.synchronize()
        return out.cpu(),z.cpu(),cv.cpu(),st.cpu()
    def error(a,b):
        aa=a.float();bb=b.float();diff=aa-bb
        return dict(max_abs=float(diff.abs().max()),relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(bb).clamp_min(1e-8)))
    def match(a,b):
        e=error(a,b);return dict(**e,within_frozen_math_gate=e['max_abs']<=.005 and e['relative_l2']<=.03)
    qkv,ba=inputs();before_raw=raw.cpu()
    out,z,after_conv,after_ssm=call(raw,qkv,ba,1);after_raw=raw.cpu()
    q,k,v,zexpected=qkv.split([2048,2048,6144,6144],dim=-1);newrows=torch.cat([q,k,v],dim=-1)
    joined=torch.cat([initial_conv[columns[0],:3],newrows],dim=0)
    reference=base.cpu_reference(torch,F,qkv,ba,weights,alog,bias,initial_conv[columns[0]],initial_ssm[columns[0]],1,g)
    known={f'before_slot{s}_row{r}':initial_conv[s,r] for s in range(slots) for r in range(6)}
    known.update({f'new_qkv_token{t}':newrows[t] for t in range(4)})
    row_map=[]
    for slot in range(slots):
        for row in range(6):
            value=after_conv[slot,row]
            changed=int((value.view(torch.uint8)!=initial_conv[slot,row].view(torch.uint8)).sum())
            labels=[name for name,vec in known.items() if torch.equal(value,vec)]
            row_map.append(dict(physical_slot=slot,row=row,changed_bytes=changed,exact_source_labels=labels,head_values=value[:8].float().tolist()))
    publication=dict(row_map=row_map,
        each_prefix_first3=[torch.equal(after_conv[slot,:3],joined[j+1:j+4]) for j,slot in enumerate(columns)],
        each_prefix_tail3_untouched=[torch.equal(after_conv[slot,3:],initial_conv[slot,3:]) for slot in columns],
        rolling_column0_exact=torch.equal(after_conv[columns[0]],reference[2]),
        noncolumn0_conv_untouched=torch.equal(after_conv[[s for s in range(slots) if s!=columns[0]]],initial_conv[[s for s in range(slots) if s!=columns[0]]]),
        ssm_prefix_error=match(after_ssm[columns],reference[3]),output_error=match(out,reference[0]),z_exact=torch.equal(z,zexpected.reshape(4,48,128)),
        page_padding_exact=torch.equal(after_raw.view(slots,page)[:,g['conv_bytes']+g['ssm_bytes']:],before_raw.view(slots,page)[:,g['conv_bytes']+g['ssm_bytes']:]))
    torch.save(dict(geometry=g,columns=columns,weights=weights,alog=alog,bias=bias,qkv=qkv,ba=ba,
        conv_before=initial_conv,conv_after=after_conv,ssm_before=initial_ssm,ssm_after=after_ssm,
        output=out,z=z),args.output.parent/'publication-tensors.pt')
    next_qkv,next_ba=inputs();reads=[]
    for accepted in [1,2,3,4]:
        fresh=after_raw.to('xpu');o2,z2,c2,s2=call(fresh,next_qkv,next_ba,accepted)
        # HypothesisA: worker rolling-column0/accepted-window contract.
        rolling=base.cpu_reference(torch,F,next_qkv,next_ba,weights,alog,bias,
            after_conv[columns[0]],after_ssm[columns[accepted-1]],accepted,g)
        # HypothesisB: native per-prefix column, first3 history, no window offset.
        selected=after_conv[columns[accepted-1]].clone()
        legacy=base.cpu_reference(torch,F,next_qkv,next_ba,weights,alog,bias,
            selected,after_ssm[columns[accepted-1]],1,g)
        row=dict(accepted=accepted,physical_source_column=columns[accepted-1],
            rolling_output=match(o2,rolling[0]),per_prefix_output=match(o2,legacy[0]),
            rolling_ssm=match(s2[columns],rolling[3]),per_prefix_ssm=match(s2[columns],legacy[3]),
            outputs_finite=bool(torch.isfinite(o2).all() and torch.isfinite(s2[columns]).all()),
            expected_hypotheses_differ=not torch.equal(rolling[0],legacy[0]))
        reads.append(row)
        torch.save(dict(accepted=accepted,qkv=next_qkv,ba=next_ba,output=o2,z=z2,conv_after=c2,
            ssm_prefixes=s2[columns],rolling_reference_output=rolling[0],per_prefix_reference_output=legacy[0]),
            args.output.parent/f'read-accepted{accepted}-tensors.pt')
        print(json.dumps(dict(event='read_case',**row)),flush=True)
    complete=len(reads)==4 and all(r['outputs_finite'] for r in reads) and publication['z_exact'] and publication['page_padding_exact']
    result=dict(image=base.IMAGE,native_sha256=base.NATIVE_SHA,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),geometry=g,physical_state_columns=columns,
        publication=publication,reads=reads,diagnostic_completed=complete,
        scope='Observed native conv publication/read semantics only; original rolling-contract failure preserved; no backend qualification')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='summary',diagnostic_completed=complete,publication_prefix_matches=publication['each_prefix_first3'],rolling_column0_exact=publication['rolling_column0_exact'])),flush=True)
    return 0 if complete else 1


if __name__=='__main__':raise SystemExit(main())
