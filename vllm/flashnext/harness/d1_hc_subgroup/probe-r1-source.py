"""D1 M1 up/collapse component; run only with explicit root GPU grant."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

S = Path(__file__).resolve().parent
R = Path('/mnt/vm_8tb/github/b70_ai_things')
B = Path('/mnt/vm_8tb/b70/build/vllm-flashnext')
HC = B/'vllm/vllm/models/qwen4_exp/xpu/hc_decode.py'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output)
    signal.alarm(120)
    import torch
    torch.set_num_threads(2)
    assert torch.xpu.device_count()==1
    original_so=Path(os.environ['B70_O4_SO'])
    torch.ops.load_library(str(original_so))
    torch.ops.load_library(str(S/'libd1_hc_sg_r1.so'))
    spec=importlib.util.spec_from_file_location('d1_hc_original',HC)
    hc=importlib.util.module_from_spec(spec);spec.loader.exec_module(hc)
    fused=torch.ops.b70_o4.w8_up_collapse
    cand=torch.ops.b70_d1_hc_sg_r1.up_collapse_m1
    def original(x,w,s,flat):
        gate=torch.ops.b70_o4.w8_gemv(x,w,s,-8)
        y=torch.empty((1,2560),dtype=torch.bfloat16,device='xpu')
        hc._hc_collapse[(1,5)](flat,gate,y,2560,4,512)
        return y
    variants={'original':original,'existing_fused':lambda x,w,s,f:fused(x,w,s,f,2560),'candidate':cand}
    result={'scope':'M1 only; no model/EP/PLE/near-full-VRAM/serving claim',
            'device':torch.xpu.get_device_name(0),'torch':torch.__version__,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (S/'upcol.cpp',S/'probe.py',S/'libd1_hc_sg_r1.so',original_so,HC)},
            'gate':[],'timing':[]}
    torch.manual_seed(20261021)
    inputs=[]
    for i in range(96):
        # Distinct model-sized packed Q8 weights/scales and ready activation/flat
        # buffers. No quantizer, down/activation readiness, EP or HC-norm cost.
        w=torch.randint(-127,128,(10240,320),device='xpu',dtype=torch.int8)
        s=(torch.rand((10240,10),device='xpu')*.009+.001).half()
        act=(torch.randn((1,320),device='xpu')*.1).bfloat16()
        flat=(torch.randn((1,10240),device='xpu')*.2).bfloat16()
        inputs.append((act,w,s,flat))
    # Numerical gate includes finite random, allzero, large gates/sigmoid tails.
    # Original operations retain the real baseline LPR8 sum/rounding sequence.
    for case,scale in [('random',1.),('zero',0.),('large_gate',100.)]:
        for i in (0,1,95):
            x,w,s,f=inputs[i]
            x2=(x.float()*scale).bfloat16()
            ys={n:fn(x2,w,s,f)for n,fn in variants.items()}
            torch.xpu.synchronize()
            for n,y in ys.items():
                assert torch.equal(y.view(torch.int16),ys['original'].view(torch.int16)),(case,i,n)
            result['gate'].append({'case':case,'weight_index':i,'all_bitexact':True})
    def capture(fn):
        held=fn();torch.xpu.synchronize()
        g=torch.xpu.XPUGraph()
        with torch.xpu.graph(g): held=fn()
        g.replay();torch.xpu.synchronize()
        return g,held
    def timed(g):
        # Three repeated timing samples per graph; no per-op event overlay.
        vals=[]
        for _ in range(3):
            start=torch.xpu.Event(enable_timing=True);end=torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(12):g.replay()
            end.record();end.synchronize();vals.append(start.elapsed_time(end)/12)
        return vals
    for count in (1,96):
        graphs={};outputs={}
        for n,fn in variants.items():
            graphs[n],outputs[n]=capture(lambda fn=fn:[fn(*x)for x in inputs[:count]])
        # Validate every distinct weight, then same graphs with changed live inputs.
        for i in range(count):
            for n in variants:
                assert torch.equal(outputs[n][i].view(torch.int16),outputs['original'][i].view(torch.int16)),(count,i,n)
        for n,g in graphs.items():
            vals=timed(g)
            result['timing'].append({'chain_weights':count,'variant':n,'graph_ms_samples':vals,
                'weight_scale_bytes':count*10240*320*17//16,
                'ready_act_flat_bytes':count*(320+10240)*2})
        for x,w,s,f in inputs[:count]:
            x.mul_(-.5);f.add_(.125)
        for g in graphs.values():g.replay()
        torch.xpu.synchronize()
        for i in range(count):
            for n in variants:
                assert torch.equal(outputs[n][i].view(torch.int16),outputs['original'][i].view(torch.int16)),('changed',count,i,n)
        del graphs,outputs
    # Explicit production fallback contract: helper itself must reject M2.
    x,w,s,f=inputs[0]
    try:cand(x.expand(2,-1).contiguous(),w,s,f.expand(2,-1).contiguous())
    except RuntimeError:result['M2_rejected']=True
    else:raise AssertionError('M2 should require original fallback')
    result['verdict']='COMPONENT_PASS; timing is ready-input M1 HC only'
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
