"""P1 components. GPU execution requires an explicit coordinated card-1 slot.
Real weights/card-1 placement; synthetic routes. No serving speed claim.
"""
from pathlib import Path
import argparse, gc, importlib.util, json, os, signal, statistics, sys, time
import numpy as np
import torch
from gguf import GGUFReader

P=argparse.ArgumentParser()
P.add_argument('--out', required=True)
a=P.parse_args()
signal.alarm(180)
torch.set_grad_enabled(False)
torch.manual_seed(3301)
B=Path('/mnt/vm_8tb/b70/build/vllm-flashnext')
R=Path('/mnt/vm_8tb/github/b70_ai_things')
torch.ops.load_library(str(B/'f1/_C.f1bucket2.so'))
torch.ops.load_library(str(B/'p1/_C.p1selected.so'))
rows=[]

def save(row):
    rows.append(row)
    print(json.dumps(row),flush=True)
    Path(a.out).write_text(json.dumps(rows,indent=2)+'\n')

def exact(x,y,label):
    assert torch.equal(x.view(torch.int16),y.view(torch.int16)) if x.dtype==torch.bfloat16 else torch.equal(x.view(torch.int32),y.view(torch.int32)), label

def timing(fn, warm=2, count=5):
    for _ in range(warm): fn()
    torch.xpu.synchronize()
    ts=[]
    for _ in range(count):
        start,end=torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize();ts.append(start.elapsed_time(end))
    return statistics.median(ts)

# Existing real placement uses only CPU metadata; do not load EP transport.
spec=importlib.util.spec_from_file_location('p1_placement',R/'vllm/flashnext/plugins/b70_ep_xpu/b70_ep_xpu/placement.py')
placement=importlib.util.module_from_spec(spec); sys.modules[spec.name]=placement;spec.loader.exec_module(placement)
plans=placement.flashnext_plans(q0=138)
shards=sorted(Path('/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS').glob('*.gguf'))
readers=[GGUFReader(str(s)) for s in shards]
tensors={t.name:t for reader in readers for t in reader.tensors}

def pack(layer,kind):
    tt=tensors[f'blk.{layer}.ffn_{kind}_exps.weight']
    ar=np.asarray(tt.data).reshape(512,-1)
    p=torch.from_numpy(np.array(ar[list(plans[layer].card1)],copy=True)).to('xpu')
    return p,int(tt.tensor_type)

# Cover all five real gate/up types plus both down types and destination modes.
for layer in (0,1,8,12,35):
    for kind in ('gate','down'):
        q,typ=pack(layer,kind)
        er,cols=(640,2560) if kind=='gate' else (2560,640)
        sel=torch.tensor([373,0,191,7,23]+list(range(100,127)),dtype=torch.int32,device='xpu')
        for mode in ((0,1,2) if kind=='gate' else (0,)):
            pair=er if mode else 0
            n=sel.numel(); out_rows=n*er*(2 if mode else 1)
            ref=torch.full((out_rows,cols),float('nan'),dtype=torch.bfloat16,device='xpu')
            got=ref.clone()
            gathered=q.index_select(0,sel).flatten()
            torch.ops.b70_gguf.prefill_dequant_wide(gathered,typ,n*er,cols,ref,mode,pair)
            torch.ops.b70_p1.prefill_dequant_selected(q.flatten(),typ,n*er,cols,got,mode,pair,sel,er)
            torch.xpu.synchronize(); exact(ref,got,f'dq layer{layer}/{kind}/mode{mode}')
            row=dict(gate='dq',layer=layer,kind=kind,type=typ,mode=mode,selected_experts=n,exact=True)
            if mode == 0:
                def control():
                    packed=q.index_select(0,sel).flatten()
                    return torch.ops.b70_gguf.prefill_dequant_wide(packed,typ,n*er,cols,ref,mode,pair)
                def candidate():
                    return torch.ops.b70_p1.prefill_dequant_selected(q.flatten(),typ,n*er,cols,got,mode,pair,sel,er)
                row['gather_dequant_ms']=timing(control)
                row['selected_dequant_ms']=timing(candidate)
            save(row)
            del ref,got,gathered
        del q
    gc.collect();torch.xpu.empty_cache()

print('P1_COMPONENT_OK',flush=True)
