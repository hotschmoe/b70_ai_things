"""X3 exact XPU Ag/Rs operator timing at a 12288-token model chunk.

Run through xpu-run --card both, torchrun 2 workers. No topology env overrides.
Includes the repository's eager/compiled two-rank correctness oracle before
and after the budget calls, using the current campaign torch/runtime.
"""
from datetime import timedelta
import importlib.util
import json
import os
import sys
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
torch.xpu.set_device(int(os.environ['LOCAL_RANK']))
spec = importlib.util.spec_from_file_location('x3_health', '/mnt/vm_8tb/github/b70_ai_things/bin/xpu-collective-health.py')
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)
dist.init_process_group('xccl', timeout=timedelta(seconds=60))
print('X3_GROUP_READY',rank,flush=True)

def collective_health(stage):
    from torch.distributed import _functional_collectives as funcol
    dev=torch.device('xpu',int(os.environ['LOCAL_RANK']))
    base=torch.arange(4*5120,dtype=torch.int32,device=dev).remainder_(31).to(torch.bfloat16).reshape(4,5120)
    eager=health.make_input(base,rank,0)
    dist.all_reduce(eager)
    torch.xpu.synchronize()
    torch.testing.assert_close(eager,health.expected(base,2,0),atol=0,rtol=0)
    def reduce(value):
        return funcol.wait_tensor(funcol.all_reduce(value,'sum',dist.group.WORLD))
    compiled=torch.compile(reduce,fullgraph=True,dynamic=False)
    for iteration in range(10):
        value=compiled(health.make_input(base,rank,iteration))
        torch.xpu.synchronize()
        torch.testing.assert_close(value,health.expected(base,2,iteration),atol=0,rtol=0)
    dist.barrier()
    print('X3_COLLECTIVE_HEALTH_OK',stage,rank,'compiled_iterations=10',flush=True)


collective_health('before')
from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
from b70_gguf_xpu import prefill, prefill_dq, serve_prefill
from gguf import GGUFReader
import numpy as np

comm=XpuCommunicator.__new__(XpuCommunicator)
comm.world_size=2
comm.rank_in_group=rank
comm.device_group=dist.group.WORLD
comm.cpu_group=dist.new_group(backend='gloo',timeout=timedelta(seconds=120))
device=torch.device('xpu',int(os.environ['LOCAL_RANK']))
reader=GGUFReader('/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf')
ts={t.name:t for t in reader.tensors}
packs=[torch.from_numpy(np.array(ts[f'blk.3.ffn_{p}_exps.weight'].data[rank*256:(rank+1)*256],copy=True)).to(device) for p in ('gate','up','down')]
prefill.load()
os.environ.update(B70_PREFILL='1',B70_PREFILL_DQ_T='0',B70_PREFILL_REDUCE='grouped',B70_PREFILL_BUCKET='1',B70_PREFILL_EXPERT_TILE='64',B70_PREFILL_EP_TRACE='1')
# Initialized ballast approximates the loaded model's 23.92 GiB residency.
ballast=[torch.full((1024*1024*1024,),1,device=device,dtype=torch.uint8) for _ in range(22)]
torch.xpu.synchronize()
print('X3_LOADED_READY',rank,torch.xpu.memory_allocated(),flush=True)
for sync in ('1','0'):
    os.environ['B70_PREFILL_EP_SYNC']=sync
    for sizes in ([12288,12288],[12288,1]):
        n=sizes[rank]
        x=torch.full((n,2560),.125*(rank+1),device=device,dtype=torch.bfloat16)
        logits=torch.zeros(n,512,device=device,dtype=torch.float32)
        for iteration in range(4):
            print('X3_LOADED_BEGIN',rank,sync,sizes,iteration,flush=True)
            begin=time.perf_counter()
            marks=[torch.xpu.Event(enable_timing=True) for _ in range(4)]
            marks[0].record()
            xx,ll=comm.all_gatherv([x,logits],dim=0,sizes=sizes)
            marks[1].record()
            ids=torch.full((sum(sizes),10),-1,device=device,dtype=torch.int32)
            ids[:,:5]=torch.tensor([3,8,17,41,65],device=device,dtype=torch.int32)
            weights=torch.full((sum(sizes),10),.1,device=device,dtype=torch.float32)
            y=serve_prefill.moe_prefill(xx,ids,*packs,weights,16,16,42,2560,640)
            marks[2].record()
            print('X3_LOADED_EXPERT_DONE',rank,sync,iteration,flush=True)
            z=comm.reduce_scatterv(y,dim=0,sizes=sizes)
            marks[3].record()
            torch.xpu.synchronize()
            assert torch.isfinite(z).all().item()
            row=dict(rank=rank,sync=sync,sizes=sizes,iteration=iteration,
                     wall_ms=(time.perf_counter()-begin)*1000,
                     event_ms=[a.elapsed_time(b) for a,b in zip(marks,marks[1:])])
            with Path(f'/mnt/vm_8tb/b70/build/vllm-flashnext/x3/loaded-ep-rank{rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print('X3_LOADED_OK',json.dumps(row),flush=True)
            samples=y[[0,sizes[0]]].cpu()
            refs=[None,None]
            dist.all_gather_object(refs,samples,group=comm.cpu_group)
            expected=(refs[0]+refs[1])[rank].to(device)
            torch.testing.assert_close(z[0],expected,atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(z[-1],expected,atol=1e-5,rtol=1e-5)
            print('X3_LOADED_RS_EXACT',rank,sync,sizes,iteration,flush=True)
            del xx,ll,ids,weights,y,z
        del x,logits
    prefill_dq._SCRATCH.clear()
    torch.xpu.empty_cache()
del ballast,packs
prefill_dq._SCRATCH.clear()
torch.xpu.empty_cache()
collective_health('after')
dist.destroy_process_group()
print('X3_LOADED_EP_DONE',rank,flush=True)
