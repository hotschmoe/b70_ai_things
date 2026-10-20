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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plugins'/'b70_gguf_xpu'))
from b70_gguf_xpu.prefill_reduce import pad_tokens, reduce_grouped

# Dynamic expert capacities must reuse compiled kernels. Check padding,
# invalid slots and slot-order weighted reduction at three capacities.
device=torch.device('xpu',int(os.environ['LOCAL_RANK']))
t,k,h,e=17,2,64,4
flat=[i%e for i in range(t*k)]
flat[0]=-1
order_cpu=[i for ex in range(e) for i,which in enumerate(flat) if which==ex]
starts_cpu=[0]
for ex in range(e):starts_cpu.append(starts_cpu[-1]+flat.count(ex))
order=torch.tensor(order_cpu,device=device,dtype=torch.int32)
starts=torch.tensor(starts_cpu,device=device,dtype=torch.int32)
x=torch.randn(t,h,device=device,dtype=torch.bfloat16)
weights=torch.rand(t,k,device=device)
for cap in (16,24,40):
    padded=torch.zeros(e,cap,h,device=device,dtype=torch.bfloat16)
    pad_tokens(x,padded,order,starts,k)
    expected=torch.zeros_like(padded)
    slots=torch.zeros(t*k,h,device=device,dtype=torch.bfloat16)
    for ex in range(e):
        for row,slot in enumerate(order_cpu[starts_cpu[ex]:starts_cpu[ex+1]]):
            expected[ex,row]=x[slot//k]
            slots[slot]=x[slot//k]
    torch.testing.assert_close(padded,expected,atol=0,rtol=0)
    got=reduce_grouped(padded,order,starts,weights,t,k,h)
    reference=(slots.view(t,k,h).float()*weights[:,:,None]).sum(1)
    torch.testing.assert_close(got,reference,atol=1e-6,rtol=1e-6)
print('X3_DYNAMIC_CAPACITY_OK',rank,flush=True)
from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator

comm = XpuCommunicator.__new__(XpuCommunicator)
comm.world_size = 2
comm.rank_in_group = rank
comm.device_group = dist.group.WORLD
rows = []
try:
    for sizes, label in (([12288,1], 'c1_12288_idle1'), ([6144,6144], 'balanced_global12288')):
        n = sizes[rank]
        device = torch.device('xpu', int(os.environ['LOCAL_RANK']))
        x = torch.full((n,2560),rank+1,dtype=torch.bfloat16,device=device)
        weights = torch.ones(n,10,dtype=torch.float32,device=device)
        ids = torch.full((n,10),rank,dtype=torch.int32,device=device)
        y = torch.full((sum(sizes),2560),rank+1,dtype=torch.bfloat16,device=device)
        def dispatch():
            return comm.all_gatherv([x,weights,ids],dim=0,sizes=sizes)
        def combine():
            return comm.reduce_scatterv(y,dim=0,sizes=sizes)
        for name, fn in (('dispatch_3_gathers',dispatch),('combine_reduce_scatter',combine)):
            print('X3_COLLECTIVE_BEGIN',rank,label,name,flush=True)
            for _ in range(3): fn()
            torch.xpu.synchronize()
            samples, walls = [], []
            for _ in range(5):
                dist.barrier()
                a,b = torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
                t=time.perf_counter();a.record();result=fn();b.record();b.synchronize()
                samples.append(a.elapsed_time(b));walls.append((time.perf_counter()-t)*1000)
            if name.startswith('dispatch'):
                assert torch.all(result[0][:sizes[0]]==1).item()
                assert torch.all(result[0][sizes[0]:]==2).item()
                assert torch.all(result[2][:sizes[0]]==0).item()
                assert torch.all(result[2][sizes[0]:]==1).item()
            else:
                assert torch.all(result==3).item()
            row=dict(rank=rank,label=label,op=name,sizes=sizes,event_ms=statistics.median(samples),wall_ms=statistics.median(walls),samples=samples)
            print('X3_COLLECTIVE',json.dumps(row),flush=True)
            rows.append(row)
    Path(f'/mnt/vm_8tb/b70/build/vllm-flashnext/x3/collectives-rank{rank}.json').write_text(json.dumps(rows,indent=2))
    collective_health('after')
finally:
    dist.destroy_process_group()
print('X3_COLLECTIVES_DONE',rank,flush=True)
