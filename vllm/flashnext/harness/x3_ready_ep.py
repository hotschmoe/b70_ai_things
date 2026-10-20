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
ballast=[torch.full((1024*1024*1024,),1,device=device,dtype=torch.uint8) for _ in range(24)]
torch.xpu.synchronize()
print('X3_LOADED_READY',rank,torch.xpu.memory_allocated(),flush=True)

from types import SimpleNamespace
from vllm.config import CUDAGraphMode
from vllm.forward_context import override_forward_context
from b70_gguf_xpu.prefill_policy import is_long_prefill
for mode,sizes,rows,want in ((CUDAGraphMode.NONE,[8,8],16,False),(CUDAGraphMode.FULL,[16,16],32,False),(CUDAGraphMode.NONE,[12288,1],12289,True)):
    context=SimpleNamespace(cudagraph_runtime_mode=mode,dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(sizes)))
    with override_forward_context(context):
        assert is_long_prefill(rows)==want
        if not want:
            assert comm._prefill_ready("AG",rows) is None
print('X3_PREFILL_POLICY_OK',rank,flush=True)
from b70_gguf_xpu.prefill_memory import profile_allocator
previous=torch.xpu.get_per_process_memory_fraction()
os.environ['B70_PREFILL_ALLOC_FRACTION']='0.90'
try:
    with profile_allocator():
        assert abs(torch.xpu.get_per_process_memory_fraction()-.90)<1e-6
        raise ValueError('X3 allocator restore sentinel')
except ValueError as exc:
    assert str(exc)=='X3 allocator restore sentinel'
assert torch.xpu.get_per_process_memory_fraction()==previous
os.environ.pop('B70_PREFILL_ALLOC_FRACTION')
print('X3_PROFILE_ALLOC_RESTORE_OK',rank,flush=True)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
os.environ.update(B70_PREFILL_EP_DRAIN='1',B70_PREFILL_EP_SYNC='1',B70_PREFILL_EP_TRACE='0')
sizes=[12288,12288]
n=sizes[rank]
torch.manual_seed(303+rank)
seed=torch.randn(1,2560,device=device,dtype=torch.bfloat16).expand(n,-1).contiguous()
wp=torch.randn(2560,2560,device=device,dtype=torch.bfloat16)*.01
wgu=torch.randn(1280,2560,device=device,dtype=torch.bfloat16)*.01
wd=torch.randn(2560,640,device=device,dtype=torch.bfloat16)*.01
rt=ts['blk.3.ffn_gate_inp.weight']
router=torch.from_numpy(np.array(rt.data,copy=True))
if int(rt.tensor_type)==30:
    router=router.view(torch.bfloat16)
elif int(rt.tensor_type) not in (0,1):
    raise RuntimeError('router fixture must be an unquantized GGUF tensor')
router=router.reshape(512,2560).to(device,torch.bfloat16)
aux=torch.xpu.Stream(device=device)
rows=[]
allocator_phase='baseline'
for iteration in range(48):
    if iteration==24:
        try:
            before=torch._C._accelerator_getAllocatorSettings()
            torch._C._accelerator_setAllocatorSettings('garbage_collection_threshold:0.9')
            after=torch._C._accelerator_getAllocatorSettings()
            allocator_phase='gc0.9'
            print('X3_ALLOC_GC_CONFIG',rank,repr(before),repr(after),flush=True)
        except RuntimeError as exc:
            print('X3_ALLOC_GC_UNAVAILABLE',rank,str(exc),flush=True)
    start=time.perf_counter()
    x=torch.nn.functional.rms_norm(torch.nn.functional.linear(seed,wp),(2560,))
    aux.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(aux):
        gu=torch.nn.functional.linear(x,wgu)
        shared=torch.nn.functional.linear(torch.nn.functional.silu(gu[:,:640])*gu[:,640:],wd)
    logits=torch.nn.functional.linear(x,router).float()
    a,b,c,d=[torch.xpu.Event(enable_timing=True) for _ in range(4)]
    a.record()
    xx,ll=comm.all_gatherv([x,logits],dim=0,sizes=sizes)
    b.record()
    weights,global_ids,_=fused_topk(xx,ll,10,True)
    ids=torch.where((global_ids>=rank*256)&(global_ids<(rank+1)*256),global_ids-rank*256,-1).to(torch.int32)
    y=serve_prefill.moe_prefill(xx,ids,*packs,weights,16,16,42,2560,640)
    c.record()
    z=comm.reduce_scatterv(y.to(torch.bfloat16),dim=0,sizes=sizes)
    d.record();torch.xpu.synchronize()
    assert torch.isfinite(z).all().item()
    row=dict(rank=rank,iteration=iteration,allocator_phase=allocator_phase,wall_ms=(time.perf_counter()-start)*1000,ag_ms=a.elapsed_time(b),expert_ms=b.elapsed_time(c),rs_ms=c.elapsed_time(d),allocated=torch.xpu.memory_allocated(),reserved=torch.xpu.memory_reserved(),free=torch.xpu.mem_get_info()[0])
    rows.append(row)
    print('X3_READY_EP',json.dumps(row),flush=True)
    # Same-row inputs per rank give two exact reference rows, captured only
    # after all queued work completes so the test does not hide readiness.
    samples=y.to(torch.bfloat16)[[0,sizes[0],sizes[0]-1,sum(sizes)-1]].cpu().float()
    refs=[None,None]
    dist.all_gather_object(refs,samples,group=comm.cpu_group)
    expected=(refs[0]+refs[1]).to(device,torch.bfloat16)
    torch.testing.assert_close(z[0],expected[rank],atol=0,rtol=0)
    torch.testing.assert_close(z[-1],expected[rank+2],atol=0,rtol=0)
    seed=x
    del xx,ll,weights,global_ids,ids,y,z,gu,shared,logits
Path(f'/mnt/vm_8tb/b70/build/vllm-flashnext/x3/ready-ep-rank{rank}.json').write_text(json.dumps(rows,indent=2))
del ballast,packs,seed,x,wp,wgu,wd,router
prefill_dq._SCRATCH.clear();torch.xpu.empty_cache()
collective_health('after')
dist.destroy_process_group()
print('X3_READY_EP_DONE',rank,flush=True)
