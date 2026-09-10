"""CPU only: reference publication, shape, and exact native call signature."""
import ast
import importlib.util
import json
from pathlib import Path
import torch
import torch.nn.functional as F

p=Path(__file__).with_name('oracle.py')
spec=importlib.util.spec_from_file_location('oracle',p);o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
torch.set_num_threads(2)
rows=[]
for tp in (1,2):
 g=o.geometry(tp,'fp8');kh,vh,d=g['kh'],g['vh'],128
 w_native,a_native,b_native=o.make_parameters(torch,torch.Generator().manual_seed(42),g)
 assert w_native.dtype==b_native.dtype==torch.float16 and a_native.dtype==torch.float32
 assert torch.count_nonzero(b_native)>0
 conv=torch.arange(6,dtype=torch.float16)[:,None].expand(6,g['conv_dim']).clone()
 state=torch.zeros(vh,d,d);weights=torch.zeros(g['conv_dim'],4)
 qkv=torch.zeros(4,(2*kh+2*vh)*d,dtype=torch.float16)
 qkv[:,-vh*d:]=torch.arange(4,dtype=torch.float16)[:,None]
 ba=torch.zeros(4,2*vh,dtype=torch.float16)
 for accepted in (1,2,3,4):
  out,z,newconv,states=o.cpu_reference(torch,F,qkv,ba,weights,torch.zeros(vh),torch.zeros(vh),conv,state,accepted,g)
  assert torch.count_nonzero(out)==torch.count_nonzero(states)==0
  assert torch.equal(z[:,0,0],torch.arange(4,dtype=torch.float16))
  assert torch.equal(newconv[:2,0],torch.tensor([accepted,accepted+1],dtype=torch.float16))
  assert torch.count_nonzero(newconv[2:])==0
  rows.append(dict(tp=tp,accepted=accepted,output_shape=list(out.shape),ssm_tape_shape=list(states.shape),conv_rows=list(newconv[:,0].float().tolist())))
# CPU alias shape matches XPU construction; no native operator is executed.
for tp in (1,2):
 for kv in ('fp16','fp8'):
  g=o.geometry(tp,kv);raw=torch.zeros(12,g['page'],dtype=torch.uint8)
  conv=raw[:,:g['conv_bytes']].view(torch.float16).view(12,6,g['conv_dim'])
  ssm=raw[:,g['conv_bytes']:g['conv_bytes']+g['ssm_bytes']].view(torch.float32).view(12,g['vh'],128,128)
  assert conv.stride(0)*2==ssm.stride(0)*4==g['page']
import vllm_xpu_kernels._xpu_C
schema=torch.ops._xpu_C.gdn_attention.default._schema
call=next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.ops._xpu_C.gdn_attention')
assert not call.keywords and len(call.args)==len(schema.arguments)
print(json.dumps(dict(rows=rows,native_schema_argument_count=len(schema.arguments),call_argument_count=len(call.args),verdict='PASS CPU only; zero/rank0 reference/accepted windows and actual ABI arity; no native execution'),indent=2))
