"""CPU-only pinned ELF/source/env preparation checks. No Torch import."""
import hashlib
import json
from pathlib import Path
import subprocess
R=Path('/mnt/vm_8tb/github/b70_ai_things')
B=Path('/mnt/vm_8tb/b70/build/vllm-flashnext')
H=R/'vllm/flashnext/harness'
EP=B/'f1/b70_ep_C.f1c.so'
O4=R/'vllm/flashnext/plugins/b70_o4_xpu/b70_o4_xpu/_C.so'
EXPECTED={str(EP):'249cf94e44a9a31225058886d67677e83cde270527f74edfc333be1ef141b8c0',
          str(O4):'0377d20b562fc00671117c5977a93cfc259fe5a584d11bdce78d481067136af9'}
for p,h in EXPECTED.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
b=EP.read_bytes();o=O4.read_bytes()
assert b'moe_ep_route(Tensor x, Tensor logits, int layer, int topk, int renorm) -> Tensor' in b
assert b'prefill_init' in b
assert b'w8_up_collapse' in o and b'w8_down_act' in o
syms=subprocess.run(['nm','-D','-C',str(EP)],capture_output=True,text=True,check=True).stdout
assert 'b70ep::ep_moe_route(' in syms and 'b70ep::ep_route_dispatch(' in syms
assert 'b70ep::b70_ep_route_dispatch_k' in syms
base={}
for p in [H/'o4_recipe.env',H/'f1_prefill_delta.env']:
 for line in p.read_text().splitlines():
  line=line.split('#',1)[0].strip()
  if '='in line:
   k,v=line.split('=',1);base[k]=v.strip()
assert base['B70_EP_SO']==str(EP)
assert base['B70_DECODE_HC_Q8']=='1' and base['B70_EP_FUSE_SHARED']=='1'
assert all(k not in base for k in ['B70_DECODE_HC_Q8F','B70_EP_ROUTE','B70_DECODE_HC_FUSE','B70_FA_KV_SPLITS'])
control=dict(base,B70_GDN_BA_DETERMINISTIC='0',B70_D1_FA_PLAN='0')
treatment=dict(control,B70_EP_ROUTE='1',B70_DECODE_HC_Q8F='1')
print(json.dumps({'verdict':'CPU_ELF_SOURCE_COMPATIBLE; not a loaded route execution qualification',
 'SO_sha256':EXPECTED,'route_symbols':[l for l in syms.splitlines()if 'ep_moe_route('in l or 'ep_route_dispatch('in l],
 'route_schema_present':True,'prefill_schema_present':True,'control_env':control,
 'treatment_env':treatment,'treatment_only_delta':{k:treatment[k]for k in treatment if k not in control},
 'expected_no_rebuild':True,'torch_imported':False,'gpu_executed':False},indent=2))
