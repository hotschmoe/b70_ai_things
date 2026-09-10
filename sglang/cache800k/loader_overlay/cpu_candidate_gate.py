"""No-device candidate import/digest/buffer/source gate. No model loading."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import torch
from sglang.srt.model_executor.b70_calibrated_kv import scale_loader,scale_plan

p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True)
p.add_argument('--old-artifact',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
a=p.parse_args()
assert not Path('/dev/dri').exists(),'CPU-only gate must not expose GPU devices'
manifest=json.loads((a.context/'manifest.json').read_bytes())
assert scale_loader.EXPECTED_ARTIFACT_SHA256==manifest['reviewed_artifact_sha256']
assert hashlib.sha256(Path(inspect.getfile(scale_loader)).read_bytes()).hexdigest()==manifest['inputs']['overlay/b70_calibrated_kv/scale_loader.py']
runner=Path(inspect.getfile(scale_loader)).parent.parent/'model_runner.py'
assert hashlib.sha256(runner.read_bytes()).hexdigest()==manifest['inputs']['overlay/model_runner.py']
tree=ast.parse(runner.read_text())
load=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='load_model')
calls={}
for node in ast.walk(load):
    if isinstance(node,ast.Call):calls.setdefault(ast.unparse(node.func),[]).append(node.lineno)
assert calls['try_load'][0]<calls['self.maybe_precompile_model_kernels_after_loading'][0]
assert calls['try_load'][0]<calls['load_kv_cache_scales'][0]
kwargs=dict(model=None,model_root='/',kv_dtype='fp8_e4m3',is_draft_worker=False,
    tp_rank=0,tp_size=1,pp_size=1,dcp_size=1,dp_attention=False,prefill_backend='triton',
    decode_backend='triton',device='cpu',gpu_id=0,model_dtype=torch.float16)
for path,expected in [(a.old_artifact,'unreviewed calibration artifact digest'),
                       (a.context/'fresh-scales.json','only reviewed for XPU')]:
    try:scale_loader.try_load(scale_path=str(path),**kwargs)
    except ValueError as exc:assert expected in str(exc),(expected,str(exc))
    else:raise AssertionError('digest/device gate did not reject before allocation')
class Layer(torch.nn.Module):
    def __init__(self,layer_id):
        super().__init__();self.layer_id=layer_id
        self.k_scale=self.v_scale=self.k_scale_float=self.v_scale_float=None
records=[]
for role in ('target','draft'):
    inventory=({f'model.layers.{i}.attn':i for i in range(3,64,4)} if role=='target'
               else {'model.model.layers.0.attn':0})
    for tp in (1,2):
        for rank in range(tp):
            plan=scale_plan.make_plan((a.context/'fresh-scales.json').read_bytes(),a.config.read_bytes(),
                role=role,tp_rank=rank,tp_size=tp,module_inventory=inventory)
            model=torch.nn.Module()
            for name,layer_id in inventory.items():
                node=model
                parts=name.split('.')
                for part in parts[:-1]:
                    if not hasattr(node,part):node.add_module(part,torch.nn.Module())
                    node=getattr(node,part)
                node.add_module(parts[-1],Layer(layer_id))
            n=scale_loader.install_validated_plan(model,plan,radix_attention_type=Layer,
                tensor_factory=lambda value:torch.tensor(value,dtype=torch.float32,device='cpu'))
            for name,module in model.named_modules():
                if not isinstance(module,Layer):continue
                for k in ('k','v'):
                    t=getattr(module,k+'_scale');value=getattr(module,k+'_scale_float')
                    assert t.device.type=='cpu' and t.dtype==torch.float32 and t.ndim==0
                    assert t.item()==value and torch.isfinite(t) and value>0
                    assert name+'.'+k+'_scale' in model.state_dict()
            records.append(dict(role=role,tp=tp,rank=rank,coverage=n))
print(json.dumps(dict(digest=scale_loader.EXPECTED_ARTIFACT_SHA256,
    old_artifact='rejected',new_artifact_digest='accepted before CPU device gate',
    load_order='before precompile and generic loader',cpu_buffer_plans=records),indent=2))
