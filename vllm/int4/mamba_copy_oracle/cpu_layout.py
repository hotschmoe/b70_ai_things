"""CPU-only actual metadata/copy-spec AST check; never imports vLLM or uses XPU."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import oracle


def function(source,name):
    node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[];node.returns=None
    for arg in node.args.args:arg.annotation=None
    return node


root=Path('/opt/venv/lib/python3.12/site-packages/vllm')
worker=(root/'v1/worker/mamba_utils.py').read_text()
model=(root/'model_executor/layers/mamba/mamba_utils.py').read_text()
assert hashlib.sha256(worker.encode()).hexdigest()==oracle.WORKER_SHA
assert hashlib.sha256(model.encode()).hexdigest()==oracle.MODEL_SHA
scope=dict(torch=torch,is_conv_state_dim_first=lambda:False,MambaCopySpec=NS,logger=NS(warning_once=lambda *a:None))
for name in ['get_conv_copy_spec','get_temporal_copy_spec']:
    exec(compile(ast.Module(body=[function(model,name)],type_ignores=[]),'<actual-copy-spec>','exec'),scope)
exec(compile(ast.Module(body=[function(worker,'initialize_from_forward_context')],type_ignores=[]),'<actual-metadata>','exec'),scope)
rows=[]
for case in oracle.cases():
    g=oracle.geometry(case);page=g['page_bytes'];raw=torch.zeros(8*page,dtype=torch.uint8);pages=raw.view(8,page)
    conv=pages[:,:g['conv_bytes']].view(torch.float16).view(8,case['history'],10240)
    temporal=pages[:,g['conv_bytes']:g['conv_bytes']+g['temporal_bytes']].view(torch.float32).view(8,48,128,128)
    names=['state_base_addrs','state_block_strides','state_elem_sizes','state_inner_sizes','state_conv_widths','state_group_indices','state_dim_row_count','state_dim_row_stride']
    ctx=NS(**{name:torch.zeros(2,dtype=torch.int64) for name in names},mamba_group_ids=[0],num_groups=1,is_initialized=False,block_table_ptrs=torch.zeros(1,dtype=torch.int64))
    bt=torch.tensor([oracle.BLOCK_IDS],dtype=torch.int32)
    scope['initialize_from_forward_context'](ctx,NS(kv_cache_groups=[NS(layer_names=['fixture'])]),{'fixture':NS(kv_cache=[conv,temporal])},(scope['get_conv_copy_spec'],scope['get_temporal_copy_spec']),[bt])
    assert ctx.state_block_strides.tolist()==[page,page]
    assert ctx.state_inner_sizes.tolist()==[10240,48*128*128]
    assert temporal.data_ptr()-conv.data_ptr()==g['conv_bytes']
    expected=oracle.regions(case)
    actual=[]
    if expected:
        for state,fn in [(conv,scope['get_conv_copy_spec']),(temporal,scope['get_temporal_copy_spec'])]:
            spec=fn(state,oracle.BLOCK_IDS,case['src'],case['bias']+1)
            actual.append((spec.start_addr-raw.data_ptr(),state[oracle.BLOCK_IDS[case['dst']]].data_ptr()-raw.data_ptr(),spec.num_elements*state.element_size()))
        assert actual==expected,(actual,expected)
        # Reference independence: no copy destination overlaps any source region.
        for src,dst,size in actual:
            for src2,dst2,size2 in actual:assert dst+size<=src2 or src2+size2<=dst
    rows.append(dict(case=case,geometry=g,regions=actual,metadata_stride=ctx.state_block_strides.tolist()))
print(json.dumps(dict(passed=True,cases=rows,execution='CPU metadata and actual copy-spec AST only; kernels NOT RUN'),indent=2))
