"""Small actual-SGLang FP8 store/Triton read oracle. Default is CPU only.

XPU execution must be launched by the project GPU lease/health lifecycle.
This script never loads a model or verifies calibration/model provenance.
"""
import argparse
import ast
import hashlib
import json
import math
import struct
import inspect
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace as NS
import torch


def load_store(source, device):
    pool_path = source / 'python/sglang/srt/mem_cache/memory_pool.py'
    if device != 'cpu':
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        installed_pool=Path(inspect.getsourcefile(MHATokenToKVPool))
        assert installed_pool.read_bytes() == pool_path.read_bytes(), 'installed pool source mismatch'
        backend='layers/attention/triton_backend.py'
        assert (installed_pool.parents[1]/backend).read_bytes() == (source/'python/sglang/srt'/backend).read_bytes(), 'installed backend source mismatch'
        return MHATokenToKVPool.set_kv_buffer, MHATokenToKVPool._store_kv_layer
    # Execute exact source methods with the same fallback branch as XPU.
    # Only diagnostic no-ops/location unwrap and capture-mode query are stubbed.
    tree = ast.parse(pool_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MHATokenToKVPool')
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_set_kv_buffer_impl']
    nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('set_kv_buffer','_store_kv_layer')]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
    scope = dict(torch=torch, _is_cuda=False, _is_hip=False, _is_cpu=False,
                 unwrap_write_loc=lambda x:(x,None,None),
                 maybe_detect_oob=lambda *a:None, maybe_detect_kernel_facing_loc=lambda *a:None)
    runner = ModuleType('sglang.srt.model_executor.runner')
    runner.get_is_capture_mode = lambda:False
    sys.modules[runner.__name__] = runner
    exec(compile(module,str(pool_path),'exec'),scope)
    return scope['set_kv_buffer'], scope['_store_kv_layer']


def pool_fixture(source, device, slots, heads, dim):
    setter, store = load_store(source,device)
    pool = NS(size=slots-1,page_size=1,kernel_page_blocks=slots,
              is_quantized_kv_cache=False,dtype=torch.float8_e4m3fn,
              store_dtype=torch.uint8,use_hnd=False,kv_cache_layout='NHD',
              start_layer=0,row_dim=heads*dim,v_row_dim=heads*dim,
              alt_stream=None,device_module=torch.xpu if device!='cpu' else None,
              k_buffer=[torch.full((slots,heads,dim),0x35,dtype=torch.uint8,device=device)],
              v_buffer=[torch.full((slots,heads,dim),0x39,dtype=torch.uint8,device=device)])
    pool._store_kv_layer=MethodType(store,pool)
    pool.set_kv_buffer=MethodType(setter,pool)
    return pool


def reference(q,k,v,prefix):
    # Query j sees prefix + j + 1 tokens; FP32 oracle, original FP16 Q.
    k=k.float().repeat_interleave(q.shape[1]//k.shape[1],dim=1)
    v=v.float().repeat_interleave(q.shape[1]//v.shape[1],dim=1)
    scores=torch.einsum('qhd,khd->hqk',q.float(),k)/math.sqrt(q.shape[-1])
    allowed=torch.arange(k.shape[0])[None,:] <= prefix+torch.arange(q.shape[0])[:,None]
    scores.masked_fill_(~allowed.unsqueeze(0),float('-inf'))
    return torch.einsum('hqk,khd->qhd',scores.softmax(-1),v)


def install_fixture_scales(loader_root,device):
    sys.path.insert(0,str(loader_root))
    from scale_loader import install_validated_plan
    def f32(value):
        return struct.unpack('f',struct.pack('f',value))[0]
    class FixtureLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_id=0
            self.k_scale=self.v_scale=self.k_scale_float=self.v_scale_float=None
    model=torch.nn.Module();model.attn=FixtureLayer()
    # Numeric fixtures from prior layer3 calibration; NOT calibrated identity.
    k,v=f32(.029828752790178572),f32(.02104317801339286)
    plan={'rows':[{'module_path':'attn','layer_id':0,'k_scale':k,'v_scale':v}]}
    install_validated_plan(model,plan,radix_attention_type=FixtureLayer,
                           tensor_factory=lambda x:torch.tensor(x,dtype=torch.float32,device=device))
    return model.attn


def normalized_reference(value, scale, device):
    # Measured XPU TensorIterator device-scalar semantics: the zero-dimensional
    # FP32 scale is converted to the input dtype before division. CPU div_ has
    # different scalar handling. Use FP64 arithmetic to make the reference
    # independent of either device's division kernel, then round the result.
    effective = float(torch.tensor(scale, dtype=value.dtype)) if device == 'xpu' else scale
    assert math.isfinite(effective) and effective > 0
    return (value.double() / effective).to(value.dtype)


def exact_power2_store(source, device, layer, k, v, loc, loc_cpu, slots, heads, dim):
    pool = pool_fixture(source, device, slots, heads, dim)
    a, b = k.clone(), v.clone()
    pool.set_kv_buffer(layer, loc, a, b,
                      torch.tensor(.125, device=device), torch.tensor(.25, device=device))
    for name, actual, original, normalized, scale, sentinel in (
        ('K', pool.k_buffer[0].cpu(), k.cpu(), a.cpu(), .125, 0x35),
        ('V', pool.v_buffer[0].cpu(), v.cpu(), b.cpu(), .25, 0x39)):
        expected_half = (original.double() / scale).half()
        assert torch.equal(normalized, expected_half), name + ' power2 normalization mismatch'
        assert torch.equal(actual[loc_cpu], expected_half.to(torch.float8_e4m3fn).view(torch.uint8)), name + ' power2 bytes mismatch'
        untouched = torch.ones(slots, dtype=torch.bool); untouched[loc_cpu] = False
        assert actual[untouched].eq(sentinel).all(), name + ' power2 untouched slots changed'


def run(args):
    torch.set_num_threads(2)
    backend_path=args.source/'python/sglang/srt/layers/attention/triton_backend.py'
    backend_ast=ast.parse(backend_path.read_text())
    extend=next(n for n in ast.walk(backend_ast) if isinstance(n,ast.FunctionDef) and n.name=='forward_extend')
    clone_calls=[n for n in ast.walk(extend) if isinstance(n,ast.Call)
        and ast.unparse(n.func)=='self._set_kv_buffer' and len(n.args)==7
        and ast.unparse(n.args[3])=='k.clone()' and ast.unparse(n.args[4])=='v.clone()']
    assert len(clone_calls)==1, 'actual extend clone protection source changed'
    device=args.device
    n,slots,h,d,qh=67,160,4,256,24
    layer=install_fixture_scales(args.loader_root,device)
    ks,vs=layer.k_scale_float,layer.v_scale_float
    gen=torch.Generator(device='cpu').manual_seed(8123)
    k0=torch.randn(n,h,d,generator=gen).half()
    v0=torch.randn(n,h,d,generator=gen).half()
    # Nonidentity physical slot mapping crosses the 64-token tile boundary.
    loc_cpu=(torch.arange(n)*37+11)%slots
    assert len(set(loc_cpu.tolist()))==n
    pool=pool_fixture(args.source,device,slots,h,d)
    k,v=k0.to(device),v0.to(device)
    loc=loc_cpu.to(device)
    exact_power2_store(args.source,device,layer,k,v,loc,loc_cpu,slots,h,d)
    pool.set_kv_buffer(layer,loc,k.clone(),v.clone(),layer.k_scale,layer.v_scale)
    kb=pool.k_buffer[0].cpu();vb=pool.v_buffer[0].cpu()
    # XPU measured device-scalar conversion, FP16 division result, FP8 cast.
    nk=normalized_reference(k0,ks,device)
    nv=normalized_reference(v0,vs,device)
    ek=nk.to(torch.float8_e4m3fn)
    ev=nv.to(torch.float8_e4m3fn)
    original_nk=k0.clone().div_(torch.tensor(ks))
    original_nv=v0.clone().div_(torch.tensor(vs))
    assert torch.equal(kb[loc_cpu],ek.view(torch.uint8)),'K bytes differ from explicit scalar-promotion reference'
    assert torch.equal(vb[loc_cpu],ev.view(torch.uint8)),'V bytes differ from explicit scalar-promotion reference'
    untouched=torch.ones(slots,dtype=torch.bool);untouched[loc_cpu]=False
    assert kb[untouched].eq(0x35).all() and vb[untouched].eq(0x39).all()
    assert torch.equal(k.cpu(),k0) and torch.equal(v.cpu(),v0),'caller clone protection failed'
    # The pool API itself mutates FP16 K/V; extend's clone protection is required.
    direct_input_k=k.clone();direct_input_v=v.clone()
    pool.set_kv_buffer(layer,loc,direct_input_k,direct_input_v,layer.k_scale,layer.v_scale)
    assert torch.equal(direct_input_k.cpu(),nk)
    assert torch.equal(direct_input_v.cpu(),nv)
    assert not torch.equal(direct_input_k.cpu(),k0)
    directk=(k0.float()/ks).to(torch.float8_e4m3fn).view(torch.uint8)
    directv=(v0.float()/vs).to(torch.float8_e4m3fn).view(torch.uint8)
    result={'device':device,'torch_version':torch.__version__,'shape':{'sequence':n,'q_heads':qh,'kv_heads':h,'head_dim':d},
            'scale_fixture':{'k':ks,'v':vs,'identity':'numeric fixture only, no calibrated model claim'},
            'bytes':'pass','power2_exact_store':'pass','untouched_slots':'pass','cloned_inputs':'pass',
            'write_scalar_semantics':'scale rounded to FP16 before division' if device=='xpu' else 'CPU FP32 scalar division',
            'effective_write_scales':{'k':float(torch.tensor(ks).half()) if device=='xpu' else ks,
                                      'v':float(torch.tensor(vs).half()) if device=='xpu' else vs},
            'read_scales':{'k':ks,'v':vs},
            'original_fp32_scalar_reference_differences':{
                'k_half':int((nk.view(torch.int16)!=original_nk.view(torch.int16)).sum()),
                'v_half':int((nv.view(torch.int16)!=original_nv.view(torch.int16)).sum()),
                'k_fp8_bytes':int((ek.view(torch.uint8)!=original_nk.to(torch.float8_e4m3fn).view(torch.uint8)).sum()),
                'v_fp8_bytes':int((ev.view(torch.uint8)!=original_nv.to(torch.float8_e4m3fn).view(torch.uint8)).sum())},
            'direct_pool_input_mutation':'confirmed FP16 normalized values','actual_extend_clone_source':'pass',
            'fp16_intermediate_vs_fp32_direct_byte_differences':{
                'k':int((ek.view(torch.uint8)!=directk).sum()),
                'v':int((ev.view(torch.uint8)!=directv).sum())},'attention':[]}
    kd,vd=ek.float()*ks,ev.float()*vs
    result['dequant_write_rmse']={'k':float((kd-k0.float()).square().mean().sqrt()),
                                 'v':float((vd-v0.float()).square().mean().sqrt())}
    if device!='cpu':
        from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
        from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
        for fn,name in [(decode_attention_fwd,'decode_attention.py'),(extend_attention_fwd,'extend_attention.py')]:
            expected=args.source/'python/sglang/kernels/ops/attention'/name
            assert Path(inspect.getsourcefile(fn)).read_bytes()==expected.read_bytes(), 'installed attention source mismatch'
    for qlen in (1,4):
        q0=torch.randn(qlen,qh,d,generator=gen).half()
        q=q0.to(device)
        for mode in (('decode','extend') if qlen==1 else ('extend',)):
            prefix=n-qlen
            expected_baseline=None
            actual_baseline=None
            for label,km,vm in [('correct',1.,1.),('K_descaling_x2',2.,1.),('V_descaling_x2',1.,2.)]:
                rk,rv=ek.float()*(ks*km),ev.float()*(vs*vm)
                if mode=='extend':
                    # Fresh suffix attends in original FP16, cached prefix is dequantized.
                    rk=torch.cat([rk[:prefix],k0[prefix:].float()])
                    rv=torch.cat([rv[:prefix],v0[prefix:].float()])
                expected=reference(q0,rk,rv,prefix)
                row={'mode':mode,'query_length':qlen,'scale_case':label,
                     'reference_norm':float(expected.norm())}
                if label=='correct':
                    expected_baseline=expected
                else:
                    row['reference_scale_sensitivity_l2']=float((expected-expected_baseline).norm())
                    assert row['reference_scale_sensitivity_l2']>1e-3
                if device!='cpu':
                    out=torch.empty_like(q)
                    kbuf=pool.k_buffer[0].view(torch.float8_e4m3fn)
                    vbuf=pool.v_buffer[0].view(torch.float8_e4m3fn)
                    if mode=='decode':
                        splits=2
                        decode_attention_fwd(q,kbuf,vbuf,out,
                            torch.tensor([0,n],dtype=torch.int32,device=device),loc,
                            torch.empty((1,qh,splits,d),device=device,dtype=torch.float32),
                            torch.empty((1,qh,splits),device=device,dtype=torch.float32),
                            torch.tensor([splits],dtype=torch.int32,device=device),splits,
                            d**-.5,ks*km,vs*vm,page_size=1,enable_lean=False)
                    else:
                        extend_attention_fwd(q,k[prefix:].contiguous(),v[prefix:].contiguous(),out,
                            kbuf,vbuf,torch.tensor([0,qlen],dtype=torch.int32,device=device),
                            torch.tensor([0,prefix],dtype=torch.int32,device=device),loc[:prefix],
                            None,True,None,qlen,ks*km,vs*vm,d**-.5,page_size=1)
                    actual=out.cpu().float()
                    row.update(max_abs_error=float((actual-expected).abs().max()),
                               rmse=float((actual-expected).square().mean().sqrt()))
                    # Retain every case for diagnosis without weakening the gate.
                    row['close_mismatched_elements']=int((~torch.isclose(actual,expected,rtol=.03,atol=.015)).sum())
                    try:
                        torch.testing.assert_close(actual,expected,rtol=.03,atol=.015)
                        row['close_gate']='pass'
                    except AssertionError as exc:
                        row['close_gate']='fail'
                        row['close_error']=str(exc)
                    if label=='correct':
                        actual_baseline=actual
                    else:
                        row['actual_scale_sensitivity_l2']=float((actual-actual_baseline).norm())
                        row['scale_sensitivity_gate']='pass' if row['actual_scale_sensitivity_l2']>1e-3 else 'fail'
                result['attention'].append(row)
    result['attention_execution']='CPU reference only' if device=='cpu' else 'actual Triton kernels'
    result['source_hashes']={p:hashlib.sha256((args.source/p).read_bytes()).hexdigest() for p in (
        'python/sglang/srt/mem_cache/memory_pool.py',
        'python/sglang/srt/layers/attention/triton_backend.py',
        'python/sglang/kernels/ops/attention/decode_attention.py',
        'python/sglang/kernels/ops/attention/extend_attention.py')}
    result['loader_source_sha256']=hashlib.sha256((args.loader_root/'scale_loader.py').read_bytes()).hexdigest()
    result['attention_gate']='fail' if any(r.get('close_gate')=='fail' or r.get('scale_sensitivity_gate')=='fail' for r in result['attention']) else 'pass'
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','xpu'],default='cpu')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--loader-root',type=Path,required=True)
    result=run(parser.parse_args())
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(1 if result['attention_gate']=='fail' else 0)
