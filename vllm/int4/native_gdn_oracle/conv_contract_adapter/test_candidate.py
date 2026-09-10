#!/usr/bin/env python3
"""Actual-source CPU copy-spec, context metadata and fused address contracts."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import os
import sys
from types import ModuleType
from unittest.mock import patch
import tempfile

ROOT=Path('/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/conv-contract-adapter-candidate')
MODEL=ROOT/'vllm/model_executor/layers/mamba/mamba_utils.py'
WORKER=ROOT/'vllm/v1/worker/mamba_utils.py'

def load(path,name,scope):
    fn=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name);fn.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope);return scope[name]

class State:
    def __init__(self,array,base=100000):self.a=array;self.base=base
    def __getitem__(self,index):
        v=self.a[index];return State(v,self.base+v.ctypes.data-self.a.ctypes.data)
    def data_ptr(self):return self.base
    def dim(self):return self.a.ndim
    def size(self,i):return self.a.shape[i]
    def stride(self,i):return self.a.strides[i]//self.a.itemsize
    def element_size(self):return self.a.itemsize
    def numel(self):return self.a.size

class Pointer:
    def __init__(self,values,index=0):self.values=values;self.index=index
    def __add__(self,i):return Pointer(self.values,self.index+int(i))
class Scalar(int):
    def to(self,dtype):return Pointer(BLOCKS) if dtype=='pointer' else self
BLOCKS=[3,5,1,6,2,7,0,4]

def main():
    scope=dict(is_conv_state_dim_first=lambda:False,MambaCopySpec=lambda **kw:NS(**kw))
    normal=load(MODEL,'get_conv_copy_spec',scope);marker=load(MODEL,'get_xpu_gdn_conv_copy_spec',scope);temporal=load(MODEL,'get_temporal_copy_spec',scope)
    init=load(WORKER,'initialize_from_forward_context',dict(get_conv_copy_spec=normal,get_xpu_gdn_conv_copy_spec=marker,get_temporal_copy_spec=temporal,is_conv_state_dim_first=lambda:False))
    count=0
    for tp in [1,2]:
        dim=10240//tp;page=3276800//tp;raw=np.zeros((8,page),dtype=np.uint8)
        conv=State(raw[:,:6*dim*2].view(np.float16).reshape(8,6,dim))
        for func in [normal,marker]:
            names=['state_base_addrs','state_block_strides','state_elem_sizes','state_inner_sizes','state_conv_widths','state_group_indices','state_dim_row_count','state_dim_row_stride']
            ctx=NS(is_initialized=False,mamba_group_ids=[0],num_groups=1,block_table_ptrs=[0],**{n:[0] for n in names})
            bt=State(np.asarray([BLOCKS],dtype=np.int32),10000)
            init(ctx,NS(kv_cache_groups=[NS(layer_names=['gdn'])]),{'gdn':NS(kv_cache=[conv])},(func,),[bt])
            assert ctx.state_conv_widths==([0] if func is marker else [6])
            assert ctx.state_inner_sizes==([3*dim] if func is marker else [dim])
            if func is normal:continue
            captures=[]
            scalar=load(WORKER,'_copy_mamba_state_block',dict(tl=NS(load=lambda p:Scalar(p.values[p.index]),int64='int64',int32='int32',pointer_type=lambda t:'pointer'),_memcpy_u64_tiled=lambda src,dst,size,tile,**kw:captures.append((int(src),int(dst),int(size)))))
            for src,dst,bias in [(2,2,3),(3,2,1),(2,3,0),(3,2,2),(3,2,3)]:
                spec=marker(conv,BLOCKS,src,bias+1);captures.clear()
                scalar(0,0,src,dst,bias,Pointer([10000]),8,*[Pointer(getattr(ctx,n)) for n in names[:5]],Pointer(ctx.state_group_indices),Pointer(ctx.state_dim_row_count),Pointer(ctx.state_dim_row_stride),0,1024,False,16)
                assert captures==[(spec.start_addr,conv.base+BLOCKS[dst]*page,spec.num_elements*2)],(captures,spec)
                assert spec.num_elements==3*dim;count+=1
    # Only GDN selects the marker; default selector remains exact old pair.
    choice=load(MODEL,'gated_delta_net_state_copy_func',dict(get_conv_copy_spec=normal,get_xpu_gdn_conv_copy_spec=marker,get_temporal_copy_spec=temporal,_use_xpu_gdn_prefix_conv_copy=lambda:False))
    assert choice(None)==(normal,temporal)
    gate=load(MODEL,'_use_xpu_gdn_prefix_conv_copy',dict(is_conv_state_dim_first=lambda:False))
    with patch.dict(os.environ,{'B70_XPU_GDN_PREFIX_CONV_COPY':'0'}):assert gate() is False
    with patch.dict(os.environ,{'B70_XPU_GDN_PREFIX_CONV_COPY':'invalid'}):
        try:gate();raise AssertionError('invalid flag accepted')
        except ValueError:pass
    platform=ModuleType('vllm.platforms');platform.current_platform=NS(is_xpu=lambda:False)
    with patch.dict(os.environ,{'B70_XPU_GDN_PREFIX_CONV_COPY':'1'}),patch.dict(sys.modules,{'vllm.platforms':platform}):
        try:gate();raise AssertionError('non-XPU accepted')
        except RuntimeError as exc:assert 'non-XPU' in str(exc)
        platform.current_platform.is_xpu=lambda:True
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'wrong native');f.flush();native=ModuleType('vllm_xpu_kernels._xpu_C');native.__file__=f.name;package=ModuleType('vllm_xpu_kernels');package._xpu_C=native
            with patch.dict(sys.modules,{'vllm_xpu_kernels':package,'vllm_xpu_kernels._xpu_C':native}):
                try:gate();raise AssertionError('wrong native accepted')
                except RuntimeError as exc:assert 'identity mismatch' in str(exc)
    print('PASS '+str(count)+' actual fused-address/CPU-spec contracts; metadata/default route; invalid flag/non-XPU/native mismatch fail closed')
if __name__=='__main__':main()
