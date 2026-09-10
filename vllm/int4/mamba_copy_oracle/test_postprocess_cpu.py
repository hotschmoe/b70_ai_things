#!/usr/bin/env python3
"""Actual-source run_fused_postprocess + scalar decisions; no GPU/import vLLM."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('post',HERE/'postprocess_oracle.py');post=importlib.util.module_from_spec(spec);spec.loader.exec_module(post)
SOURCE=Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910/source-review/installed/vllm/v1/worker/mamba_utils.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==post.WORKER_SHA

def load(name,scope):
    fn=next(n for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.decorator_list=[]
    m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(m),str(SOURCE),'exec'),scope)
    return scope[name]

class Buffer:
    def __init__(self,value):self.data=np.asarray(value,dtype=np.int32)
    def __getitem__(self,item):
        out=object.__new__(Buffer);out.data=self.data[item];return out
    def __add__(self,offset):return self[offset:]
    def copy_(self,other):self.data[:]=other.data

def main():
    rows=[]
    for case in post.cases():
        copies=[];program=[0,0,0]
        scalar=load('postprocess_mamba_fused_kernel',dict(tl=NS(program_id=lambda axis:program[axis],load=lambda ptr:int(ptr.data[0]),store=lambda ptr,val:ptr.data.__setitem__(0,val)),_copy_mamba_state_block=lambda *a:copies.append((a[2],a[3],a[4]))))
        class Kernel:
            def __getitem__(self,grid):
                assert grid==(1,2,16)
                def launch(*args,**kw):
                    for state in range(2):
                        for tile in range(16):
                            program[:]=[0,state,tile];scalar(*args,**kw)
                return launch
        run=load('run_fused_postprocess',dict(postprocess_mamba_fused_kernel=Kernel(),_TEMPORAL_TILES=16,is_conv_state_dim_first=lambda:False))
        ctx=NS(is_initialized=True,num_layers=1,num_state_types=2,block_size=1600,
               num_accepted_tokens_out=Buffer([-777]*4),block_table_stride_req=8)
        for key in ['block_table_ptrs','state_base_addrs','state_block_strides','state_elem_sizes','state_inner_sizes','state_conv_widths','state_group_indices','state_dim_row_count','state_dim_row_stride']:setattr(ctx,key,None)
        accepted=Buffer([case['accepted']])
        run(ctx,1,accepted,Buffer([case['src']]),Buffer([case['scheduled']]),Buffer([case['computed']]),Buffer([case['draft']]))
        expected=[] if case['dst'] is None else [(case['src'],case['dst'],case['bias'])]
        assert sorted(set(copies))==expected,(case,copies)
        assert ctx.num_accepted_tokens_out.data.tolist()==[case['accepted_out'],-777,-777,-777]
        assert accepted.data.tolist()==[case['accepted']]
        # Independent byte oracle explicitly preserves self-overlap via snapshot.
        g=post.geometry(case,2,'fp8');before=np.random.default_rng(479600 + len(rows)).integers(0,256,8*g['page_bytes'],dtype=np.uint8)
        actual=before.copy();target=np.zeros(before.shape,dtype=bool)
        for src,dst,size in post.regions(case,g):
            actual[dst:dst+size]=before[src:src+size]
            target[dst:dst+size]=True
        assert np.array_equal(actual[~target],before[~target])
        rows.append(dict(case=case,actual_source_decisions=sorted(set(copies)),accepted_result=ctx.num_accepted_tokens_out.data.tolist(),self_overlap_covered=case['src']==case['dst']))
    print(json.dumps(dict(worker_sha256=post.WORKER_SHA,cases=rows,verdict='PASS actual helper/scalar postprocess control flow and expected memmove regions; no GPU bytecopy tested'),indent=2))

if __name__=='__main__':main()
