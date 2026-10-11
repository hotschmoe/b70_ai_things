#!/usr/bin/env python3
"""Explicit decode-only H36 reader over unchanged raw records; no device time."""
import ast
import copy
from pathlib import Path
import collect_host_critical_path36_v1 as original


def collect(paths,log,binding,requests,stages):
    tree=ast.parse(Path(original.__file__).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='collect')
    count=0
    for call in ast.walk(node):
        if isinstance(call,ast.Call)and isinstance(call.func,ast.Name)and call.func.id=='require'and len(call.args)==2 and isinstance(call.args[1],ast.Constant)and call.args[1].value=='Actual complete serial prompt/decode row coverage missing/duplicate':
            call.args[0]=ast.parse('covered == expected or covered == list(range(len(request["ids"])-1,len(request["ids"])+len(token_events)-1))',mode='eval').body
            call.args[1]=ast.Constant(value='Exact complete decode positions required; prompt verifier internals may be unobserved');count+=1
    original.require(count==1,'Exactly one declared prompt coverage scope change')
    namespace=dict(vars(original));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
    result=namespace['collect'](paths,log,binding,requests,stages)
    result.update(scope='bounded source40 HOST decode coverage; batched prompt internals unobserved; durations overlap',
                  prompt_internal_device_time_observed=False,physical_expert_traffic_observed=False,
                  reader_source_sha256=original.sha(__file__),original_reader_source_sha256=original.sha(original.__file__))
    return result
