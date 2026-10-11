#!/usr/bin/env python3
"""Same bounded V2 speed lifecycle, exact new single-card control routing."""
import ast
from pathlib import Path
import sys
import run_matched_geometry_speed_v2 as previous
import prefill_chunk64_512_control_v2 as matched
ROOT=previous.ROOT
PLAN=ROOT/'strata/flash-next/prefill-chunk64-512-control-source-plan-v2.json'


def sources():
    for path,digest in previous.c1.read(PLAN)['files'].items():
        assert previous.c1.sha(ROOT/path)==digest,'Exact chunk-control source changed '+path
    previous.q.source_pins()
    return previous.c1.sha(PLAN)


def parent_ports():
    if sys.flags.optimize:raise ValueError('Ordinary Python required')
    tree=ast.parse(Path(previous.__file__).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');counts=[0,0]
    for n in ast.walk(node):
        if isinstance(n,ast.Constant)and n.value=='strata/flash-next/matched_geometry_c140_diagnostic_v2.py':n.value='strata/flash-next/prefill_chunk64_512_control_v2.py';counts[0]+=1
        if isinstance(n,ast.Tuple)and [getattr(x,'value',None)for x in n.elts]==['matched-one','matched-pair']:
            n.elts=[ast.Constant(value='chunk64'),ast.Constant(value='chunk512')];counts[1]+=1
    assert counts==[1,1],'Exact controller and profile routes required'
    count=0
    for parent in ast.walk(node):
        for field,value in ast.iter_fields(parent):
            if not isinstance(value,list):continue
            for index,stmt in enumerate(value):
                if isinstance(stmt,ast.Expr)and isinstance(stmt.value,ast.Call)and isinstance(stmt.value.func,ast.Name)and stmt.value.func.id=='run'and len(stmt.value.args)>1 and isinstance(stmt.value.args[1],ast.Constant)and stmt.value.args[1].value=='speed-client':
                    value.insert(index,ast.parse('matched.memory_admission(out,m)').body[0]);count+=1;break
    assert count==1,'Exactly one actual startup memory admission before timed client'
    namespace=dict(vars(previous));namespace.update(__file__=__file__,__name__=__name__,matched=matched,PLAN=PLAN,sources=sources)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
    return namespace


def main():
    parent_ports()['main']()


if __name__=='__main__':main()
