#!/usr/bin/env python3
"""CPU actual-reference composition and no-readback source guard."""
import argparse
import ast
import importlib.util
from pathlib import Path
import json

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-oracle',type=Path,required=True);a=p.parse_args();here=Path(__file__).parent;o=module(here/'oracle.py','composition');base=module(a.base_oracle,'native_reference')
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);g=base.geometry(1,'fp8');rng=torch.Generator().manual_seed(40);rows=[]
    for case in o.cases():
        conv=torch.randn(8,6,g['conv_dim'],generator=rng).half()*.1;ssm=torch.randn(8,g['vh'],128,128,generator=rng)*.01
        first=o.BLOCKS[case['src']:case['src']+4];second=o.BLOCKS[case['next_col']:case['next_col']+4];params=base.make_parameters(torch,rng,g)
        qkv=[torch.randn(4,(2*g['kh']+2*g['vh'])*128,generator=rng).half()*.1 for _ in range(2)];ba=[torch.randn(4,2*g['vh'],generator=rng).half()*.1 for _ in range(2)]
        active=(None,conv.clone(),ssm.clone());nomove=(None,conv.clone(),ssm.clone())
        for state in [active,nomove]:o.reference_step(torch,F,base,g,params,state,qkv[0],ba[0],first,1)
        if case['post_dst'] is not None:o.reference_copy(active,case['src'],case['post_dst'],case['post_bias'])
        moved=case['src']!=case['next_col']
        if moved:o.reference_copy(active,case['src'],case['next_col'],case['accepted_out']-1)
        actual=o.reference_step(torch,F,base,g,params,active,qkv[1],ba[1],second,1 if moved else case['accepted_out'])
        wanted=o.reference_step(torch,F,base,g,params,nomove,qkv[1],ba[1],first,case['accepted'])
        assert torch.equal(actual[0],wanted[0]);assert torch.equal(active[1][second,:3],nomove[1][first,:3]);assert torch.equal(active[2][second],nomove[2][first]);rows.append(case['name'])
    # The actual chain's only synchronization calls are explicit synced guards.
    source=(here/'oracle.py').read_text();start=source.index('            invoke(active,0');end=source.index('            # First readback',start);tree=ast.parse('def chain():\n'+source[start:end])
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ('cpu','item','tolist','numpy')
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.xpu.synchronize':
            assert any(isinstance(parent,ast.If) and ast.unparse(parent.test)=='synced' and node in parent.body for parent in ast.walk(tree))
    print(json.dumps(dict(cases=rows,verdict='PASS five CPU composed-vs-nomove exact contracts and no intermediate host-readback guard')))
if __name__=='__main__':main()
