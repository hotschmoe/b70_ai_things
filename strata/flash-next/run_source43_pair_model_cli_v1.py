#!/usr/bin/env python3
"""Matched fresh43 one/two GPU full49 topology diagnostic; no new math."""
import ast,copy,os
from pathlib import Path
from types import SimpleNamespace
import run_source43_normal_model_cli_v1 as base
import source43_pair_topology_v1 as topology
from serial37_canonical_json_v3 import canonical
ROOT=base.ROOT;HERE=base.HERE;c=base.c;require=base.require;time=base.time;intake=base.intake;Protocol=base.Protocol;contract=base.contract
PLAN=HERE/'source43-pair-model-source-plan-v1.json';PHASES=topology.PHASES

def closure():
    for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Actual43 topology source changed '+name)
    base.closure();return c.sha(PLAN)
def admission(plan):return base.admission(plan)
def phase_recipe(*args,**kwargs):return topology.recipe(*args,**kwargs)
def memory_admission(*args,**kwargs):return topology.memory_admission(*args,**kwargs)
def native_command(directory,m,plan,pid,binding):
    name,command=base.life.contract.native_command(directory,m,plan,pid,binding)
    for card in plan['cards']:
        for path in('/dev/dri/card'+str(card),'/dev/dri/renderD'+str(128+card)):
            gid=str(os.stat(path).st_gid)
            if gid not in[command[i+1]for i,x in enumerate(command[:-1])if x=='--group-add']:
                at=command.index(m['runtime']['image']);command[at:at]=['--group-add',gid]
    return name,command
life=SimpleNamespace(**{**vars(base.life),'contract':SimpleNamespace(**{**vars(base.life.contract),'native_command':native_command})})

def ports():
    namespace=base.ports();namespace.update({**globals(),'__file__':__file__,'__name__':__name__})
    # The original robust phase's sole startup parser call is topology-aware;
    # native launch/streams/normal retirement remains the exact frozen body.
    node=copy.deepcopy(next(n for n in ast.parse(Path(base.original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='run_phase'));count=0
    for call in ast.walk(node):
        if isinstance(call,ast.Call)and ast.unparse(call.func)=='life.prepared_ctrl.memory_admission':call.func=ast.Name('memory_admission',ast.Load());call.args=[ast.Name('startup',ast.Load()),ast.Name('m',ast.Load()),ast.Name('plan',ast.Load())];count+=1
    require(count==1,'One exact startup memory call required');exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[actual43-pair-startup-only-port]','exec'),namespace);return namespace
_PORTS=ports()
def run_phase(*args,**kwargs):return _PORTS['run_phase'](*args,**kwargs)
def main():return _PORTS['main']()
if __name__=='__main__':main()
