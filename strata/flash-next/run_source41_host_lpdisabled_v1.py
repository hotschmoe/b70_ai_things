#!/usr/bin/env python3
"""Small source-bound H36 port over actual source41 mode0 model actor V3."""
import ast,copy,hashlib
from types import SimpleNamespace
from pathlib import Path
import run_source41_model_cli_v3 as original
import collect_source41_lpdisabled_host_v1 as collector
import source41_lpdisabled_contract_v1 as omission
ROOT=original.ROOT;HERE=original.HERE;PLAN=HERE/'source41-host-lpdisabled-source-plan-v1.json';c=original.c;require=original.require
PHASES=('source41-host-OFF','source41-host-ON')
admission=original.admission;Protocol=omission.Protocol;intake=original.intake;life=original.life;time=original.time
equivalence=omission.equivalence
requests=omission.requests
contract=SimpleNamespace(**{**vars(original.contract),'equivalence':equivalence,'requests':requests})

def closure():
    value=c.read(PLAN)
    for name,want in value['files'].items():require(c.sha(ROOT/name)==want,'Exact source41 host control changed '+name)
    original.closure();return c.sha(PLAN)

def phase_recipe(phase,metadata,cfg,inputs,binding):
    require(phase in PHASES,'Exactly source41 host OFF/ON supported')
    descriptor,recipe=original.phase_recipe('source41-off',metadata,cfg,inputs,binding)
    recipe['mode']=phase;recipe['host_trace_enabled']=phase==PHASES[1];recipe['logprobs_policy']='omitted; native default -1, no LP readback/CPU work';recipe['H36_export_IO_outside_request_span']=True
    require(recipe['materialized_mode']==0 and recipe['observer_enabled']is False,'Normal materializedOFF and numeric/QSA observers OFF required')
    if phase==PHASES[1]:recipe['native_env'].update(STRATA_DECODE_TIMING='1',STRATA_CRITICAL_PATH_TRACE='1',STRATA_CRITICAL_PATH_TRACE_DIR='/results/host-traces',STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256=binding)
    return descriptor,recipe

def run_phase(*args,**kwargs):
    result=_PORTS['native_phase'](*args,**kwargs);out,phase=args[:2];directory=out/phase
    command=c.read(directory/'command.json');native_pid=collector.native_namespace(result['terminal'],command);c.write(directory/'native-namespace.json',{'native_pid':native_pid,'scope':'direct exec bash replacement/no-init/private PID namespace','terminal_command_source_recollected':True})
    rows=[r['raw']for r in c.read(directory/'requests.json')]
    for row in rows:collector.no_lp(row)
    if phase==PHASES[1]:
        files=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));rows=[r['raw']for r in c.read(directory/'requests.json')];coverage=collector.collect(files,directory/'engine.combined.log',args[5],rows,[[0,0,48]],native_pid);c.write(directory/'host-coverage.json',coverage)
    else:require(not list((directory/'host-traces').iterdir()),'H36 OFF unexpectedly exported records')
    return result

def ports():
    namespace={**vars(original),'__file__':__file__,'__name__':__name__,'PLAN':PLAN,'closure':closure,'phase_recipe':phase_recipe,'run_phase':run_phase,'contract':contract,'Protocol':Protocol}
    tree=ast.parse(Path(original.__file__).read_bytes());nodes=[];counts={'choices':0,'phases':0,'prerequisite':0,'comparison_phase':0,'comparison_name':0,'trace_directory':0,'unused_ARM':0}
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef)or node.name not in('run_phase','main'):continue
        node=copy.deepcopy(node)
        if node.name=='run_phase':
            node.name='native_phase'
            class RemoveUnusedArm(ast.NodeTransformer):
                def visit_Expr(self,n):
                    call=n.value
                    if isinstance(call,ast.Call)and isinstance(call.func,ast.Attribute)and call.func.attr=='write_text'and isinstance(call.func.value,ast.BinOp)and isinstance(call.func.value.right,ast.Constant)and call.func.value.right.value=='ARM':counts['unused_ARM']+=1;return None
                    return self.generic_visit(n)
            node=RemoveUnusedArm().visit(node)
            for n in ast.walk(node):
                if isinstance(n,ast.Tuple)and [getattr(x,'value',None)for x in n.elts]==['captures','p30','ple-input','qsa']:n.elts.append(ast.Constant('host-traces'));counts['trace_directory']+=1
        else:
            for n in ast.walk(node):
                if isinstance(n,ast.keyword)and n.arg=='choices'and isinstance(n.value,ast.Tuple)and [getattr(x,'value',None)for x in n.value.elts]==['off-equivalence','original-fidelity']:n.value.elts=[ast.Constant('host-latency')];counts['choices']+=1
                if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='phases'for t in n.targets):
                    require(isinstance(n.value,ast.IfExp),'Exact original conditional phase roster required');n.value=ast.Tuple([ast.Constant(p)for p in PHASES],ast.Load());counts['phases']+=1
                if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)and isinstance(n.test.left,ast.Attribute)and n.test.left.attr=='phase':
                    value=n.test.comparators[0]
                    if isinstance(value,ast.Constant)and value.value=='original-fidelity':value.value='host-latency';counts['prerequisite']+=1
                    elif isinstance(value,ast.Constant)and value.value=='off-equivalence':value.value='host-latency';counts['comparison_phase']+=1
                if isinstance(n,ast.Constant)and n.value=='OFF_comparison':n.value='H36_OFF_ON_comparison';counts['comparison_name']+=1
        nodes.append(node)
    require(counts=={'choices':1,'phases':1,'prerequisite':1,'comparison_phase':1,'comparison_name':1,'trace_directory':1,'unused_ARM':1},'Only seven exact named original lifecycle port edits permitted')
    exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[explicit-V3-host-port]','exec'),namespace);namespace['derivation_counts']=counts;return namespace
_PORTS=ports()
if __name__=='__main__':_PORTS['main']()


