#!/usr/bin/env python3
"""Small source-bound H36 port over actual source41 mode0 model actor V3."""
import ast,copy,hashlib
from types import SimpleNamespace
from pathlib import Path
import run_source41_model_cli_v3 as original
import collect_host_critical_path36_v1 as collector
ROOT=original.ROOT;HERE=original.HERE;PLAN=HERE/'source41-host-latency-source-plan-v1.json';c=original.c;require=original.require
PHASES=('source41-host-OFF','source41-host-ON')
admission=original.admission;Protocol=original.Protocol;intake=original.intake;life=original.life;time=original.time
def equivalence(left,right):
    original.contract.equivalence(left,right)
    return {'H36_OFF_ON_equivalence_passed':True,'scope':'source41mode0 matched four bounded fresh native host-trace control only'}
contract=SimpleNamespace(**{**vars(original.contract),'equivalence':equivalence})

def closure():
    value=c.read(PLAN)
    for name,want in value['files'].items():require(c.sha(ROOT/name)==want,'Exact source41 host control changed '+name)
    original.closure();return c.sha(PLAN)

def phase_recipe(phase,metadata,cfg,inputs,binding):
    require(phase in PHASES,'Exactly source41 host OFF/ON supported')
    descriptor,recipe=original.phase_recipe('source41-off',metadata,cfg,inputs,binding)
    recipe['mode']=phase;recipe['host_trace_enabled']=phase==PHASES[1];recipe['logprobs_policy']='explicit20; includes full-vocabulary readback and LP CPU work';recipe['H36_export_IO_outside_request_span']=True
    require(recipe['materialized_mode']==0 and recipe['observer_enabled']is False,'Normal materializedOFF and numeric/QSA observers OFF required')
    if phase==PHASES[1]:recipe['native_env'].update(STRATA_DECODE_TIMING='1',STRATA_CRITICAL_PATH_TRACE='1',STRATA_CRITICAL_PATH_TRACE_DIR='/results/host-traces',STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256=binding)
    return descriptor,recipe

def run_phase(*args,**kwargs):
    result=_PORTS['native_phase'](*args,**kwargs);out,phase=args[:2];directory=out/phase
    if phase==PHASES[1]:
        files=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));rows=[r['raw']for r in c.read(directory/'requests.json')];coverage=collector.collect(files,directory/'engine.combined.log',args[5],rows,[[0,0,48]]);c.write(directory/'host-coverage.json',coverage)
    else:require(not list((directory/'host-traces').iterdir()),'H36 OFF unexpectedly exported records')
    return result

def ports():
    namespace={**vars(original),'__file__':__file__,'__name__':__name__,'PLAN':PLAN,'closure':closure,'phase_recipe':phase_recipe,'run_phase':run_phase,'contract':contract}
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


def preregister_lp_disabled(metadata,cfg,inputs,binding,phase):
    """Recipe-only next control; omission avoids req_logprobs>=0 readback branch."""
    descriptor,recipe=phase_recipe(phase,metadata,cfg,inputs,binding)
    recipe['planned_protocol_lines']=['GEN 64 fresh=1 ckpt=1 temperature=0 '+' '.join(map(str,row['ids']))for row in recipe['requests']]+['QUIT']
    recipe.update(logprobs_policy='omitted; native source default -1',LPdisabled_runtime_enabled=False,LP0_is_disabled=False,original_LP20_runtime_control_changed=False)
    return descriptor,recipe
