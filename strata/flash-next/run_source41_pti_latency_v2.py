#!/usr/bin/env python3
"""Thin actual41 LPdisabled control: both H36 ON; fresh PTI ON only."""
import ast,copy,os,signal,time
from pathlib import Path
from types import SimpleNamespace
from serial37_canonical_json_v3 import canonical
import run_source41_host_lpdisabled_v1 as base
import source41_pti_runtime_v1 as tool
from source41_pti_raw_budget_v2 import Watcher,TOTAL,INTERVAL,publish_final
ROOT=base.ROOT;HERE=base.HERE;c=base.c;require=base.require;PLAN=HERE/'source41-pti-latency-source-plan-v2.json';PHASES=('source41-pti-OFF','source41-pti-ON');_RUNTIME=None
Protocol=base.Protocol;intake=base.intake;contract=base.contract;collector=base.collector

def closure():
    for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Exact real-model PTI source changed '+name)
    base.closure();return c.sha(PLAN)

def set_runtime(path):
    global _RUNTIME
    _RUNTIME=tool.binding(path)

def admission(plan):return base.admission(plan)
def phase_recipe(phase,metadata,cfg,inputs,binding):
    require(phase in PHASES and _RUNTIME is not None,'Exact filled fresh runtime/phase required');tool.inherited_env(cfg['env']);require(_RUNTIME['image']==metadata['runtime']['image'],'Actual profiler/model same image required')
    descriptor,recipe=base.phase_recipe(base.PHASES[1],metadata,cfg,inputs,binding);recipe.update(mode=phase,PTI_enabled=phase==PHASES[1],both_arms_H36_enabled=True,total_raw_cap_bytes=TOTAL,total_raw_sample_seconds=INTERVAL,metadata_cap_bytes=512<<20,tool_runtime=_RUNTIME)
    if recipe['PTI_enabled']:
        recipe['native_env'].update(tool.CLOCK_ENV);recipe['native_argv']=['/pti/bin/unitrace',*tool.OPTIONS,*recipe['native_argv']]
    return descriptor,recipe

def native_command(directory,m,plan,pid,binding):
    name,command=base.life.contract.native_command(directory,m,plan,pid,binding)
    if plan['PTI_enabled']:
        index=command.index(m['runtime']['image']);command[index:index]=['-v',_RUNTIME['install_root']+':/pti:ro']
    return name,command
life=SimpleNamespace(**{**vars(base.life),'contract':SimpleNamespace(**{**vars(base.life.contract),'native_command':native_command})})

def native_pid(terminal,command,pti):
    if not pti:return collector.native_namespace(terminal,command)
    host=terminal['HostConfig'];require(type(host.get('PidMode'))is str and host['PidMode']in('','private')and(host.get('Init')is None or host.get('Init')is False)and '--init'not in command and '--pid'not in command,'Actual PTI native private/no-init namespace required');require(terminal['Config']['Entrypoint']==['/bin/bash']and command[-2]=='-c'and command[-1].startswith('exec 2>&1; source /opt/intel/oneapi/setvars.sh ')and '; exec /pti/bin/unitrace 'in command[-1],'Actual direct PTI launcher exec required')
    expected=tool.CLOCK_ENV;env=terminal['Config']['Env'];require(all(sum(x.startswith(k+'=')for x in env)==1 and k+'='+v in env for k,v in expected.items()),'Actual clock flags differ');require(not any(x.startswith('UNITRACE_')and x.split('=',1)[0]not in expected for x in env),'Metrics/KMD/other profiler overrides forbid native exec branch')
    return 1

def run_phase(*args,**kwargs):
    out,phase=args[:2];watch=Watcher(out,lambda:os.kill(os.getpid(),signal.SIGUSR1));watch.start();value=None;error=None;directory=out/phase
    try:
        value=_PORTS['native_phase'](*args,**kwargs);plan=c.read(directory/'recipe.json');pid=native_pid(value['terminal'],c.read(directory/'command.json'),plan['PTI_enabled']);c.write(directory/'native-namespace.json',{'native_pid':pid,'PTI_enabled':plan['PTI_enabled'],'exec_branch_source_checked':True})
        (directory/'leaf.stderr').write_bytes((directory/'engine.combined.log').read_bytes());rows=[r['raw']for r in c.read(directory/'requests.json')];files=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));c.write(directory/'host-coverage.json',collector.collect(files,directory/'engine.combined.log',args[5],rows,[[0,0,48]],pid));require(canonical(tool.binding(_RUNTIME['intake_path']))==canonical(_RUNTIME),'Fresh profiler bytes/source changed during arm')
    except BaseException as exc:error=exc
    finally:
        # All raw/coverage writes above are included before watcher retirement.
        proof=watch.close()
        if directory.is_dir():proof=publish_final(directory/'raw-budget.json',proof,out)
    if error is not None:raise error
    require(proof['passed'],'Actual raw postwrite budget failed; original actor already retired');return value

def final_total_budget(out,report):
    # Original main has already completed model/posthealth/full4/source writes.
    publish_final(out/'report.json',report,out)


def ports():
    namespace={**vars(base.original),'__file__':__file__,'__name__':__name__,'PLAN':PLAN,'closure':closure,'phase_recipe':phase_recipe,'run_phase':run_phase,'contract':contract,'Protocol':Protocol,'life':life,'set_runtime':set_runtime,'final_total_budget':final_total_budget};nodes=[];counts={'main':0,'dirs':0,'argument':0,'intake':0}
    for name in('native_phase','main'):
        # Existing source-bound LPdisabled port functions, not changed frozen globals.
        original_node=copy.deepcopy(next(n for n in ast.parse(Path(base.original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==('run_phase'if name=='native_phase'else'main')))
        if name=='native_phase':
            original_node.name=name
            class Arm(ast.NodeTransformer):
                def visit_Expr(self,n):
                    call=n.value
                    if isinstance(call,ast.Call)and isinstance(call.func,ast.Attribute)and call.func.attr=='write_text'and isinstance(call.func.value,ast.BinOp)and getattr(call.func.value.right,'value',None)=='ARM':return None
                    return self.generic_visit(n)
            original_node=Arm().visit(original_node)
            for n in ast.walk(original_node):
                if isinstance(n,ast.Tuple)and [getattr(x,'value',None)for x in n.elts]==['captures','p30','ple-input','qsa']:n.elts.extend([ast.Constant('host-traces'),ast.Constant('unitrace')]);counts['dirs']+=1
        else:
            counts['main']+=1
            for n in ast.walk(original_node):
                if isinstance(n,ast.keyword)and n.arg=='choices'and isinstance(n.value,ast.Tuple)and [getattr(x,'value',None)for x in n.value.elts]==['off-equivalence','original-fidelity']:n.value.elts=[ast.Constant('host-latency')]
                if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='phases'for t in n.targets):n.value=ast.Tuple([ast.Constant(p)for p in PHASES],ast.Load())
                if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)and isinstance(n.test.left,ast.Attribute)and n.test.left.attr=='phase'and isinstance(n.test.comparators[0],ast.Constant):n.test.comparators[0].value='host-latency'
                if isinstance(n,ast.Constant)and n.value=='OFF_comparison':n.value='H36_OFF_ON_comparison'
            parse=next(n for n in ast.walk(original_node)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='a'for t in n.targets)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='parse_args')
            at=original_node.body.index(parse);original_node.body.insert(at,ast.parse("p.add_argument('--pti-intake',type=Path,required=True)").body[0]);original_node.body.insert(at+2,ast.parse('set_runtime(a.pti_intake)').body[0]);counts['argument']+=1;counts['intake']+=1
        if name=='main':
            anchor=next(i for i,n in enumerate(original_node.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='require'and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant)and n.value.args[1].value=='Actual constructor/model phase failed');original_node.body.insert(anchor,ast.parse('final_total_budget(out,report)').body[0])
        nodes.append(original_node)
    require(counts=={'main':1,'dirs':1,'argument':1,'intake':1},'Exact PTI lifecycle port anchors required');exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[explicit-PTI-port]','exec'),namespace);return namespace
_PORTS=ports()
if __name__=='__main__':_PORTS['main']()
