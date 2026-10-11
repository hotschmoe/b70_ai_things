#!/usr/bin/env python3
"""Actual43 cold four GEN4 scope; sampled query metadata and unchanged raw operations."""
import ast,copy,os,signal,time
from pathlib import Path
from types import SimpleNamespace
import run_source43_normal_model_cli_v1 as base
import source43_sampled_pti_runtime_v1 as fresh_tool
import pti_bounded_gen4_contract_v1 as bounded
import pti_warm_gen4_contract_v1 as warm
import pti_owned_temporal_control_v2 as temporal
import pti_warm_owned_exec_v2 as temporal_exec
import pti_temporal_fixture_binding_v2 as fixture
from types import SimpleNamespace
tool=SimpleNamespace(**vars(fresh_tool))
import source41_lpdisabled_contract_v1 as omission
import collect_source41_lpdisabled_host_v1 as collector
import native_source41_cli_protocol_v4 as protected
from source41_pti_raw_budget_v2 import Watcher,TOTAL,INTERVAL,publish_final
from serial37_canonical_json_v3 import canonical
import pti_sampled_census_exec_v1 as census
ROOT=base.ROOT;HERE=base.HERE;c=base.c;require=base.require;intake=base.intake
PLAN=HERE/'source43-warm-pti-source-plan-v2.json';PHASES=('source43-warm-pti-OFF','source43-warm-pti-ON');_RUNTIME=None;_CASE=None
class Protocol(protected.Protocol):request=omission._REQUEST['request']
contract=SimpleNamespace(**{**vars(base.contract),'requests':warm.requests,'equivalence':warm.equivalence})
def set_case(role,pid=None,start_ticks=None):
    global _CASE
    require(role=='short','Long524 scope held after actual normalization cap failure; only useful short54 warm capture permitted')
    if pid is None:pid=os.getpid();start_ticks=__import__('native_qsa40_owned_launch_v9').identity(pid)['start_ticks']
    require(type(pid)is int and pid>0 and type(start_ticks)is int and start_ticks>0,'Original producer identity required')
    _CASE=(role,pid,start_ticks)

def closure():
    for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Exact43 PTI source changed '+name)
    fixture.binding();base.closure();return c.sha(PLAN)
def set_runtime(path,expected_intake_sha256):
    global _RUNTIME,tool
    _RUNTIME=fresh_tool.binding(path,expected_intake_sha256)
    tool=SimpleNamespace(**{**vars(fresh_tool),'LIB_SHA':_RUNTIME['injection']['sha256'],'TOOL_SHA':_RUNTIME['tool']['sha256'],'binding':lambda p:fresh_tool.binding(p,expected_intake_sha256)})
def admission(plan):return base.admission(plan)
def prerequisite(root):
    from validate_source43_normal_model_cli_v1 import finalized_binding
    result=finalized_binding(root);require(result['source_generation']==43 and result['materialized_mode']==0 and result['OFF_equivalence_passed']is True,'Fresh43 normal model control required');return result

def phase_recipe(phase,metadata,cfg,inputs,binding):
    require(phase in PHASES and _RUNTIME is not None,'Exact actual43 PTI phase/runtime required');tool.inherited_env(cfg['env']);require(_RUNTIME['image']==metadata['runtime']['image'],'Actual same-image profiler required')
    descriptor,recipe=base.phase_recipe('source43-off',metadata,cfg,inputs,binding);recipe.update(mode=phase,PTI_enabled=phase==PHASES[1],logprobs_policy='omitted native default-1; seed omitted; greedytemperature0',both_arms_H36_enabled=True,total_raw_cap_bytes=TOTAL,total_raw_sample_seconds=INTERVAL,metadata_cap_bytes=512<<20,tool_runtime=_RUNTIME,H36_export_IO_outside_request_span=True)
    recipe['native_env'].update(STRATA_DECODE_TIMING='1',STRATA_CRITICAL_PATH_TRACE='1',STRATA_CRITICAL_PATH_TRACE_DIR='/results/host-traces',STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256=binding)
    if recipe['PTI_enabled']:
        recipe['native_env'].update(tool.CLOCK_ENV);recipe['native_env'].update(tool.NATIVE_ENV);recipe['native_argv']=['/pti/bin/unitrace',*tool.OPTIONS,*recipe['native_argv']]
    recipe=bounded.construct(recipe,inputs)
    require(_CASE is not None,'Actual original warm-case owner required');role,pid,ticks=_CASE
    template=next(r for r in recipe['requests']if r['role']==role)
    recipe['requests']=[{**template,'ordinal':1,'window_role':'warmup'},{**template,'ordinal':2,'window_role':'selected'}]
    recipe.update(capture_kind='one same-process paused warmup plus selected GEN4',maximum_total_output_tokens=8,role_input=role,temporal_session=temporal.session_name(pid,ticks,binding)if recipe['PTI_enabled']else None)
    if recipe['PTI_enabled']:
        recipe['native_env'].update(UNITRACE_StartPaused='1',UNITRACE_Session=recipe['temporal_session'])
        recipe['native_argv'][1:1]=['--start-paused','--session',recipe['temporal_session']]
    return descriptor,recipe

def native_command(directory,m,recipe,pid,binding):
    name,cmd=base.life.contract.native_command(directory,m,recipe,pid,binding)
    if recipe['PTI_enabled']:
        at=cmd.index(m['runtime']['image']);cmd[at:at]=['-v',str(HERE/'pti_owned_temporal_control_v2.py')+':/pti-temporal.py:ro'];at=cmd.index(m['runtime']['image']);cmd[at:at]=['-v',_RUNTIME['install_root']+':/pti:ro','-v',str(HERE/'pti_mapped_injection_census_v2.py')+':/pti-census.py:ro']
    return name,cmd
life=SimpleNamespace(**{**vars(base.life),'contract':SimpleNamespace(**{**vars(base.life.contract),'native_command':native_command})})

def warm_control(directory,name,cmd,m,binding,recipe,phase):
    if not recipe['PTI_enabled']:return None
    result=temporal_exec.run(directory,name,cmd,m['runtime']['image'],binding,phase,tool.TOOL_SHA,recipe['temporal_session']);c.write(directory/(phase+'-owned.json'),result);return result

def mapped(directory,name,cmd,image,binding,phase):
    if '/pti/bin/unitrace'not in cmd[-1]:return None
    receipt=census.run(directory,name,cmd,image,binding,phase,tool.LIB_SHA);c.write(directory/(phase+'-owned.json'),receipt);return receipt

def native_pid(terminal,command,pti):
    if not pti:return collector.native_namespace(terminal,command)
    import run_source41_pti_latency_v3 as old
    # Exact same PID1 launcher rule, but explicitly add the new versioned role
    # to its only permitted UNITRACE keys. Frozen module globals stay unchanged.
    node=copy.deepcopy(next(n for n in ast.parse(Path(old.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_pid'));require(_CASE is not None,'Original warm case required');session=temporal.session_name(_CASE[1],_CASE[2],c.sha(PLAN));view=SimpleNamespace(**{**vars(tool),'CLOCK_ENV':{**tool.CLOCK_ENV,'UNITRACE_StartPaused':'1','UNITRACE_Session':session}});ns={**vars(old),'tool':view};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[actual43-finalizer-PID1]','exec'),ns);pid=ns['native_pid'](terminal,command,True)
    env=terminal['Config']['Env'];require('STRATA_DIAGNOSTIC_PTI_FINALIZE=1'in env,'Actual native explicitfinalizer flag missing');return pid

def run_phase(*args,**kwargs):
    out,phase=args[:2];directory=out/phase;watch=Watcher(out,lambda:os.kill(os.getpid(),signal.SIGUSR1));error=None;result=None
    try:
        watch.start();result=_PORTS['native_phase'](*args,**kwargs);recipe=c.read(directory/'recipe.json');pid=native_pid(result['terminal'],c.read(directory/'command.json'),recipe['PTI_enabled']);c.write(directory/'native-namespace.json',{'native_pid':pid,'PTI_enabled':recipe['PTI_enabled'],'exec_branch_source_checked':True})
        rows=[row['raw']for row in c.read(directory/'requests.json')];files=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]));coverage=collector.collect(files,directory/'engine.combined.log',args[5],rows,[[0,0,48]],pid);c.write(directory/'host-coverage.json',coverage);from validate_source43_warm_pti_v2 import graph_binding,control_binding,clock_window;c.write(directory/'warm-graph-proof.json',graph_binding(directory))
        if recipe['PTI_enabled']:
            control_binding(directory,recipe,result['terminal'],{'source_plan_sha256':args[5],'producer_pid':os.getpid(),'producer_start_ticks':__import__('native_qsa40_owned_launch_v9').identity(os.getpid())['start_ticks']});c.write(directory/'warm-clock-window.json',clock_window(directory))
        (directory/'leaf.stderr').write_bytes((directory/'engine.combined.log').read_bytes());require(canonical(tool.binding(_RUNTIME['intake_path']))==canonical(_RUNTIME),'Actual profiler runtime changed during arm')
    except BaseException as exc:error=exc;watch.latched=True;watch.errors.append('owned phase/start: '+type(exc).__name__+': '+str(exc))
    finally:
        proof=watch.close()
        if directory.is_dir():proof=publish_final(directory/'raw-budget.json',proof,out)
    if error is not None:raise error
    require(proof['passed'],'Raw postwrite observed budget failed; original native already retired');return result

def ports():
    nodes=[copy.deepcopy(n)for n in ast.parse(Path(base.original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('run_phase','main')]
    native=next(n for n in nodes if n.name=='run_phase');native.name='native_phase'
    class RemoveArm(ast.NodeTransformer):
        def visit_Expr(self,n):
            call=n.value
            if isinstance(call,ast.Call)and isinstance(call.func,ast.Attribute)and call.func.attr=='write_text'and isinstance(call.func.value,ast.BinOp)and getattr(call.func.value.right,'value',None)=='ARM':return None
            return self.generic_visit(n)
    native=RemoveArm().visit(native);nodes[0]=native
    for n in ast.walk(native):
        if isinstance(n,ast.Tuple)and [getattr(x,'value',None)for x in n.elts]==['captures','p30','ple-input','qsa']:n.elts.extend([ast.Constant('host-traces'),ast.Constant('unitrace')])
    actual_try=next(n for n in native.body if isinstance(n,ast.Try));loop=next(n for n in actual_try.body if isinstance(n,ast.For));at=actual_try.body.index(loop)
    before_nodes=ast.parse("c.write(directory/'ready-observed.json',{'monotonic_ns':time.monotonic_ns(),'epoch':time.time(),'process_started_epoch':protocol.started_epoch});mapped(directory,name,cmd,m['runtime']['image'],binding,'after_READY')").body;actual_try.body[at:at]=before_nodes;after_nodes=ast.parse("c.write(directory/'last-DONE-observed.json',{'monotonic_ns':time.monotonic_ns(),'epoch':time.time()});mapped(directory,name,cmd,m['runtime']['image'],binding,'after_last_DONE_before_QUIT')").body;actual_try.body[at+len(before_nodes)+1:at+len(before_nodes)+1]=after_nodes
    loop=next(n for n in actual_try.body if isinstance(n,ast.For))
    at_loop=actual_try.body.index(loop);actual_try.body[at_loop:at_loop]=ast.parse("warm_control(directory,name,cmd,m,binding,plan,'temporal-before-warmup')").body
    loop.body[:0]=ast.parse("if row['ordinal']==2:warm_control(directory,name,cmd,m,binding,plan,'temporal-before-selected')").body
    loop.body.extend(ast.parse("c.write(directory/('warmup-DONE-observed.json' if row['ordinal']==1 else 'selected-DONE-observed.json'),{'monotonic_ns':time.monotonic_ns(),'epoch':time.time()})").body)
    at_loop=actual_try.body.index(loop);actual_try.body[at_loop+1:at_loop+1]=ast.parse("warm_control(directory,name,cmd,m,binding,plan,'temporal-after-selected')").body
    main=next(n for n in nodes if n.name=='main');obsolete=next(n for n in main.body if isinstance(n,ast.If)and ast.unparse(n.test)=="a.phase == 'original-fidelity'");main.body.remove(obsolete)
    source=ast.unparse(ast.Module(nodes,type_ignores=[])).replace("choices=('off-equivalence', 'original-fidelity')","choices=('host-latency',)").replace("phases = ('source40-control', 'source41-off') if a.phase == 'off-equivalence' else ('source41-original-fidelity',)","phases = PHASES").replace("a.phase == 'off-equivalence'","a.phase == 'host-latency'").replace("'OFF_comparison'","'H36_OFF_ON_comparison'")
    source=source.replace('a = p.parse_args()',"p.add_argument('--pti-intake',type=Path,required=True)\n    p.add_argument('--pti-intake-sha256',required=True)\n    p.add_argument('--prompt-role',choices=('short',),required=True)\n    a = p.parse_args()\n    set_case(a.prompt_role)\n    set_runtime(a.pti_intake,a.pti_intake_sha256)\n    require(a.off_receipt is not None,'Fresh43 normal model prerequisite required')\n    prior=prerequisite(a.off_receipt)")
    source=source.replace("phases = PHASES","report['role_input']=a.prompt_role\n    report['OFF_prerequisite']=prior\n    report['OFF_prerequisite_root']=str(a.off_receipt.resolve())\n    phases = PHASES")
    source=source.replace("require(report['passed'], 'Actual constructor/model phase failed')","publish_final(out/'report.json',report,out)\n    require(report['passed'], 'Actual constructor/model phase failed')")
    result={**vars(base.original),**vars(base),**globals(),'__file__':__file__};exec(compile(source,'[actual43-LPdisabled-H36-PTI-parent]','exec'),result);return result
_PORTS=ports()
if __name__=='__main__':_PORTS['main']()
