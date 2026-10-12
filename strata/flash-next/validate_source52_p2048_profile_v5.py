"""Independent narrow V5 producer admission; no H36/CPU/calibration claims."""
import argparse,ast,copy,json,re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import run_source52_p2048_profile_v5 as actor
from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
import validate_source43_bounded_pti_v1 as pti_reader
import validate_source43_warm_pti_v3 as warm_reader
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require;HERE=actor.HERE;ROOT=actor.ROOT
PLAN=HERE/'source52-p2048-profile-reader-v5-source-plan-v1.json'

def source_binding():
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Producer reader source changed '+name)
 return c.sha(PLAN)

@contextmanager
def configured_actor(report,recipe,module=actor):
 p=module.parent;old=p.NS;old_gen=module.GEN
 try:
  require(recipe['requests'][0]['max_new']==8,'Actual public scope is GEN8 only');module.GEN=8;p.NS=module.NS;p.configure(recipe['natural_settings']);p.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);ns=module.NS;ns['fresh']=p.fresh
  ns['set_api'](report['API_prepared_root'],report['API_prepared_sha256']);ns['set_case'](report['role_input'],report['producer_pid'],report['producer_start_ticks']);saved=recipe['tool_runtime'];ns['set_runtime'](saved['intake_path'],saved['exact_intake_sha256']);ns['_PORTS']=p.base.boundary_ports(ns)
  active=SimpleNamespace(**{**vars(module),**ns});require(active.__file__==module.__file__ and active.fresh.SDK is not None and active.contract.requests is module.two.requests,'Exact configured private actor/GEN8 contract required')
  yield active
 finally:p.NS=old;module.GEN=old_gen

def inherited(function,module,active,*args):
 node=copy.deepcopy(next(n for n in ast.parse(Path(module.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==function));ns={**vars(module),'actor':active};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[unchanged-'+function+']','exec'),ns);return ns[function](*args)

def selected_clock_boundaries(directory):
 controls=[c.read(directory/(x+'.json'))for x in('temporal-before-selected','temporal-after-selected')]
 path=directory/'unitrace/clock.1.jsonl';require(path.is_file()and not path.is_symlink()and 0<path.stat().st_size<=512<<20,'Exact bounded actual clock file required');before=c.sha(path);limit=controls[0]['clock_before']['last_sequence'];selected_lo=controls[0]['clock_after']['last_sequence'];selected_hi=controls[1]['clock_before']['last_sequence'];require(0<limit<=selected_lo<selected_hi,'Ordered complete actual selected sequence range required');seq=0;selected=0
 with path.open('rb')as stream:
  for raw in stream:
   require(raw.endswith(b'\n')and len(raw)<=4096,'Complete bounded clock record required');r=json.loads(raw);seq+=1;require(type(r['sequence'])is int and r['sequence']==seq and r['pid']==1,'Actual clock PID/sequence differs')
   if r['kind']=='instance_normalization':require(seq>limit,'Paused warmup normalization is not admitted');selected+=int(selected_lo<seq<=selected_hi)
  for control in controls:
   for key in('clock_before','clock_after'):
    e=control[key];expected=bytes.fromhex(e['tail_hex']);require(e['path']=='/results/unitrace/clock.1.jsonl'and 0<=e['tail_offset']<e['bytes']<=path.stat().st_size,'Actual saved clock byte boundary differs');stream.seek(e['tail_offset']);observed=stream.read(len(expected));require(observed==expected and __import__('hashlib').sha256(observed).hexdigest()==e['tail_sha256'],'Original immutable clock prefix changed');last=json.loads(bytes.fromhex(e['last_line_hex']));require(last['sequence']==e['last_sequence']and last['pid']==1 and expected.splitlines()[-1]==bytes.fromhex(e['last_line_hex']),'Actual endpoint sequence association differs')
 require(seq>=selected_hi and selected>0 and c.sha(path)==before,'Current complete selected clock prefix absent/changed');return {'paused_warmup_normalizations':0,'selected_sequence_bounds':[selected_lo,selected_hi],'selected_observed_normalizations':selected,'clock_sha256':before,'clock_alignment_qualified':False,'model_operation_coverage_qualified':False}

def arm_binding(root,report,directory,recipe,active):
 require(recipe['engine_generation']==52 and recipe['cards']==[0,1]and recipe['stages']==[[0,0,24],[1,24,48]]and recipe['PTI_enabled']is True,'Actual52 paired PTI recorder required');env=recipe['native_env'];argv=recipe['native_argv'];require(env['STRATA_CRITICAL_PATH_TRACE']=='0'and recipe['host_trace_enabled']is False and recipe['both_arms_H36_enabled']is False and recipe['Chrome_CPU_host_lane_recorded']is False,'Disabled host lanes must stay disabled');require('--chrome-device-logging'in argv and '--chrome-call-logging'not in argv and '--chrome-kernel-logging'not in argv,'Exact device-only recorder required');require(not list((directory/'host-traces').glob('*')),'Disabled H36 emitted data');rows=c.read(directory/'requests.json');views=active.contract.requests(rows,recipe['requests']);require(len(rows)==2 and all(len(x['raw']['ids'])==2048 for x in rows),'Two actual P2048 requests required')
 for row,want in zip(rows,recipe['requests']):
  raw=row['raw'];require(canonical(raw)==canonical(c.read(directory/('request-%d.json'%row['ordinal']))),'Individual native record differs');done=raw['done'].split();require(done[5]!='length'or len(raw['output_ids'])==8,'GEN8 length must exhaust cap');require(done[5]!='stop'or raw['output_ids'][-1]in(248044,248046),'Stop must be actual EOS');require(all(t not in(248044,248046)for t in raw['output_ids'][:-1]),'EOS must terminate output')
 terminal=c.read(directory/'result.json')['terminal'];require(active.native_pid(terminal,c.read(directory/'command.json'),True)==1,'Actual owned direct native PID1 required');require(re.findall(r'^B70_PTI_MODEL_FINALIZE_V1 pid=(\d+) result=(\d+)$',(directory/'engine.combined.log').read_text(),re.M)==[('1','0')],'Exactly one successful native PTI finalizer required')
 pti_reader.old.budget(directory,root)
 census=inherited('census_binding',pti_reader,active,directory,recipe,terminal,rows);controls=inherited('control_binding',warm_reader,active,directory,recipe,terminal,report);clock=selected_clock_boundaries(directory)
 require(canonical(active._RUNTIME)==canonical(recipe['tool_runtime']),'Exact current PTI runtime envelope differs');asset=recipe['expert_profile_binding'];require(actor.parent.profile_binding(asset['path'],asset['sha256'])==asset,'Actual original profile changed')
 return {'phase':directory.name,'requests':views,'mapped_injection':census,'owned_temporal_controls':len(controls),'clock_boundary':clock,'copy_engine':env['STRATA_COPY_ENGINE'],'H36_coverage_qualified':False,'Chrome_CPU_host_lane_recorded':False}

def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();report=c.read(root/'report.json');report_sha=c.sha(root/'report.json');before={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'};require(before==report['artifacts'],'Original complete artifact roster differs');first=c.read(root/actor.parent.PHASES[0]/'recipe.json')
 try:
  with configured_actor(report,first)as active:
   result=boundary_reader_ports(active)['finalized_binding'](root);arms=[];views=[]
   for index,phase in enumerate(active.PHASES):
    directory=root/phase;recipe=c.read(directory/'recipe.json');require(recipe['native_env']['STRATA_COPY_ENGINE']==str(index),'Actual matched copy0/1 route differs');arm=arm_binding(root,report,directory,recipe,active);arms.append(arm);views.append(arm['requests'])
   comparison=active.contract.equivalence(*views);require(canonical(comparison)==canonical(report['H36_OFF_ON_comparison']),'Original GEN8 output comparison differs');require(actor.closure()==report['source_plan_sha256'],'Original producer closure changed');proof=active.fresh.binding();result.update(producer_admitted=True,reader_source_plan_sha256=source,producer_report_sha256=report_sha,actual52_SDK=proof,source_generation=52,native_processes=2,requests_per_process=2,GEN_cap=8,arms=arms,matched_copy0_1_outputs=True,H36_OFF_ON_equivalence_passed=False,H36_coverage_qualified=False,Chrome_CPU_host_lane_recorded=False,clock_alignment_qualified=False,complete_model_operation_coverage_qualified=False,physical_idle_qualified=False,physical_bus_bandwidth_qualified=False,clean_serving_speed_qualified=False,full_model_math_qualified=False,API_qualified=False,shelf_qualified=False,inspector_execution_allowed=True);return result
 finally:require(source_binding()==source and c.sha(root/'report.json')==report_sha and before=={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'},'Closing source/producer/raw roster changed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists()and a.root.resolve()not in a.output.resolve().parents,'New external read-only proof output required');c.write(a.output,finalized_binding(a.root))
