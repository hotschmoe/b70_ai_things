#!/usr/bin/env python3
"""Matched paired50 singleton staging OFF/ON; experiment-boundary identity.
Paired model execution remains deferred until the one-card staging run passes.
"""
import ast,copy
from pathlib import Path
from types import SimpleNamespace
import run_source43_warm_pti_v3 as original
from serial37_canonical_json_v3 import canonical
import source50_model_sdk_binding_v1 as fresh

def repeat_equivalence(*arms):
 original.require(len(arms)in(1,2)and all(len(arm)==2 and canonical(arm[0])==canonical(arm[1]) for arm in arms),'Warmup/selected repeat differs')
 if len(arms)==2:original.require(canonical(arms[0])==canonical(arms[1]),'Matched arms differ')
 return {'warmup_selected_equal':True,'matched_arms_equal':len(arms)==2,'scope':'native bounded outputs only; no original fullmath or speed promotion'}

_API=None;_API_SHA=None

def set_api(root,expected_sha):
 global _API,_API_SHA
 root=Path(root);original.require(root.is_absolute()and root.resolve()==root and original.c.sha(root/'prepared.json')==expected_sha,'Exact current174 prepared metadata bytes required');_API=root;_API_SHA=expected_sha

def closure():
 plan=original.HERE/'source50-pair24-mirror-t1-model-source-plan-v1.json'
 for name,want in original.c.read(plan)['files'].items():original.require(original.c.sha(original.ROOT/name)==want,'Exact paired50 staging source changed '+name)
 import c1_source43_api_controller_v3 as api
 api.source_binding();registry=api.registry_binding();original.require(registry['actual_models']==174 and registry['all173_complete_rows_preserved']is True and registry['old173_current_global_hash_gate_passed']is False,'Exact174 association preserving all173 required');return original.c.sha(plan)

def prerequisite(root):
 original.require(_API is not None,'Explicit current174 API prepared source required');m=original.c.read(_API/'prepared.json');normal=m['historical_prerequisites']['normal'];original.require(Path(root).resolve()==Path(normal['root']).resolve()and original.c.sha(_API/'prepared.json')==_API_SHA,'Only the authenticated named normal history is allowed')
 return {'scope':'original normal metadata association; full current API admission still required','root':normal['root'],'current_model_identity_qualified':False,'current_GPU_health_transferred':False,'API_prepared_root':str(_API),'API_prepared_sha256':_API_SHA}

def admission(plan):
 import c1_source43_api_controller_v3 as api
 original.require(_API is not None and original.c.sha(_API/'prepared.json')==_API_SHA,'Current prepared association changed');current=api.validate_prepared(_API);root=Path(current['historical_prerequisites']['normal']['root']);m=original.c.read(root/'prepared.snapshot.json');cfg=original.c.read(root/'config.snapshot.json')
 original.require(canonical(cfg)==canonical(current['original_template_config'])and canonical(plan['source43'])==canonical(current['source43_sdk']),'Actual current43 SDK/config association differs')
 original.require(canonical(original.c.read(Path(plan['template_path'])/'prepared.json'))==canonical(m)and original.c.sha(Path(plan['template_path'])/'prepared.json')==plan['template_prepared_sha256']and original.c.sha(Path(plan['template_path'])/'server-config.json')==plan['template_config_sha256']and canonical(original.c.read(Path(plan['template_path'])/'server-config.json'))==canonical(cfg),'Original CLI template bytes/config differ from authenticated normal snapshots')
 original.require(m['pack']==current['pack']and canonical(m['runtime'])==canonical(current['runtime'])and m['runtime_receipt']==current['runtime_receipt'],'Current actual43 runtime/pack differs')
 original.require(original.c.sha(plan['actual_model_identity_path'])==plan['actual_model_identity_sha256']and original.c.sha(plan['inputs_path'])==plan['inputs_sha256'],'Actual model identity/input receipt changed');proof=fresh.binding();return m,cfg

def preflight(plan,output):
 m=original.c.read(Path(plan['template_path'])/'prepared.json');cfg=original.c.read(Path(plan['template_path'])/'server-config.json');inputs=original.c.read(plan['inputs_path'])
 original.require(original.c.sha(Path(plan['template_path'])/'prepared.json')==plan['template_prepared_sha256']and original.c.sha(Path(plan['template_path'])/'server-config.json')==plan['template_config_sha256'],'Actual template byte pins required')
 commands=[]
 for phase in NS['PHASES']:
  descriptor,recipe=NS['phase_recipe'](phase,m,cfg,inputs,original.c.sha(NS['PLAN']));name,command=NS['life'].contract.native_command(Path(output)/phase,descriptor,recipe,NS['_CASE'][1],original.c.sha(NS['PLAN']))
  original.require(recipe['PTI_enabled'] is False and recipe['native_env']['STRATA_PF_GROUPED_F16']=='0' and recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0' and recipe['native_argv'][recipe['native_argv'].index('--short-read')+1]=='0'and recipe['native_argv'][recipe['native_argv'].index('--spec')+1]=='2'and recipe['native_argv'][recipe['native_argv'].index('--pipeline-windows')+1]=='0'and len(recipe['requests'])==2 and len(recipe['requests'][0]['ids'])==54 and all(r['max_new']==64 for r in recipe['requests'])and recipe['native_env']['STRATA_VERIFY_MIRROR_STAGE_T1']==str(NS['PHASES'].index(phase)),'Paired50 staging construction must retain groupedOFF/reserved-spec2/GEN64 geometry')
  argv=recipe['native_argv'];original.require('--native'in argv and argv[argv.index('--spec')+1]=='2'and int(argv[argv.index('--prefill')+1])>0 and not any(k in argv for k in('--mtp','--batch-mtp','--keep-canonical')),'Native IQ needs shard/native+reserved spec2+prefill with no drafter')
  commands.append({'phase':phase,'container_name':name,'native_command':command})
 return {'pure_command_construction_passed':True,'commands':commands,'GPU_launch_performed':False,'current_admission_still_required':True}


def stat_snapshot(shards):return {str(path):original.c.stat_signature(path) for path in shards}

def campaign_stats(report,current):
 original.require(report.get('identity_scope')=='whole_matched_arms' and report.get('post_full4',{}).get('passed') is True,'Completed campaign-level final identity required')
 baseline={row['path']:row['stat_after'] for row in report['pre_full4']['rows']}
 original.require(current==baseline and {row['path']:row['stat_after'] for row in report['post_full4']['rows']}==baseline,'Campaign initial/final/current stat identity differs')
 for arm in report['per_arm'].values():original.require(arm['pre_stat']==baseline and arm['post_stat']==baseline,'Inter-arm stat/inode/size/time mutation')
 return True

import run_source47_grouped_f16_model_v2 as boundary_base
def boundary_ports(namespace):return boundary_base.boundary_ports(namespace)

def set_sdk(root,receipt_sha):
 value=fresh.configure(root,receipt_sha);NS['intake'].SDK=fresh.SDK;return value

import run_source43_pair24_latency_v2 as pair_software
memory_admission=pair_software.memory_admission

def native_command_pair(directory,m,plan,pid,binding):
 import os
 name,command=native_command(directory,m,plan,pid,binding)
 for card in(0,1):
  for path in('/dev/dri/card'+str(card),'/dev/dri/renderD'+str(128+card)):
   gid=str(os.stat(path).st_gid)
   if gid not in[command[i+1]for i,x in enumerate(command[:-1])if x=='--group-add']:
    at=command.index(m['runtime']['image']);command[at:at]=['--group-add',gid]
 return name,command

def derive():
 namespace={**vars(original),'__file__':__file__,'PLAN':original.HERE/'source50-pair24-mirror-t1-model-source-plan-v1.json','PHASES':('source50-pair24-mirror-t1-OFF','source50-pair24-mirror-t1-ON'),'closure':closure,'admission':admission,'prerequisite':prerequisite,'preflight':preflight,'set_api':set_api,'set_sdk':set_sdk,'fresh':fresh,'memory_admission':memory_admission}
 namespace['contract']=SimpleNamespace(**{**vars(original.contract),'equivalence':repeat_equivalence})
 nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)]
 phase=next(n for n in nodes if n.name=='phase_recipe');count=0
 for n in ast.walk(phase):
  if isinstance(n,ast.keyword)and n.arg=='PTI_enabled':n.value=ast.Constant(False);count+=1
 original.require(count==1,'Sole native OFF phase predicate required')
 phase.body[-1:-1]=ast.parse("require('--short-read'not in recipe['native_argv'],'Original default short-read geometry required');recipe['native_argv'].extend(['--short-read','0']);recipe.update(short_read_control=0,only_native_argument_delta=['--short-read','0'],numeric_head_residual_observed=False)").body
 phase.body[-1:-1]=ast.parse("descriptor.update(source=str(fresh.SDK/'source'),executable=str(fresh.SDK/'build/strata'),alias='source50-mode0-mirror-t1-native-CLI-diagnostic');recipe['engine_generation']=50;recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']='0';recipe['native_env']['STRATA_PF_GROUPED_F16']='0';recipe['native_env']['STRATA_VERIFY_MIRROR_STAGE_T1']='1'if phase==PHASES[1]else'0';recipe['native_argv'][recipe['native_argv'].index('--spec')+1]='2';require(not any(k in recipe['native_argv']for k in('--mtp',)),'Serial noMTP/no-pipeline staging required');recipe['native_argv'].extend(['--pipeline-windows','0']);[r.update(max_new=64)for r in recipe['requests']];require(all(recipe['native_env'].get(k)in(None,'8')for k in('STRATA_GU_LANES','STRATA_DOWN_LANES')),'Default8 decode lanes retained');recipe.update(host_trace_enabled=False,both_arms_H36_enabled=False,capture_kind='fresh50 short54 twoGEN64 warmup selected; no profiler',maximum_total_output_tokens=128,bounded_window_schema=2,selected_request_count=2,graph_coverage_qualified=False,full_model_math_qualified=False)").body
 phase.body[-1:-1]=ast.parse("require(not any(k in recipe['native_argv']for k in('--layer-split','--split-device','--trim-stage-weights')),'No inherited topology flags');recipe['native_argv'].extend(['--layer-split','24','--split-device','1','--trim-stage-weights']);recipe['native_env']['ZE_AFFINITY_MASK']='0,1';descriptor.update(cards=[0,1],alias='source50-mode0-pair24-mirror-t1-native-CLI-diagnostic');recipe.update(cards=[0,1],stages=[[0,0,24],[1,24,48]],paired_model_runtime_qualified=False,both_device_headroom_qualified=False)").body
 # The run-phase's unchanged graph proof executes with the actual new reader.
 class ReaderName(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='validate_source43_warm_pti_v3':n.module='validate_source50_pair24_mirror_t1_model_v1'
   return n
 nodes=[ReaderName().visit(n)for n in nodes if n.name not in('closure','admission','prerequisite')]
 port=next(n for n in nodes if n.name=='ports')
 class ParentDelta(ast.NodeTransformer):
  def visit_Constant(self,n):
   if type(n.value)is str and "p.add_argument('--pti-intake'"in n.value:n.value=n.value.replace("    a = p.parse_args()","    p.add_argument('--sdk',type=Path,required=True)\n    p.add_argument('--sdk-receipt-sha256',required=True)\n    p.add_argument('--api-prepared',type=Path,required=True)\n    p.add_argument('--api-prepared-sha256',required=True)\n    a = p.parse_args()\n    set_sdk(a.sdk,a.sdk_receipt_sha256)\n    set_api(a.api_prepared,a.api_prepared_sha256)")
   if type(n.value)is str and "report['OFF_prerequisite']=prior"in n.value:n.value=n.value.replace("    phases = PHASES","    report['SDK50_root']=str(a.sdk)\n    report['SDK50_receipt_sha256']=a.sdk_receipt_sha256\n    report['API_prepared_root']=str(a.api_prepared)\n    report['API_prepared_sha256']=a.api_prepared_sha256\n    report['command_preflight']=command_preflight\n    phases = PHASES")
   return n
 nodes=[ParentDelta().visit(n)for n in nodes]
 actual=next(n for n in nodes if n.name=='run_phase');actual.body=ast.parse("out,phase=args[:2];watch=Watcher(out,lambda:os.kill(os.getpid(),signal.SIGUSR1));result=None\ntry:\n watch.start();result=_PORTS['native_phase'](*args,**kwargs)\nfinally:\n proof=watch.close()\n if (out/phase).is_dir():proof=publish_final((out/phase)/'raw-budget.json',proof,out)\nrequire(proof['passed'],'Actual raw quota failed');fresh.binding();return result").body
 insert=next(i for i,n in enumerate(port.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='result'for t in n.targets))
 port.body[insert:insert]=ast.parse("source=source.replace('metadata, cfg = admission(plan)','command_preflight = preflight(plan, a.output)\\n    metadata, cfg = admission(plan)')").body
 exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[single-OFF-short-read0]', 'exec'),namespace)
 pair_node=copy.deepcopy(next(n for n in ast.parse(Path(__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='native_command_pair'));exec(compile(ast.fix_missing_locations(ast.Module([pair_node],type_ignores=[])),__file__+'[paired-render-groups]','exec'),namespace)
 namespace['intake']=SimpleNamespace(**{**vars(original.intake),'SDK':fresh.SDK});namespace['life']=SimpleNamespace(**{**vars(original.life),'prepared_ctrl':SimpleNamespace(**{**vars(original.life.prepared_ctrl),'memory_admission':memory_admission}),'contract':SimpleNamespace(**{**vars(original.life.contract),'native_command':namespace['native_command_pair']})});namespace['_PORTS']=boundary_ports(namespace);return namespace
NS=derive()
for name in('PLAN','PHASES','contract','closure','set_case','set_runtime','admission','prerequisite','phase_recipe','native_command','life','warm_control','mapped','native_pid','run_phase','ports'):
 globals()[name]=NS[name]
def __getattr__(name):
 if name in NS:return NS[name]
 raise AttributeError(name)
if __name__=='__main__':NS['_PORTS']['main']()
