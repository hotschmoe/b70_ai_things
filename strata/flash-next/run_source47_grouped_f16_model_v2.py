#!/usr/bin/env python3
"""Safe47 OFF-only CPU construction; reusable experiment-boundary identity port.
No unchanged grouped47 ON retry or additional baseline GPU run requested.
"""
import ast,copy
from pathlib import Path
from types import SimpleNamespace
import run_source43_warm_pti_v3 as original
from serial37_canonical_json_v3 import canonical
import source47_grouped_model_binding_v1 as fresh

def repeat_equivalence(*arms):
 original.require(len(arms)in(1,2)and all(len(arm)==2 and canonical(arm[0])==canonical(arm[1]) for arm in arms),'Warmup/selected repeat differs')
 if len(arms)==2:original.require(canonical(arms[0])==canonical(arms[1]),'Matched arms differ')
 return {'warmup_selected_equal':True,'matched_arms_equal':len(arms)==2,'scope':'native bounded outputs only; no original fullmath or speed promotion'}

_API=None;_API_SHA=None

def set_api(root,expected_sha):
 global _API,_API_SHA
 root=Path(root);original.require(root.is_absolute()and root.resolve()==root and original.c.sha(root/'prepared.json')==expected_sha,'Exact current174 prepared metadata bytes required');_API=root;_API_SHA=expected_sha

def closure():
 plan=original.HERE/'source47-grouped-f16-model-source-plan-v2.json'
 for name,want in original.c.read(plan)['files'].items():original.require(original.c.sha(original.ROOT/name)==want,'Exact shortread2 source changed '+name)
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
  original.require(recipe['PTI_enabled'] is False and recipe['native_env']['STRATA_PF_GROUPED_F16']=='0' and recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0' and recipe['native_argv'][-2:]==['--short-read','0']and len(recipe['requests'])==2 and len(recipe['requests'][0]['ids'])==524,'Safe47 construction must retain OFF flags and actual geometry')
  commands.append({'phase':phase,'container_name':name,'native_command':command})
 return {'pure_command_construction_passed':True,'commands':commands,'GPU_launch_performed':False,'current_admission_still_required':True}


def stat_snapshot(shards):return {str(path):original.c.stat_signature(path) for path in shards}

def campaign_stats(report,current):
 original.require(report.get('identity_scope')=='whole_matched_arms' and report.get('post_full4',{}).get('passed') is True,'Completed campaign-level final identity required')
 baseline={row['path']:row['stat_after'] for row in report['pre_full4']['rows']}
 original.require(current==baseline and {row['path']:row['stat_after'] for row in report['post_full4']['rows']}==baseline,'Campaign initial/final/current stat identity differs')
 for arm in report['per_arm'].values():original.require(arm['pre_stat']==baseline and arm['post_stat']==baseline,'Inter-arm stat/inode/size/time mutation')
 return True

def boundary_ports(namespace):
 # Capture only this private generated source; frozen original globals are untouched.
 import builtins
 captured=[]
 def compiler(source,*args,**kwargs):captured.append(source);return builtins.compile(source,*args,**kwargs)
 namespace['compile']=compiler;ports=namespace['ports']();namespace.pop('compile')
 tree=ast.parse(captured[-1]);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
 attempt=next(n for n in main.body if isinstance(n,ast.Try));loop=next(n for n in attempt.body if isinstance(n,ast.For)and ast.unparse(n.target)=='phase')
 at=attempt.body.index(loop)
 attempt.body[at:at]=ast.parse("report['identity_scope']='whole_matched_arms';report['schema']=2\nreport['pre_pages']=intake.guard(shards,out/'guardian','campaign-pre')\nreport['pre_full4']=life.q.upload_v2.full_buffered_identity(HERE/'model-lock.json',lock,shards,out/'pre-full4.json',time.time())\nrequire(report['pre_full4']['passed'],'Fresh complete campaign pre-scan failed')\nreport['pre_after_pages']=intake.guard(shards,out/'guardian','campaign-pre-after')\nbaseline={row['path']:row['stat_after'] for row in report['pre_full4']['rows']}").body
 first_health=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='h' for t in n.targets))
 loop.body[2:first_health]=ast.parse("arm['pre_pages']=intake.guard(shards,out/'guardian',phase+'-pre')\narm['pre_stat']=stat_snapshot(shards);require(arm['pre_stat']==baseline,'Pre-arm model stat changed')\narm['pre_source_binding']=fresh.binding();require(closure()==source,'Pre-arm source closure changed')").body
 inner=next(n for n in loop.body if isinstance(n,ast.Try));post=list(inner.finalbody)
 # Preserve terminal, health and journal, replacing only the full payload scan.
 index=next(i for i,n in enumerate(post)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='after' for t in n.targets))
 inner.finalbody=post[:index]+ast.parse("arm['post_pages']=intake.guard(shards,out/'guardian',phase+'-post')\narm['post_stat']=stat_snapshot(shards);require(arm['post_stat']==baseline,'Post-arm model stat changed')\narm['post_source_binding']=fresh.binding();require(closure()==source,'Post-arm source closure changed')").body
 outer=next(n for n in attempt.finalbody if isinstance(n,ast.If))
 replacement=ast.parse("if not report['errors']:\n last=report['per_arm'][phases[-1]];report['post_health']=last['post_health'];report['aggregate_post_health_label']=phases[-1]+'-post'\nelse:\n try:report['post_health']=life.health(out,'post');life.journal(out,'post',started);report['aggregate_post_health_label']='post'\n except BaseException as exc:report['errors'].append('post-health: '+str(exc))\ntry:\n report['post_before_pages']=intake.guard(shards,out/'guardian','campaign-post-before')\n after=max(report['actual_model_terminal_epoch'],report.get('post_health',{}).get('finished_epoch',time.time()))\n report['post_full4']=life.q.upload_v2.full_buffered_identity(HERE/'model-lock.json',lock,shards,out/'post-full4.json',after)\n report['post_pages']=intake.guard(shards,out/'guardian','campaign-post')\n require(report['post_full4']['passed'],'Fresh complete campaign post-scan failed')\n campaign_stats(report,stat_snapshot(shards))\nexcept BaseException as exc:report['errors'].append('post-source: '+str(exc))").body
 at=attempt.finalbody.index(outer);attempt.finalbody[at:at+1]=replacement
 ports.update(stat_snapshot=stat_snapshot,campaign_stats=campaign_stats,fresh=namespace['fresh'])
 exec(compile(ast.fix_missing_locations(tree),__file__+'[campaign-boundary-identity]','exec'),ports)
 return ports

def derive():
 namespace={**vars(original),'__file__':__file__,'PLAN':original.HERE/'source47-grouped-f16-model-source-plan-v2.json','PHASES':('source47-grouped-f16-OFF',),'closure':closure,'admission':admission,'prerequisite':prerequisite,'preflight':preflight,'set_api':set_api,'fresh':fresh}
 namespace['contract']=SimpleNamespace(**{**vars(original.contract),'equivalence':repeat_equivalence})
 nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)]
 phase=next(n for n in nodes if n.name=='phase_recipe');count=0
 for n in ast.walk(phase):
  if isinstance(n,ast.keyword)and n.arg=='PTI_enabled':n.value=ast.Constant(False);count+=1
 original.require(count==1,'Sole native OFF phase predicate required')
 phase.body[-1:-1]=ast.parse("require('--short-read'not in recipe['native_argv'],'Original default short-read geometry required');recipe['native_argv'].extend(['--short-read','0']);recipe.update(short_read_control=0,only_native_argument_delta=['--short-read','0'],numeric_head_residual_observed=False)").body
 phase.body[-1:-1]=ast.parse("descriptor.update(source=str(fresh.SDK/'source'),executable=str(fresh.SDK/'build/strata'),alias='source47-mode0-grouped-f16-native-CLI-diagnostic');recipe['engine_generation']=47;recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']='0';recipe['native_env']['STRATA_PF_GROUPED_F16']='0';require(all(recipe['native_env'].get(k)in(None,'8')for k in('STRATA_GU_LANES','STRATA_DOWN_LANES')),'Default8 decode lanes retained');recipe.update(host_trace_enabled=False,both_arms_H36_enabled=False,capture_kind='fresh47 long524 twoGEN4 warmup selected; no profiler',graph_coverage_qualified=False,full_model_math_qualified=False)").body
 # The run-phase's unchanged graph proof executes with the actual new reader.
 class ReaderName(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='validate_source43_warm_pti_v3':n.module='validate_source47_grouped_f16_model_v1'
   return n
 nodes=[ReaderName().visit(n)for n in nodes if n.name not in('closure','admission','prerequisite')]
 case=next(n for n in nodes if n.name=='set_case')
 for x in ast.walk(case):
  if isinstance(x,ast.Constant)and x.value=='short':x.value='prefill_about512'
 recipe_node=next(n for n in nodes if n.name=='phase_recipe')
 for x in ast.walk(recipe_node):
  if isinstance(x,ast.Constant)and x.value=='short':x.value='prefill_about512'
 port=next(n for n in nodes if n.name=='ports')
 class ParentDelta(ast.NodeTransformer):
  def visit_Constant(self,n):
   if type(n.value)is str and "p.add_argument('--pti-intake'"in n.value:n.value=n.value.replace("choices=('short',)","choices=('prefill_about512',)").replace("    a = p.parse_args()","    p.add_argument('--api-prepared',type=Path,required=True)\n    p.add_argument('--api-prepared-sha256',required=True)\n    a = p.parse_args()\n    set_api(a.api_prepared,a.api_prepared_sha256)")
   if type(n.value)is str and "report['OFF_prerequisite']=prior"in n.value:n.value=n.value.replace("    phases = PHASES","    report['API_prepared_root']=str(a.api_prepared)\n    report['API_prepared_sha256']=a.api_prepared_sha256\n    report['command_preflight']=command_preflight\n    phases = PHASES")
   return n
 nodes=[ParentDelta().visit(n)for n in nodes]
 actual=next(n for n in nodes if n.name=='run_phase');actual.body=ast.parse("out,phase=args[:2];watch=Watcher(out,lambda:os.kill(os.getpid(),signal.SIGUSR1));result=None\ntry:\n watch.start();result=_PORTS['native_phase'](*args,**kwargs)\nfinally:\n proof=watch.close()\n if (out/phase).is_dir():proof=publish_final((out/phase)/'raw-budget.json',proof,out)\nrequire(proof['passed'],'Actual raw quota failed');fresh.binding();return result").body
 insert=next(i for i,n in enumerate(port.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='result'for t in n.targets))
 port.body[insert:insert]=ast.parse("source=source.replace('metadata, cfg = admission(plan)','command_preflight = preflight(plan, a.output)\\n    metadata, cfg = admission(plan)')").body
 exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[single-OFF-short-read0]', 'exec'),namespace)
 namespace['intake']=SimpleNamespace(**{**vars(original.intake),'SDK':fresh.SDK});namespace['life']=SimpleNamespace(**{**vars(original.life),'contract':SimpleNamespace(**{**vars(original.life.contract),'native_command':namespace['native_command']})});namespace['_PORTS']=boundary_ports(namespace);return namespace
NS=derive()
for name in('PLAN','PHASES','contract','closure','set_case','set_runtime','admission','prerequisite','phase_recipe','native_command','life','warm_control','mapped','native_pid','run_phase','ports'):
 globals()[name]=NS[name]
def __getattr__(name):
 if name in NS:return NS[name]
 raise AttributeError(name)
if __name__=='__main__':NS['_PORTS']['main']()
