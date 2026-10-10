#!/usr/bin/env python3
"""Recompute source-qualified actual 2/4/6 scoped native/API/serial proof joins."""
import argparse,json,os,re,sys,time
from pathlib import Path
from batch_numerical_proofs_v37 import require,sha,read,write,PLAN_SHA,lane_contract,off_on_histories,owner_proofs,validate_artifacts,confined_file,source_observers_off,ROOT
from batch_numerical_prefixes_v2 import compare_all
from batch37_runtime_evidence_v1 import child_binding,journal_binding
HERE=Path(__file__).resolve().parent

def snapshot_plan_join(directory,parent,plan,child):
 directory=Path(directory).resolve();input_path=directory/'input-plan.snapshot.json';child_path=directory/'child/plan.snapshot.json';external_path=Path(parent['plan'])
 require(read(input_path)==plan==read(child_path)==read(external_path),'External/input/child snapshot plan content differs')
 require(parent['plan_sha256']==child['plan_sha256']==sha(input_path)==sha(child_path)==sha(external_path),'Complete external/input/child plan SHA association differs')
 return parent['plan_sha256']

def final_source_join(directory,parent,plan,child):
 from batch_numerical_proofs_v37 import genuine_baseline,engine_binding,ROOT
 from batch_numerical_execution_v37 import verify_model_identity
 from source_page_watchdog_v3 import guard,KNOWN_PAGES
 directory=Path(directory).resolve();snapshot_plan_join(directory,parent,plan,child)
 prepared,chain=genuine_baseline(Path(plan['prepared']),plan['lane']);require(chain==plan['baseline_source_proof']==parent['prepared_chain'],'Current genuine C137/source37 preparation chain differs')
 require(prepared['engine_receipt_sha256']==plan['engine_receipt_sha256'] and prepared['cards']==plan['cards'] and Path(plan['engine_root']).resolve()==Path(prepared['engine_receipt']).resolve().parent,'Current prepared SDK/topology/path differs');engine_binding(Path(plan['engine_root']),plan['lane'])
 path=directory/'post-model-identity.json';binding=parent['post_model_identity'];require(Path(binding['path']).resolve()==path and binding['sha256']==sha(path),'Postidentity exact parent path/hash association differs')
 identity=read(path);require(identity['started']>=journal_binding(directory,parent,'post')['finished_epoch'],'New full4 before complete postjournal');lock_path=HERE/'model-lock.json';lock=read(lock_path);files=[f for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')];shards=[ROOT/lock['destination']/f['path'] for f in files];health=read(directory/'post-health.json')
 require(health['finished_epoch']==parent['post_health_finished_epoch'] and health['passed'] is True,'Posthealth chronology/parent association differs');boundary=max(parent['child_terminal_epoch'],child['finished_epoch'],health['finished_epoch'])
 require(identity['passed'] is True and identity['lock_sha256']==sha(lock_path) and identity['model_revision']==lock['revision'] and identity['started']>=boundary and identity['finished']>=identity['started'] and identity['after_child_terminal_epoch']==boundary and parent['finished_epoch']>=identity['finished'],'Actual terminal/posthealth full4 source chronology differs')
 require(len(identity['rows'])==len(files)==len(shards)==4,'Exact four publisher shard identity required')
 for row,want,source in zip(identity['rows'],files,shards):
  st=source.stat();current=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
  require(Path(row['path']).resolve()==source.resolve() and row['passed'] is True and row['bytes']==want['size'] and row['sha256']==row['expected_sha256']==want['sha256'] and row['stat_before']==row['stat_after']==current,'Current ordered shard path/publisher hash/stat5 differs')
 current=verify_model_identity(path,lock,shards);pages=guard(shards[2])
 for name in ('known_pages_before_hash','known_pages_after_hash'):
  saved=parent[name];require(saved['passed'] is True and Path(saved['path']).resolve()==shards[2].resolve() and saved['stat_before']==saved['stat_after']==identity['rows'][2]['stat_after'],'Stored both-page source/path/stat proof differs')
  require(len(saved['rows'])==2 and [(row['offset'],row['expected_sha256']) for row in saved['rows']]==list(KNOWN_PAGES),'Exact two known-page offsets/digests required')
  for row in saved['rows']:
   rawpath=Path(row['preserved_path']);require(rawpath.resolve().parent==directory and not rawpath.is_symlink() and rawpath.stat().st_size==4096 and row['passed'] is True and row['bytes']==4096 and sha(rawpath)==row['sha256']==row['expected_sha256'],'Preserved known-page view changed/escaped/failed')
  require([(r['offset'],r['sha256']) for r in saved['rows']]==[(r['offset'],r['sha256']) for r in pages['rows']],'Current known-page view differs from recorded brackets')
 require(parent['known_pages_before_hash']['epoch']<=identity['started']<=identity['finished']<=parent['known_pages_after_hash']['epoch']<=parent['finished_epoch'],'Known pages must bracket new full4 scan before parent final')
 # Recheck current source/canonical page binding after reader work as well.
 _,after=genuine_baseline(Path(plan['prepared']),plan['lane']);require(after==chain,'Current C137/SDK source changed during readonly join');guard(shards[2])
 return {'snapshot_plan_sha256':parent['plan_sha256'],'current_C137_source_chain':chain,'current_post_model_identity':current,'current_known_pages':pages,'post_terminal_and_health_boundary':boundary,'new_full4_bracketed_by_two_page_views':True}

def parent_arm(directory):
 require(__debug__ and sys.flags.optimize==0 and os.environ.get('PYTHONOPTIMIZE','0') in ('','0'),'Strict frozen final audit requires assertions enabled; optimized Python refused')
 directory=Path(directory).resolve();parent=read(directory/'parent-qualification.json');plan=read(directory/'input-plan.snapshot.json');child=read(directory/'child/report.json')
 source_observers_off(plan['env']);child_binding(directory,parent,plan,child)
 from batch_numerical_execution_v37 import output_budget_contract,manifest_binding
 manifest_binding(plan)
 require(plan.get('schema')==4 and plan.get('harness_generation')==37,'OldV6/foreign finalized harness refused');budget=output_budget_contract(plan['kind'],plan['slots'],plan['native_diagnostic_max_new_by_request'],plan['api_max_new_by_request'],plan.get('serial_group_job_count'));require(plan['output_budget_contract']==budget and plan['max_new_by_request']==budget['actual_submission_budget'],'Final pertransport output budget changed')
 require(parent['passed'] is True and parent['scoped_collection_or_serial_arm_qualified'] is True and parent['child_return_code']==0 and parent['owned_containers_terminal'] is True and parent['forced_cleanup'] is False and not parent['interrupted'] and not parent['errors'],'Actual owned source/health parent arm failed')
 require(plan['dependency_sha256']['audit_batch_numerical_suite_v37.py']==sha(Path(__file__)),'Original final auditor source differs from immutable arm plan')
 for name,digest in plan['dependency_sha256'].items():require(sha(HERE/name)==digest,'Final consumed source dependency changed '+name)
 require(parent['wrapper_sha256']==sha(HERE/'qualify_batch_numerical_v37.py') and parent['controller_sha256']==sha(HERE/'batch_numerical_execution_v37.py'),'Actual arm parent/controller source changed')
 require(plan['source_plan_sha256']==lane_contract(plan['lane'])['sha256'] and parent['plan_sha256']==sha(Path(parent['plan'])) and parent['child_report_sha256']==sha(directory/'child/report.json'),'Actual source/plan/child hash join differs')
 require(parent['pre_health_passed'] and parent['post_health_passed'] and parent['kernel_fault_gate_passed'],'Actual per-arm health/journal proof absent')
 for stage in ('pre','post'):
  health=read(directory/(stage+'-health.json'));require(health['passed'] and health['cards']==[0,1] and health['health_image']=='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067','Per-card/compiled pair health identity differs')
  require(len(health['files'])==2 and all(row['return_code']==0 and row['error'] is None and sha(row['path'])==row['sha256'] for row in health['files']),'Actual health logs/commands absent or changed')
  commands=[[str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',health['health_image']],[str(ROOT/'bin/xpu-collective-health'),'--img',health['health_image'],'--p2p','0','--timeout','180']]
  for row,argv,label in zip(health['files'],commands,('strict','compiled-pair')):
   log=directory/(stage+'-'+label+'.log');cmd=directory/(stage+'-'+label+'.command.json');require(row['path']==str(log) and row['command']==argv==read(cmd) and row['command_file_sha256']==sha(cmd) and row['stdout_sha256']==sha(log) and health['started_epoch']<=row['started_epoch']<=row['finished_epoch']<=health['finished_epoch'],'Actual health exactargv/log/chronology differs')
 binding=parent['post_model_identity'];require(sha(binding['path'])==binding['sha256'],'Actual post4 receipt changed');identity=read(binding['path']);lock=read(HERE/'model-lock.json');files=[f for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')]
 require(identity['passed'] is True and identity['lock_sha256']==sha(HERE/'model-lock.json') and identity['started']>=max(child['finished_epoch'],parent['post_health_finished_epoch']) and len(identity['rows'])==len(files)==4,'Actual mandatory postterminal/posthealth full4 chronology or source lock differs')
 for row,want in zip(identity['rows'],files):require(row['passed'] and row['bytes']==want['size'] and row['sha256']==row['expected_sha256']==want['sha256'] and row['stat_before']==row['stat_after'] and len(row['stat_before'])==5,'Actual complete publisher shard/stat5 proof differs')
 validate_artifacts(directory/'child',child['artifact_bindings'])
 require(parent['source_guard_generation']==3,'Both known-page source guard missing')
 final_source_join(directory,parent,plan,child)
 return parent,plan,child

def native_histories(directory,plan):
 roster=read(directory/'child/requests.json');rows={}
 for index in range(plan['slots']):
  row=roster[str(2001+index)];require(row['ids']==plan['tokens']['target'][index] and row['done'],'Actual native target submitted/terminal differs');words=row['done'].split();rows[str(index)]={'submitted_ids':row['ids'],'sampling':{'temperature':0},'max_new':plan['max_new'],'native_generated_ids':row['generated'],'native_finish':words[3] if words[0]=='BDONE' else words[5],'native_cancel_completed':words[0]=='BDONE' and words[3]=='cancel'}
 return rows

def api_histories(directory,plan):
 events=[json.loads(l) for l in (directory/'child/api-native-trace.jsonl').read_text().splitlines()];begins={e['call']:e for e in events if e['kind']=='engine_begin'};ends={e['call']:e for e in events if e['kind']=='engine_end'};prefix=read(directory/'child/prefixes.json');rows={}
 for index,messages in enumerate(plan['messages']['target']):
  matching=[call for call,e in begins.items() if e['rendered_prompt']['messages']==messages];require(len(matching)==1,'Exact logical API input roster ambiguous');call=matching[0];begin=begins[call];end=ends[call];segments=[seg for history in prefix['native_segments'].values() for seg in history if seg['call']==call];require(segments and all(seg['terminal'] for seg in segments),'Actual API native history/terminal absent');segments.sort(key=lambda x:x['send_sequence']);actual=[token for seg in segments for token in seg['generated']]
  rows[str(index)]={'submitted_ids':begin['submitted_ids'],'sampling':begin['sampling'],'max_new':begin['max_new'],'native_generated_ids':actual,'native_finish':end['engine_last'].get('finish'),'native_cancel_completed':end['cancelled'] and any(e['kind']=='native_send' and e.get('call')==call and e['line'].startswith(('BSTOP ','STOP')) for e in events)}
 return rows

def vectors(directory):
 result={}
 plan=read(directory/'input-plan.snapshot.json');report=read(directory/'child/report.json');stages=[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])];trace=(directory/'child/engine.combined.log').read_text()
 from audit_batch_fidelity_coverage_v5 import coverage as initial
 from audit_batch_fidelity_coverage_v6 import coverage as migration
 if plan['kind']=='native':
  old=report['report']['raw'];new=initial(trace,old['requests'],stages,directory/'child/captures')
 else:
  old=report['raw']['raw'];new=migration(trace,old['initial']['requests'],old['migration_requests'],stages,directory/'child/captures')
 require(json.loads(json.dumps(new))==old,'Final original coverage/hash/provenance differs from child-terminal receipt')
 for line in (directory/'child/engine.combined.log').read_text().splitlines():
  if not line.startswith('SBF vector '):continue
  f=dict(re.findall(r'(\w+)=([^ ]+)',line));role='admission' if f['phase'].startswith('admission_') else 'later' if f['phase'].startswith('batch_step_') else 'solo_migration';key=((int(f['rid']),role),int(f['layer']));require(key not in result,'Duplicate actual RID/role/layer vector');require(Path(f['file']).parent==Path('/results/captures'),'Actual producer file path outside exact capture root');file=confined_file(directory/'child/captures',Path(f['file']).name);count=248320 if int(f['layer'])==-1 else 10240;require(f['canonical']=='le_f32' and int(f['floats'])==count and int(f['bytes'])==count*4 and file.stat().st_size==count*4 and not file.is_symlink(),'Final actual vector geometry changed');result[key]=str(file)
 return result

def serial_vectors(directory):
 result={};child=directory/'child';plan=read(directory/'input-plan.snapshot.json');source=read(Path(plan['batch_parent'])/'child/serial-jobs.json')['jobs'];expected=source[plan['group_index']*6:plan['group_index']*6+6];seen=[]
 import serial_prefix_qualification_v6 as collector
 for path in child.glob('serial-*/requests.json'):
  for row in read(path):
   job=row['job'];require(job in expected and job not in seen,'Serial source token job changed/duplicated');seen.append(job);key=(job['rid'],job['role']);meta=row['meta'];require(row['raw']['ids']==job['ids'] and row['raw']['fresh']==1,'Actual serial input/fresh role differs from selected source job');new=collector.extract(row['raw'],path.parent/'captures',True,True,[tuple(x) for x in plan['stage_ranges']]);require(json.loads(json.dumps(new))==meta,'Final original serial collector/hash/provenance differs');items=[(-1,meta['logits'][0]['path'])]+[(int(v['layer']),v['path']) for v in meta['residuals']]
   require({layer for layer,name in items}==set(range(-1,48)),'Actual serial fullhead/all48 absent')
   for layer,name in items:require((key,layer) not in result,'Duplicate serial group identity');result[key,layer]=name
 require(seen==expected,'Complete source-bound selected serial group job roster differs')
 return result

def paired_two_control(control,engine_sha256):
 """Actual source37 OFF/ON plus every full49 serial group before4/6."""
 import importlib
 reader_path=HERE/'validate_private_native_offon_source37_v1.py'
 require(type(control.get('reader_sha256')) is str and len(control['reader_sha256'])==64 and sha(reader_path)==control['reader_sha256'],'Explicit frozen private OFF/ON reader source required before4/6')
 reader=importlib.import_module('validate_private_native_offon_source37_v1')
 require(Path(reader.__file__).resolve()==reader_path.resolve(),'Foreign private reader import refused')
 off=Path(control['off_root']).resolve();on=Path(control['on_root']).resolve();private=reader.finalized_binding(off,on)
 a,offplan,_=parent_arm(off);b,onplan,_=parent_arm(on)
 require(offplan['slots']==onplan['slots']==2 and offplan['kind']==onplan['kind']=='native' and offplan['diagnostic']==0 and onplan['diagnostic']==1 and offplan['cards']==onplan['cards']==[0,1] and offplan['engine_receipt_sha256']==onplan['engine_receipt_sha256']==engine_sha256,'Actual paired2 same-newSDK native OFF/ON prerequisite differs')
 batch=vectors(on);serial={};groups=[];bindings=[]
 for root in control['serial_roots']:
  root=Path(root).resolve();parent,plan,_=parent_arm(root)
  require(plan['kind']=='serial' and plan['slots']==2 and Path(plan['batch_parent']).resolve()==on and plan['engine_receipt_sha256']==engine_sha256,'Actual paired2 serial source association differs')
  rows=serial_vectors(root);require(not(set(rows)&set(serial)),'Repeated actual paired2 serial rows');serial.update(rows);groups.append(plan['group_index']);bindings.append({'root':str(root),'parent_sha256':sha(root/'parent-qualification.json')})
 jobs=read(on/'child/serial-jobs.json')['jobs'];require(len(groups)==len(set(groups)) and set(groups)==set(range((len(jobs)+5)//6)),'Complete actual paired2 selected-prefix serial group roster required')
 raw=compare_all(batch,serial);require(raw['passed'],'Actual paired2 complete48/head serial arithmetic control failed')
 return {'off_root':str(off),'on_root':str(on),'off_parent_sha256':sha(off/'parent-qualification.json'),'on_parent_sha256':sha(on/'parent-qualification.json'),'private_reader_sha256':control['reader_sha256'],'actual_private_off_on_binding':private,'actual_all49_serial':raw,'actual_serial_parents':bindings,'engine_receipt_sha256':engine_sha256,'full_model_math_qualified':False,'latency_qualified':False}

def audit(roster):
 arms={};rows=[];engine=None;prepared=None;cards=None
 for entry in roster:
  path=Path(entry['directory']);parent,plan,child=parent_arm(path);key=(plan['slots'],plan['kind'],plan['diagnostic'],plan.get('group_index'),plan.get('batch_parent'))
  require(key not in arms,'Duplicate topology/case arm');arms[key]=(path,parent,plan,child)
  identity=(plan['engine_receipt_sha256'],plan['prepared_sha256'],tuple(plan['cards']))
  if engine is None:engine,prepared,cards=identity
  require(identity==(engine,prepared,cards),'Old SDK/C1 proof, foreign topology or independently changed prepared baseline mixed into suite')
 for n in (2,4,6):
  for kind in ('native','api'):
   on=arms[n,kind,1,None,None];off=arms[n,kind,0,None,None];history=native_histories if kind=='native' else api_histories
   compare=off_on_histories(history(off[0],off[2]),history(on[0],on[2]),{str(on[2]['cancel_index'])});batch=vectors(on[0]);require(batch,'Actual on raw49 roster absent');serial={};groups=[]
   for key,entry in arms.items():
    if key[0]!=n or key[1]!='serial' or entry[2]['batch_parent']!=str(on[0].resolve()):continue
    subset=serial_vectors(entry[0]);require(not(set(subset)&set(serial)),'Duplicated actual serial group raw vectors');serial.update(subset);groups.append(key[3])
   jobs=read(on[0]/'child/serial-jobs.json')['jobs'];required=(len(jobs)+5)//6;require(set(groups)==set(range(required)),'Every actual selected consumed-prefix serial group source/health proof required')
   raw=compare_all(batch,serial);require(raw['passed'],'Actual complete per-RID head49 differs from exact-prefix serial');stages=[(i,lo,hi) for i,(lo,hi) in enumerate(on[2]['stage_ranges'])];owned=owner_proofs((on[0]/'child/engine.combined.log').read_text(),stages,n,on[2]['lane'])
   rows.append({'requested':n,'route':kind,'on_parent':str(on[0]),'off_parent':str(off[0]),'actual_off_on_token_history':compare,'actual_on_vs_exact_consumed_prefix_serial_raw49':raw,'actual_snapshot_and_slot_owner_logical_frees':owned,'serving_mirror_owner_logical_frees':owned['serving_mirror_owner_logical_frees'],'actual_off_batch_raw49':'UNOBSERVED; disabled observer has no capture copies'})
 return {'schema':5,'passed':True,'current_PLE32_semantic_source_required':True,'source29_31_historical_diagnostic_results_accepted':False,'scoped_native_API_2_4_6_collection_qualified':True,'engine_receipt_sha256':engine,'prepared_sha256':prepared,'cards':list(cards),'rows':rows,'source_plan_sha256':lane_contract(on[2]['lane'])['sha256'],'source_lane':on[2]['lane'],'full_cache_public_fresh_pin_qualified':False,'full_model_math_qualified':False,'complete_concurrent_serving_goal_qualified':False,'serving_mirror_owner_logical_frees':all(row['serving_mirror_owner_logical_frees'] is True for row in rows),'batch_off_raw49_equality_qualified':False,'latency_qualified':False,'finished_epoch':time.time(),'scope':'actual on49 versus exact-prefix native serial; off/on IDs and real cancellation common prefix; each actual process has its own full source/health/owned teardown proof. Cache-enabled/public prefix/session isolation/fairness/original own-state math remain required separate lanes'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--roster',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(read(a.roster));write(a.output,result);print(json.dumps({'scoped_collection_qualified':result['passed'],'full_goal_qualified':False}));return 0
if __name__=='__main__':raise SystemExit(main())
