#!/usr/bin/env python3
"""Recompute source-qualified actual 2/4/6 scoped native/API/serial proof joins."""
import argparse,json,re,time
from pathlib import Path
from batch_numerical_proofs_v4 import require,sha,read,write,PLAN_SHA,lane_contract,off_on_histories,owner_proofs,validate_artifacts,confined_file
from batch_numerical_prefixes_v2 import compare_all
HERE=Path(__file__).resolve().parent

def parent_arm(directory):
 parent=read(directory/'parent-qualification.json');plan=read(directory/'input-plan.snapshot.json');child=read(directory/'child/report.json')
 require(parent['passed'] is True and parent['scoped_collection_or_serial_arm_qualified'] is True and parent['child_return_code']==0 and parent['owned_containers_terminal'] is True and parent['forced_cleanup'] is False and not parent['interrupted'] and not parent['errors'],'Actual owned source/health parent arm failed')
 require(plan['dependency_sha256']['audit_batch_numerical_suite_v4.py']==sha(Path(__file__)),'Original final auditor source differs from immutable arm plan')
 for name,digest in plan['dependency_sha256'].items():require(sha(HERE/name)==digest,'Final consumed source dependency changed '+name)
 require(parent['wrapper_sha256']==sha(HERE/'qualify_batch_numerical_v4.py') and parent['controller_sha256']==sha(HERE/'batch_numerical_execution_v4.py'),'Actual arm parent/controller source changed')
 require(plan['source_plan_sha256']==lane_contract(plan['lane'])['sha256'] and parent['plan_sha256']==sha(Path(parent['plan'])) and parent['child_report_sha256']==sha(directory/'child/report.json'),'Actual source/plan/child hash join differs')
 require(parent['pre_health_passed'] and parent['post_health_passed'] and parent['kernel_fault_gate_passed'],'Actual per-arm health/journal proof absent')
 for stage in ('pre','post'):
  health=read(directory/(stage+'-health.json'));require(health['passed'] and health['cards']==[0,1] and health['health_image']=='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067','Per-card/compiled pair health identity differs')
  require(len(health['files'])==2 and all(row['return_code']==0 and row['error'] is None and sha(row['path'])==row['sha256'] for row in health['files']),'Actual health logs/commands absent or changed')
 binding=parent['post_model_identity'];require(sha(binding['path'])==binding['sha256'],'Actual post4 receipt changed');identity=read(binding['path']);lock=read(HERE/'model-lock.json');files=[f for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')]
 require(identity['passed'] is True and identity['lock_sha256']==sha(HERE/'model-lock.json') and identity['started']>=max(child['finished_epoch'],parent['post_health_finished_epoch']) and len(identity['rows'])==len(files)==4,'Actual mandatory postterminal/posthealth full4 chronology or source lock differs')
 for row,want in zip(identity['rows'],files):require(row['passed'] and row['bytes']==want['size'] and row['sha256']==row['expected_sha256']==want['sha256'] and row['stat_before']==row['stat_after'] and len(row['stat_before'])==5,'Actual complete publisher shard/stat5 proof differs')
 validate_artifacts(directory/'child',child['artifact_bindings'])
 require(parent['source_guard_generation']==3,'Both known-page source guard missing')
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
 from audit_batch_fidelity_coverage_v2 import coverage as initial
 from audit_batch_fidelity_coverage_v4 import coverage as migration
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
 return {'schema':4,'passed':True,'current_PLE32_semantic_source_required':True,'source29_31_historical_diagnostic_results_accepted':False,'scoped_native_API_2_4_6_collection_qualified':True,'engine_receipt_sha256':engine,'prepared_sha256':prepared,'cards':list(cards),'rows':rows,'source_plan_sha256':lane_contract(on[2]['lane'])['sha256'],'source_lane':on[2]['lane'],'full_cache_public_fresh_pin_qualified':False,'full_model_math_qualified':False,'complete_concurrent_serving_goal_qualified':False,'serving_mirror_owner_logical_frees':all(row['serving_mirror_owner_logical_frees'] is True for row in rows),'batch_off_raw49_equality_qualified':False,'latency_qualified':False,'finished_epoch':time.time(),'scope':'actual on49 versus exact-prefix native serial; off/on IDs and real cancellation common prefix; each actual process has its own full source/health/owned teardown proof. Cache-enabled/public prefix/session isolation/fairness/original own-state math remain required separate lanes'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--roster',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(read(a.roster));write(a.output,result);print(json.dumps({'scoped_collection_qualified':result['passed'],'full_goal_qualified':False}));return 0
if __name__=='__main__':raise SystemExit(main())
