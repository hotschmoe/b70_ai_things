#!/usr/bin/env python3
"""Matched same62 binary P30 OFF/ON, SFD in both; no own-state/fullmath proof."""
import argparse,json,os,shlex,signal,subprocess,time,re,hashlib
from pathlib import Path
import c1_serve_controller_combined_v11 as c1
from qualify_c1_serving_combined_v11 import validate_final_source_proof
import serial_prefix_qualification_v6 as base
import layer0_numerical_qualification_v9 as numeric
from merged_numerical_protocol_v2 import MergedProtocol
from collect_prefix_residual30_v1 import collect
from collect_ple_input33_v1 import collect as collect_input33
from audit_prefix30_lifecycle_v1 import audit_text
from audit_v6_actual_request_records import strengthen
ROOT=Path(__file__).resolve().parents[2];sha=c1.sha;read=c1.read;write=c1.write;require=c1.require
PREFIXES=[1,2,4,8];PLAN_SOURCE='strata/flash-next/current-ple-input33-engine-build-plan-v1.json';PLAN_SOURCE_SHA='81798dff3fd012f7b95984403ea1088a1f157d37ee2d2e6e1b612711052cd34c'
expected_stage_ranges=base.expected_stage_ranges;verify_model_identity=numeric.verify_model_identity
OFF=('STRATA_BATCH_FULL_STATE_CHAIN','STRATA_BATCH_PUBLIC_PREFIX','STRATA_BATCH_FIDELITY_DIAG','STRATA_SLOT_OWNER_TRACE','STRATA_MIRROR_OWNER_TRACE','STRATA_LAYER0_Q8_DIAG','STRATA_PREFIX30','STRATA_PLE_INPUT33')
DEPENDENCIES=['c1_serve_controller_combined_v11.py','qualify_c1_serving_combined_v11.py','c1-combined-v11-source-plan.json','c1-combined-v11-parent-source-plan.json','collect_ple_input33_v1.py','verify_ple_input33_original_v1.py','ple-input33-source-plan-v1.json','collect_prefix_residual30_v1.py','audit_prefix30_lifecycle_v1.py','serial_prefix_qualification_v6.py','layer0_numerical_qualification_v9.py','merged_numerical_protocol_v2.py','audit_v6_actual_request_records.py','source_page_watchdog_v3.py','audit_fidelity_observer_coverage.py','prefix_residual_capture_contract_v1.hpp']

def candidate_binding(plan):
 require(sha(ROOT/PLAN_SOURCE)==PLAN_SOURCE_SHA,'Exact source63 plan changed');engine=Path(plan['engine_root']);generation=c1.combined_generation_gate(engine);m=read(Path(plan['prepared'])/'prepared.json');c1.metadata_admission_gate(m);proof=validate_final_source_proof(Path(plan['prepared']),m)
 require(m['engine_receipt_sha256']==plan['engine_receipt_sha256']==generation['engine_receipt_sha256'] and sha(Path(plan['prepared'])/'prepared.json')==plan['prepared_sha256'],'Actual C111/SDK association differs')
 c1.validate_prepared(Path(plan['prepared']));env=read(Path(plan['prepared'])/'server-config.json')['env'];require(all(env.get(k,'0')=='0' for k in OFF),'Activated source features cannot qualify baseline P30 fixture')
 return {'generation':generation,'actual_C111_final_source_proof':proof,'upload_post_full4_sha256':m['upload_lifecycle']['post_model_identity_sha256']}

def manifest_binding(plan):
 require(plan['driver_sha256']==sha(Path(__file__)) and plan['source_plan_sha256']==PLAN_SOURCE_SHA,'P30 driver/plan changed')
 for name,digest in plan['dependency_sha256'].items():require(sha(ROOT/name)==digest,'P30 validation dependency changed '+name)
 require(plan['prefixes']=={str(n):plan['tokens']['cases']['A'][:n] for n in PREFIXES} and plan['max_new']==1,'Fixed P30 input/request geometry differs')
 base.geometry_contract(plan['args']);return candidate_binding(plan)

def prepare(a):
 m=read(a.prepared/'prepared.json');preflight={'engine_root':str(a.engine_root.resolve()),'prepared':str(a.prepared.resolve()),'engine_receipt_sha256':sha(a.engine_root/'receipt.json'),'prepared_sha256':sha(a.prepared/'prepared.json')};candidate_binding(preflight)
 args=list(read(a.prepared/'server-config.json')['args'])
 for flag,value in base.GEOMETRY.items():base.option(args,flag,value)
 require('--mtp' not in args and '--pipeline-windows' not in args and '--no-prefill-borrow' in args,'Unsupported P30 math/topology route')
 code=ROOT/'strata/flash-next/serial_prefix_prompt_fixture_v6.py';tokens=json.loads(subprocess.check_output(['docker','run','--rm','--network','none','--user','1000:1000','-v',str(a.engine_root/'source')+':/src:ro','-v',m['pack']+':/pack:ro','-v',str(code)+':/fixture.py:ro',m['runtime']['image'],'exec /opt/b70-c1-python/bin/python /fixture.py'],text=True,timeout=90));require(tokens['vocab']==248320 and tokens['tokenizer_sha256']==read(a.prepared/'artifact-identity.json')['tokenizer_files'],'Actual tokenizer/fixture identity differs');base.option(args,'--turn-token',tokens['turn_token'])
 identity=verify_model_identity(a.model_identity,read(ROOT/'strata/flash-next/model-lock.json'),[Path(r['path']) for r in m['model_shards']])
 plan={**preflight,'schema':1,'driver_sha256':sha(Path(__file__)),'source_plan_sha256':PLAN_SOURCE_SHA,'dependency_sha256':{'strata/flash-next/'+n:sha(ROOT/'strata/flash-next'/n) for n in DEPENDENCIES},'token_fixture_sha256':sha(code),'tokens':tokens,'args':args,'cards':m['cards'],'image':m['runtime']['image'],'model_identity':identity,'max_new':1,'prefixes':{str(n):tokens['cases']['A'][:n] for n in PREFIXES},'primary_model_name':'hotschmoe-dd','base_prepared_alias':m['alias'],'research_alias':base.pilot_alias(m['cards'],args)+'-P30-off-on-currentPLE32-input33-C111','transport':'serial nativeGEN','full_model_math_qualified':False,'graph_retirement_runtime_handle_association_observed':False}
 manifest_binding(plan);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',plan);print(json.dumps({'prepared':str(a.output/'plan.json'),'GPU_executed':False}))

def cross_sfd(captured,requests):
 result=[];seen=set()
 for frame in captured['frames']:
  d=frame['binding']
  if d['route']!='verifier':continue
  row=next(r for r in requests if r['prefix']==len(d['gen_ids']));sfd={int(v['layer']):v for v in row['meta']['residuals']};require(d['pid']==int(row['meta']['logits'][0]['pid']) and d['request']==int(row['meta']['logits'][0]['request']),'P30/SFD actual request/PID differs')
  for f in frame['fields']:
   if f['phase']!='ffn':continue
   layer=f['layer'];key=(row['prefix'],layer);require(key not in seen,'Duplicate P30 lastrow mapping');seen.add(key);v=sfd[layer]
   require(int(v['stage'])==d['stage'] and int(v['pos'])==d['first_position'] and int(v['token'])==d['gen_ids'][-1] and f['bytes']==int(v['bytes'])==40960,'P30/SFD layer/stage/row geometry differs')
   require(sha(f['path'])==f['sha256'] and sha(v['path'])==v['sha256'],'P30/SFD raw source bytes changed since collection')
   comparison=base.compare_vectors(f,v);require(comparison['bitwise_equal'],'Actual P30 lastverifier postFFN differs from SFD: '+str(key));result.append({'prefix':row['prefix'],'layer':layer,**comparison})
 require(seen=={(n,l) for n in PREFIXES for l in range(48)},'All192 actual lastrow layer pairs required');return {'passed':True,'pairs':result,'actual_row_mapping_bitwise_verified':True,'full_model_math_qualified':False}

def run_process(plan,output,on):
 name='p30_on' if on else 'p30_off';directory=output/name;directory.mkdir();(directory/'captures').mkdir();(directory/'p30').mkdir();(directory/'ple-input').mkdir();m=c1.validate_prepared(Path(plan['prepared']));env=dict(read(Path(plan['prepared'])/'server-config.json')['env']);require(all(env.get(k,'0')=='0' for k in OFF),'P30 baseline route changed');binding=sha(output/'plan.snapshot.json')
 env.update({'STRATA_FIDELITY_DIAG':'1','STRATA_FIDELITY_DIAG_ACTIVATIONS':'1','STRATA_FIDELITY_DIAG_ARM':'/results/ARM','STRATA_FIDELITY_DIAG_DIR':'/results/captures','STRATA_PREFIX_DIAG':'1','STRATA_PREFIX_LIFECYCLE_DIAG':'1','STRATA_PREFIX_DIAG_ARM':'/results/ARM','STRATA_PREFIX30':str(int(on)),'STRATA_PREFIX30_DIR':'/results/p30','STRATA_PREFIX30_BINDING_SHA256':binding,'STRATA_PLE_INPUT33':'1','STRATA_PLE_INPUT33_DIR':'/results/ple-input','STRATA_PLE_INPUT33_BINDING_SHA256':binding,'STRATA_LAYER0_Q8_DIAG':'0','STRATA_BATCH_FIDELITY_DIAG':'0','STRATA_SLOT_OWNER_TRACE':'0','STRATA_MIRROR_OWNER_TRACE':'0','STRATA_ARTIFACT_IDENTITY_SHA256':binding,'SYCL_UR_TRACE':'2'})
 command=['docker','run','-i','--name','b70-prefix-'+str(os.getpid())+'-'+name,'--label','b70.prefix.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(Path(plan['engine_root'])/'build')+':/build:ro','-v',str(Path(plan['engine_root'])/'source')+':/src:ro','-v',m['pack']+':/pack:ro','-v',str(ROOT/read(ROOT/'strata/flash-next/model-lock.json')['destination'])+':/model:ro','-v',str(directory)+':/results']
 for key,value in sorted(env.items()):command+=['-e',key+'='+value]
 command +=[plan['image'],'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+plan['args'])];write(directory/'command.json',command);protocol=None;rows=[];error=None;capture=None;mapping=None
 try:
  protocol=MergedProtocol(command,directory)
  for key in ('A','B','C'):protocol.request('warm-'+key,plan['tokens']['cases'][key],fresh=1,max_new=1)
  (directory/'ARM').write_text('ARM matched weights/warmups complete\n',encoding='ascii')
  for n in PREFIXES:
   raw=protocol.request('prefix-'+str(n),plan['prefixes'][str(n)],fresh=1,max_new=1);meta=base.extract(raw,directory/'captures',True,True,expected_stage_ranges(plan['args']));external=strengthen(raw,expected_stage_ranges(plan['args']),True,True);rows.append({'prefix':n,'raw':raw,'meta':meta,'external':external});write(directory/'requests.json',rows)
 except Exception as e:error=str(e)
 finally:
  try:rc=protocol.close() if protocol else None
  except Exception as e:rc=None;error=str(e)
  container='b70-prefix-'+str(os.getpid())+'-'+name;state=c1.inspected(container)['State'];removed=False
  if not state['Running']:subprocess.run(['docker','rm',container],check=True,capture_output=True,text=True);removed=c1.absent(container)
 try:
  logtext=(directory/'engine.combined.log').read_text();require(not re.search(r'trying a smaller expert cache|trying a \d+-token chunk|strata serve: a \d+-token chunk.*: \d+-token chunks',logtext),'Actual weights/prefill capacity fallback refused; declared matched geometry not preserved')
  input_requests={int(r['meta']['logits'][0]['request']):r['raw']['ids'] for r in rows};write(directory/'input33-requests.json',input_requests);input_proof=collect_input33(directory/'ple-input',input_requests,binding,directory/'engine.combined.log',expected_stage_ranges(plan['args'])[0]);write(directory/'input33-proof.json',input_proof)
  if on:capture=collect(directory/'p30',{int(r['meta']['logits'][0]['request']):r['raw']['ids'] for r in rows},dict(enumerate(expected_stage_ranges(plan['args']))),binding,directory/'engine.combined.log');mapping=cross_sfd(capture,rows);write(directory/'capture.json',capture);write(directory/'lastrow-mapping.json',mapping)
  else:require(not list((directory/'p30').glob('*')) and not any('P30 ' in line for line in (directory/'engine.combined.log').read_text().splitlines()),'P30OFF produced allocation/fields')
 except Exception as e:error=(error+'; ' if error else '')+str(e)
 result={'passed':error is None and rc==0 and state['ExitCode']==0 and not state.get('OOMKilled') and removed and len(rows)==4,'error':error,'removed':removed,'engine_rc':rc,'state':state,'complete_four_prefix_roster':len(rows)==4,'full_model_math_qualified':False};write(directory/'result.json',result);return result

def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);h=read(a.pre_health);require(h['passed'] and set(plan['cards'])<=set(h['cards']) and len(h.get('files',[]))==2 and 0<=time.time()-h['finished_epoch']<=300,'Fresh actual prehealth required')
 for row in h['files']:require(sha(row['path'])==row['sha256'],'Health evidence changed')
 if len(plan['cards'])>1:
  require(a.one_card_receipt is not None,'Same P30generation onecard prerequisite');one=read(a.one_card_receipt);require(one['passed'] and one['cards']==[0] and one['engine_receipt_sha256']==plan['engine_receipt_sha256'] and one['P30_logical_lifecycle_qualified'] and one.get('driver_sha256')==sha(Path(__file__)) and one.get('actual_P30_lastrow_SFD48_bitwise') is True and one.get('actual_original_input33_bitwise') is True,'Actual sameengine P30onecard qualification required')
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.snapshot.json',plan);results=[];comparisons=[];error=None
 def stop(sig,frame):raise InterruptedError('Parent requested P30stop '+str(sig))
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
 try:
  for on in (False,True):r=run_process(plan,a.output,on);results.append(r);require(r['passed'],'P30 arm failed '+str(r['error']))
  off=read(a.output/'p30_off/requests.json');on=read(a.output/'p30_on/requests.json')
  require([r['prefix'] for r in off]==[r['prefix'] for r in on]==PREFIXES,'Actual prefix roster differs')
  for left,right in zip(off,on):comparisons.append({'prefix':left['prefix'],**numeric.compare_numeric(left,right)})
 except Exception as e:error=str(e)
 write(a.output/'report.json',{'schema':1,'driver_sha256':sha(Path(__file__)),'passed':False,'numerical_and_teardown_passed':error is None and len(comparisons)==4,'error':error,'post_health_passed':False,'cards':plan['cards'],'plan_sha256':sha(a.output/'plan.snapshot.json'),'engine_receipt_sha256':plan['engine_receipt_sha256'],'finished_epoch':time.time(),'results':results,'comparisons':comparisons,'full_model_math_qualified':False,'graph_retirement_runtime_handle_association_observed':False,'P30_logical_lifecycle_qualified':False})
 if error:raise SystemExit(1)

def validate_original_input33(receipt,output,plan,binding,identity):
 require(receipt is not None,'Actual original source33 row/embedding proof for BOTH arms required');inputs=read(receipt);require(inputs['passed'] and inputs['plan_sha256']==binding and set(inputs['arms'])=={'p30_off','p30_on'},'Actual original input proof association differs')
 for name in ('p30_off','p30_on'):
  directory=output/name;input_requests={int(r['meta']['logits'][0]['request']):r['raw']['ids'] for r in read(directory/'requests.json')};recollected=collect_input33(directory/'ple-input',input_requests,binding,directory/'engine.combined.log',expected_stage_ranges(plan['args'])[0]);require(recollected==read(directory/'input33-proof.json'),'Actual native prelaunch source33 fields/log changed');proof=inputs['arms'][name];require(len(proof.get('results',[]))==4 and {r['request'] for r in proof['results']}=={1,2,3,4} and all(r['bitwise_equal'] for r in proof['results']),'Complete actual4 original-input comparisons perarm required');require(proof['passed'] and proof['actual_original_staged_input_bitwise'] and proof['proof_sha256']==sha(directory/'input33-proof.json') and proof['model_identity_sha256']==sha(identity) and proof['collector_sha256']==sha(ROOT/'strata/flash-next/collect_ple_input33_v1.py') and proof['checker_sha256']==sha(ROOT/'strata/flash-next/verify_ple_input33_original_v1.py') and proof['producer_log_sha256']==sha(directory/'engine.combined.log') and proof['requests_sha256']==sha(directory/'input33-requests.json'),'Original PLE input33 actual data/decoder/source associations differ')
 return inputs

def finalize(a):
 result=read(a.output/'report.json');health=read(a.post_health);plan=read(a.output/'plan.snapshot.json');manifest_binding(plan);require(result['numerical_and_teardown_passed'] and health['passed'] and set(result['cards'])<=set(health['cards']) and len(health.get('files',[]))==2 and health['finished_epoch']>=result['finished_epoch'],'Actual numerical/teardown/posthealth failed');
 for row in health['files']:require(sha(row['path'])==row['sha256'],'Posthealth artifact changed')
 prepared=c1.validate_prepared(Path(plan['prepared']));identity=verify_model_identity(a.model_identity,read(ROOT/'strata/flash-next/model-lock.json'),[Path(r['path']) for r in prepared['model_shards']]);numeric.post_hash_boundary(read(a.model_identity),result,health);life=read(a.lifecycle_receipt);require(life['passed'] and life['plan_sha256']==result['plan_sha256'] and life['graph_retirement_runtime_handle_association_observed'] is False,'Actual P30 lifecycle/source association differs')
 for name,enabled in [('p30_off',False),('p30_on',True)]:
  log=a.output/name/'engine.combined.log';fresh=audit_text(log.read_text(),dict(enumerate(expected_stage_ranges(plan['args']))),enabled);fresh['log_sha256']=sha(log);require(json.loads(json.dumps(fresh))==life['arms'][name],'Actual observer lifecycle bytes/trace/contexts changed')
 capture=read(a.output/'p30_on/capture.json');rows=read(a.output/'p30_on/requests.json');recollected=collect(a.output/'p30_on/p30',{int(r['meta']['logits'][0]['request']):r['raw']['ids'] for r in rows},dict(enumerate(expected_stage_ranges(plan['args']))),result['plan_sha256'],a.output/'p30_on/engine.combined.log');require(capture==recollected,'Whole earlierrow capture metadata/bytes changed');mapping=cross_sfd(capture,read(a.output/'p30_on/requests.json'));require(mapping==read(a.output/'p30_on/lastrow-mapping.json'),'Actual rowmapping proof changed')
 validate_original_input33(a.input_receipt,a.output,plan,result['plan_sha256'],a.model_identity)
 result.update(passed=True,post_health_passed=True,post_model_identity=identity,P30_logical_lifecycle_qualified=True,actual_original_input33_bitwise=True,input33_receipt_sha256=sha(a.input_receipt),lifecycle_receipt_sha256=sha(a.lifecycle_receipt),actual_P30_lastrow_SFD48_bitwise=True,whole_prefix_rows_qualified=True,complete_layer_math_qualified=False,original_own_state_reference_qualified=False);write(a.output/'report.json',result)

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
 q=sub.add_parser('prepare');
 for name in ('prepared','engine-root','model-identity','output'):q.add_argument('--'+name,type=Path,required=True)
 q=sub.add_parser('run');
 for name in ('plan','pre-health','output'):q.add_argument('--'+name,type=Path,required=True)
 q.add_argument('--group',default='prefix30');q.add_argument('--one-card-receipt',type=Path)
 q=sub.add_parser('finalize');
 for name in ('output','post-health','model-identity','lifecycle-receipt','input-receipt'):q.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();{'prepare':prepare,'run':run,'finalize':finalize}[a.mode](a)
if __name__=='__main__':main()
