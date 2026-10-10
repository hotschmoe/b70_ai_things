#!/usr/bin/env python3
"""NUM10 source35/C113 matched OFF/ON real final T1 layer0/GDN capture.
No old source29/reference21 receipt transfer or operator/full-math qualification.
"""
import argparse,array,collections,copy,hashlib,json,math,os,re,shlex,signal,struct,subprocess,time
from pathlib import Path
import serial_prefix_qualification_v6 as base
import c1_serve_controller_combined_v13 as c1
import qualify_c1_serving_combined_v13 as c113
from audit_layer0_capture_lifecycle_v2 import live_blob_bindings
from audit_v6_actual_request_records import strengthen
from merged_numerical_protocol_v2 import MergedProtocol
from source_page_watchdog_v3 import guard as page_guard
ROOT=Path(__file__).resolve().parents[2]
sha=base.sha;read=base.read;write=base.write;require=base.require
PLAN_SOURCE='strata/flash-next/current-ple-prompt35-engine-build-plan-v1.json'
PLAN_SOURCE_SHA='82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7'
PREFIXES=[1,2,4,8]
CONTRACT_SHA='86a83df856829674f03b9eb6c80bac0817cb2cdfa4459b3186f5480600e8dfdd'
expected_stage_ranges=base.expected_stage_ranges
DEPENDENCIES=['layer0_numerical_contract_v1.hpp','audit_layer0_capture_lifecycle_v2.py','verify_layer0_packets_v3.py','independent_q8_1_activation_v1.py','merged_numerical_protocol_v2.py','audit_v6_actual_request_records.py','source_page_watchdog_v3.py','run_source_upload_oracle_full_v2.py','c1_serve_controller_combined_v13.py','qualify_c1_serving_combined_v13.py','c1-combined-v13-source-plan.json','c1-combined-v13-parent-source-plan.json','current-ple-prompt35-engine-build-plan-v1.json','serial_prefix_prompt_fixture_v6.py','serial_prefix_qualification_v6.py','audit_fidelity_observer_coverage.py']

def verify_model_identity(path,lock,shards):
 identity=base.verify_model_identity(path,lock,shards);identity['current_known_pages']=page_guard(shards[2]);return identity

def engine_binding(engine,source_plan_sha):
 require(source_plan_sha==PLAN_SOURCE_SHA,'NUM10 exact source35 plan required')
 c1.combined_generation_gate(engine);return read(Path(engine)/'receipt.json')

NUMERICAL_BASELINE_OFF=('STRATA_BATCH_FULL_STATE_CHAIN','STRATA_BATCH_PUBLIC_PREFIX','STRATA_BATCH_FIDELITY_DIAG','STRATA_SLOT_OWNER_TRACE','STRATA_MIRROR_OWNER_TRACE','STRATA_PREFIX30','STRATA_PLE_INPUT33','STRATA_LAYER0_Q8_DIAG')
def numerical_baseline_gate(env):
 require(all(env.get(name,'0')=='0' for name in NUMERICAL_BASELINE_OFF),'NUM10 baseline source35 observers/cache flags must be OFF')
 require('STRATA_VERIFY_EAGER' not in env and 'STRATA_CKPT_REREAD' not in env and 'STRATA_STATE_HASH' not in env,'No eager/reread/statehash presence including0')
 require(env.get('STRATA_QFUSE','0')=='0' and env.get('STRATA_ONE_TOKEN_COMMIT','1')=='1','NUM10 normal native T1 selfcommit required')

API_PYTHON_SOURCES=('serve/server.py','serve/frontend.py','serve/artifact_identity.py','serve/batch_request_identity.py','tools/strata_tokenizer.py','tools/gguf_reader.py')

def api_source_binding(engine,manifest):
 sources=manifest['runtime']['python_sources']
 require(set(sources)==set(API_PYTHON_SOURCES),'Actual complete6 API source fingerprint roster differs')
 for name in API_PYTHON_SOURCES:require(sources[name]==sha(engine/'source'/name),'Actual API22 complete helper/source fingerprint absent '+name)
 return sources


validate_final_source_proof=c113.validate_final_source_proof

def candidate_binding(plan):
 engine=Path(plan['engine_root']);engine_binding(engine,plan.get('source_plan_sha256',PLAN_SOURCE_SHA));prepared=Path(plan['prepared']);raw=read(prepared/'prepared.json')
 c113.source_pins();c1.metadata_admission_gate(raw);proof=validate_final_source_proof(prepared,raw)
 generation=raw['combined_generation'];require(generation.get('controller_generation')==13 and generation.get('semantic_current_PLE_gather_fix32') is True and generation.get('final_window_PLE_observer_fix34') is True and generation.get('prompt_verifier_P30_capture35') is True,'Genuine corrected source35/C113 generation required')
 metadata=c1.validate_prepared(prepared);base.matched_upload_engine_gate(metadata,engine);numerical_baseline_gate(read(prepared/'server-config.json')['env'])
 require(plan['engine_receipt_sha256']==sha(engine/'receipt.json')==metadata['engine_receipt_sha256'] and plan['prepared_sha256']==sha(prepared/'prepared.json'),'Actual NUM10 engine/preparation association differs')
 return {'generation':generation,'C113_final_source_proof':proof,'engine_receipt_sha256':plan['engine_receipt_sha256'],'source_plan_sha256':PLAN_SOURCE_SHA}

def manifest_binding(plan):
 require(plan['schema']==10 and plan['driver_sha256']==sha(Path(__file__)) and plan['source_plan_sha256']==PLAN_SOURCE_SHA,'NUM10 driver/source generation differs')
 for name,want in plan['dependency_sha256'].items():require(sha(ROOT/name)==want,'NUM10 immutable dependency changed '+name)
 base.geometry_contract(plan['args']);require(plan['prefixes']=={str(n):plan['tokens']['cases']['A'][:n] for n in PREFIXES},'NUM10 exact accepted prefix roster differs')
 return candidate_binding(plan)

def prepare(a):
 engine=a.engine_root.resolve();prepared=a.prepared.resolve();preflight={'engine_root':str(engine),'prepared':str(prepared),'engine_receipt_sha256':sha(engine/'receipt.json'),'prepared_sha256':sha(prepared/'prepared.json')};candidate_binding(preflight)
 metadata=c1.validate_prepared(prepared);args=list(read(prepared/'server-config.json')['args'])
 for key,value in base.GEOMETRY.items():base.option(args,key,value)
 require('--mtp' not in args and '--pipeline-windows' not in args and '--no-prefill-borrow' in args,'NUM10 fixed serial normal-shape route required')
 code=ROOT/'strata/flash-next/serial_prefix_prompt_fixture_v6.py'
 tokens=json.loads(subprocess.check_output(['docker','run','--rm','--network','none','--user','1000:1000','-v',str(engine/'source')+':/src:ro','-v',metadata['pack']+':/pack:ro','-v',str(code)+':/fixture.py:ro',metadata['runtime']['image'],'exec /opt/b70-c1-python/bin/python /fixture.py'],text=True,timeout=90));require(tokens['vocab']==248320 and tokens['tokenizer_sha256']==read(prepared/'artifact-identity.json')['tokenizer_files'],'NUM10 actual tokenizer fixture differs');base.option(args,'--turn-token',tokens['turn_token'])
 identity=verify_model_identity(a.model_identity,read(ROOT/'strata/flash-next/model-lock.json'),[Path(row['path']) for row in metadata['model_shards']])
 plan={**preflight,'schema':10,'driver_sha256':sha(Path(__file__)),'source_plan_sha256':PLAN_SOURCE_SHA,'dependency_sha256':{'strata/flash-next/'+name:sha(ROOT/'strata/flash-next'/name) for name in DEPENDENCIES},'tokens':tokens,'prefixes':{str(n):tokens['cases']['A'][:n] for n in PREFIXES},'cards':metadata['cards'],'args':args,'image':metadata['runtime']['image'],'model_identity':identity,'primary_model_name':'hotschmoe-dd','research_alias':base.pilot_alias(metadata['cards'],args)+'-NUM10-source35-C113-L0-33-off-on','max_new':1,'observed_fields':33,'source_value_fields':31,'derived_value_fields':2,'full_model_math_qualified':False,'reference29_or21_runtime_transferred':False}
 manifest_binding(plan);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',plan);print('PREPARED NUM10 genuine source35 prerequisites; GPU executed false')

def arm_prepared_binding(plan,name,engine_root,prepared_directory,layer0):
 require(name in ('candidatecombined_off','candidatecombined_on') and layer0==(name=='candidatecombined_on'),'NUM10 same-source35 OFF/ON arms only')
 require(Path(engine_root).resolve()==Path(plan['engine_root']).resolve() and Path(prepared_directory).resolve()==Path(plan['prepared']).resolve(),'NUM10 actual arm engine/preparation differs')
 candidate_binding(plan);return c1.validate_prepared(prepared_directory)

def field_contract():
 source=ROOT/'strata/flash-next/layer0_numerical_contract_v1.hpp';require(sha(source)==CONTRACT_SHA,'Frozen33 layout differs');fields=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',source.read_text());require(len(fields)==33,'Layout malformed')
 return [{'name':name,'encoding':encoding,'bytes':int(size),'observed':True,'provenance':'DERIVED_gpu_native_exp_from_actual_gate_up' if index in (25,29) else 'actual_buffer'} for index,(name,encoding,size) in enumerate(fields)]


def collect_frame(raw,directory,binding):
 frames=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in raw['stderr'] if s.startswith('L0Q8 frame ')];require(len(frames)==1,'Actual frame marker absent/ambiguous')
 line=frames[0];path=directory/'layer0'/Path(line['metadata']).name;require(path.is_file() and not path.is_symlink(),'Actual metadata missing/symlink');frame=read(path)
 starts=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in raw['stderr'] if s.startswith('SFD request ')];require(len(starts)==1,'Actual0013 matched request absent');pid=int(starts[0]['pid']);ordinal=int(starts[0]['request'])
 require(frame['schema']==1 and frame['binding_sha256']==binding and frame['gen_ids']==raw['ids'] and frame['pid']==pid and frame['request']==ordinal and int(line['pid'])==pid and int(line['request'])==ordinal,'Source/process/request binding differs')
 require(frame['stage']==0 and frame['layer']==0 and frame['rows']==1 and frame['position']==len(raw['ids'])-1 and frame['token']==raw['ids'][-1] and frame['reused']==0,'Token/state boundary differs')
 require(frame['request_replay_marker_verified'] is True and frame['completed_nonce']==(((pid&0xffffffff)<<32)|ordinal) and frame['graph_key'] in (2,3),'Current GPU combined observer nonce/key absent')
 require(frame['full_model_math_qualified'] is False and frame['raw_fused_hidden_observed'] is False and frame['complete_preregistered_layout'] is True and frame.get('producer_mapping_verified') is True,'33-field provenance/producer coverage claim differs')
 contract=field_contract();require(len(frame['fields'])==33,'33-field quota differs');observed=[];seen=set()
 for actual,want in zip(frame['fields'],contract):
  require(all(actual[k]==want[k] for k in ['name','encoding','bytes','observed','provenance']),'Field shape/derived provenance differs');file=directory/'layer0'/Path(actual['file']).name
  require(file.is_file() and not file.is_symlink() and file.stat().st_size==want['bytes'] and file.name not in seen,'Field missing/truncated/reused');seen.add(file.name);data=file.read_bytes();require(len(data)==want['bytes'] and sha(file)==hashlib.sha256(data).hexdigest(),'Actual field changed/truncated during read')
  if want['encoding'].startswith('LE_F32'):require(all(math.isfinite(x) for x in struct.unpack('<'+'f'*(len(data)//4),data)),'Nonfinite source/derived observation')
  observed.append({'name':want['name'],'path':str(file),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'encoding':want['encoding'],'provenance':want['provenance']})
 fields={f['name']:f for f in observed};ids=struct.unpack('<10i',Path(fields['router_ids']['path']).read_bytes());require(len(set(ids))==10 and all(0<=i<512 for i in ids),'Actual router IDs invalid')
 mapping=struct.unpack('<30i',Path(fields['expert_entry_map']['path']).read_bytes());require(all(mapping[3*i:3*i+3]==(ids[i],0,i) for i in range(10)),'Actual ent_tok/dst/expert-ID mapping differs')
 pointers=frame['expert_blob_addresses'];tiers=frame['expert_tiers'];require(len(pointers)==len(tiers)==10 and len(set(pointers))==10 and all(type(p) is int and p>0 and p%4==0 for p in pointers) and all(t in (1,2) for t in tiers),'Producer pointer/tier roster invalid')
 live=live_blob_bindings((directory/'engine.combined.log').read_text(errors='replace'),pid,ordinal,pointers,tiers)
 return {'metadata':str(path),'metadata_sha256':sha(path),'frame':frame,'observed':observed,'source_value_fields':31,'derived_value_fields':2,'unobserved_original_fused_hidden':True,'source_live_binding':live,'full_model_math_qualified':False}


def compare_numeric(left,right):
 a,b=left['raw'],right['raw'];require(a['ids']==b['ids'] and a['fresh']==b['fresh']==1,'Numerical prefix/fresh pairing differs')
 require(a['output_ids']==b['output_ids'] and a['LP']==b['LP'] and a['done'].split()[5]==b['done'].split()[5],'Numerical bounded generation/LP/finish differs')
 first=base.compare_vectors(left['meta']['logits'][0],right['meta']['logits'][0]);require(first['bitwise_equal'],'Matched full first-head differs')
 x={int(v['layer']):v for v in left['meta']['residuals']};y={int(v['layer']):v for v in right['meta']['residuals']};require(sorted(x)==sorted(y)==list(range(48)),'Numerical complete first-window residual coverage differs')
 residuals={str(i):base.compare_vectors(x[i],y[i]) for i in range(48)};require(all(v['bitwise_equal'] for v in residuals.values()),'Matched first-window layer residual differs')
 return {'bounded_output_ids_lp_finish_equal':True,'head':first,'residuals':residuals,'natural_completion_required':False,'full_model_math_qualified':False}


def run_process(plan,output,name,engine_root,prepared_directory,layer0):
 directory=output/name;directory.mkdir();(directory/'captures').mkdir();(directory/'layer0').mkdir();prepared=arm_prepared_binding(plan,name,engine_root,prepared_directory,layer0);env=dict(read(prepared_directory/'server-config.json')['env']);binding=sha(output/'plan.snapshot.json')
 numerical_baseline_gate(env)
 require('STRATA_VERIFY_EAGER' not in env and env.get('STRATA_QFUSE','0')=='0' and env.get('STRATA_ONE_TOKEN_COMMIT','1')!='0','Unqualified eager/QFUSE/self-commit mode')
 env.update({'STRATA_FIDELITY_DIAG':'1','STRATA_FIDELITY_DIAG_ACTIVATIONS':'1','STRATA_FIDELITY_DIAG_ARM':'/results/ARM','STRATA_FIDELITY_DIAG_DIR':'/results/captures','STRATA_PREFIX_DIAG':'1','STRATA_PREFIX_LIFECYCLE_DIAG':'1','STRATA_PREFIX_DIAG_ARM':'/results/ARM','STRATA_LAYER0_Q8_DIAG':str(int(layer0)),'STRATA_LAYER0_Q8_DIAG_DIR':'/results/layer0','STRATA_LAYER0_Q8_DIAG_ARM':'/results/ARM','STRATA_LAYER0_Q8_DIAG_BINDING_SHA256':binding,'STRATA_ARTIFACT_IDENTITY_SHA256':binding,'STRATA_BATCH_FIDELITY_DIAG':'0','SYCL_UR_TRACE':'2'})
 command=['docker','run','-i','--name','b70-prefix-'+str(os.getpid())+'-'+name,'--label','b70.prefix.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine_root/'build')+':/build:ro','-v',str(engine_root/'source')+':/src:ro','-v',prepared['pack']+':/pack:ro','-v',str(ROOT/read(ROOT/'strata/flash-next/model-lock.json')['destination'])+':/model:ro','-v',str(directory)+':/results']
 for key,value in sorted(env.items()):command+=['-e',key+'='+value]
 command +=[plan['image'],'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+plan['args'])];write(directory/'command.json',command);protocol=None;rows=[];error=None
 try:
  protocol=MergedProtocol(command,directory)
  for key in ['A','B','C']:protocol.request('warm-'+key,plan['tokens']['cases'][key],fresh=1,max_new=1)
  (directory/'ARM').write_text('ARM fixed warmups complete\n',encoding='ascii')
  for number,n in enumerate(PREFIXES,1):
   require(len(rows)<4,'Raw frame request quota exceeded before submission');raw=protocol.request('prefix-'+str(n),plan['prefixes'][str(n)],fresh=1,max_new=1);meta=base.extract(raw,directory/'captures',True,True,expected_stage_ranges(plan['args']));external=strengthen(raw,expected_stage_ranges(plan['args']),True,True);frame=collect_frame(raw,directory,binding) if layer0 else None
   require(raw['done'].split()[5] in {'length','stop'} and not meta['ledger']['cancelled'],'Numerical request did not complete')
   if frame:require(frame['frame']['request']==number,'Layer0 warmup/ordinal quota differs')
   else:require(not any(line.startswith('L0Q8 frame ') for line in raw['stderr']),'Layer0 flag-off published a frame')
   rows.append({'prefix':n,'raw':raw,'meta':meta,'external':external,'layer0':frame});write(directory/'requests.json',rows)
 except Exception as exception:error=str(exception)
 finally:
  try:rc=protocol.close() if protocol else None
  except Exception as exception:rc=None;error=(error+'; ' if error else '')+'protocol close: '+str(exception)
  container='b70-prefix-'+str(os.getpid())+'-'+name;state=c1.inspected(container)['State'];removed=False
  if not state['Running']:
   subprocess.run(['docker','rm',container],capture_output=True,text=True,check=True);removed=c1.absent(container)
  result={'passed':error is None and rc==0 and state['ExitCode']==0 and not state.get('OOMKilled') and removed,'error':error,'engine_rc':rc,'state':state,'removed':removed,'complete_four_prefix_roster':len(rows)==4,'producer_fd_merge_before_emit':True,'canonical_combined_log_sha256':sha(directory/'engine.combined.log') if (directory/'engine.combined.log').exists() else None,'full_model_math_qualified':False};write(directory/'result.json',result)
 return result


def run(a):
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);health=read(a.pre_health);require(health['passed'] and set(plan['cards'])<=set(health['cards']) and len(health.get('files',[]))==2 and 0<=time.time()-health['finished_epoch']<=300,'NUM10 strict/compiled fresh prehealth required')
 for row in health['files']:require(sha(row['path'])==row['sha256'],'NUM10 health producer evidence changed')
 if len(plan['cards'])>1:
  require(a.one_card_receipt is not None,'Same NUM10 source35 onecard prerequisite required');one=read(a.one_card_receipt);require(one.get('schema')==10 and one.get('driver_sha256')==sha(Path(__file__)) and one.get('passed') is True and one.get('post_health_passed') is True and one.get('packet_contract_qualified') is True and one.get('layer0_logical_lifecycle_qualified') is True and one['engine_receipt_sha256']==plan['engine_receipt_sha256'] and one['cards']==[0],'Old NUM9 or foreign engine onecard proof refused')
 if len(plan['cards'])>1:
  from validate_layer0_numerical_final_v10 import finalized_binding
  _,current_one,_,_=finalized_binding(Path(a.one_card_receipt).resolve().parents[1]);require(current_one==one,'NUM10 onecard current public recollection differs')
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.snapshot.json',plan);results=[];comparisons=[];error=None
 def stop(sig,frame):raise InterruptedError('Parent requested NUM10 owned stop '+str(sig))
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
 try:
  for name,on in [('candidatecombined_off',False),('candidatecombined_on',True)]:
   result=run_process(plan,a.output,name,Path(plan['engine_root']),Path(plan['prepared']),on);results.append(result);require(result['passed'] and result['complete_four_prefix_roster'],'NUM10 actual arm failed '+str(result['error']))
  off=read(a.output/'candidatecombined_off/requests.json');on=read(a.output/'candidatecombined_on/requests.json')
  for left,right in zip(off,on):require(left['prefix']==right['prefix'],'NUM10 matched prefix roster differs');comparisons.append({'prefix':left['prefix'],'same_source35_off_on':compare_numeric(left,right)})
 except Exception as exception:error=str(exception)
 write(a.output/'report.json',{'schema':10,'driver_sha256':sha(Path(__file__)),'passed':False,'numerical_and_teardown_passed':error is None and len(comparisons)==4,'post_health_passed':False,'error':error,'cards':plan['cards'],'plan_sha256':sha(a.output/'plan.snapshot.json'),'engine_receipt_sha256':plan['engine_receipt_sha256'],'finished_epoch':time.time(),'results':results,'comparisons':comparisons,'full_model_math_qualified':False,'natural_completion_or_api_or_latency_claim':False,'layer0_logical_lifecycle_qualified':False,'reference29_or21_runtime_transferred':False})
 if error:raise SystemExit(1)

def post_hash_boundary(identity,result,health):
 times=[identity['started'],result['finished_epoch'],health['finished_epoch']]
 require(all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in times),'Final source chronology invalid')
 require(times[0]>=max(times[1:]),'Actual complete post-run hash must start after child terminal and post-health')
 return max(times[1:])


def finalize(a):
 result=read(a.output/'report.json');health=read(a.post_health);require(result['numerical_and_teardown_passed'] and health['passed'] and set(result['cards'])<=set(health['cards']) and health['finished_epoch']>=result['finished_epoch'],'Parent numerical/teardown/post-health gate failed')
 plan=read(a.output/'plan.snapshot.json');manifest_binding(plan);require(result.get('schema')==10 and result.get('driver_sha256')==sha(Path(__file__)) and result['engine_receipt_sha256']==plan['engine_receipt_sha256'] and result['plan_sha256']==sha(a.output/'plan.snapshot.json'),'NUM10 final report/plan/driver association differs');prepared=c1.validate_prepared(Path(plan['prepared']));identity=verify_model_identity(a.model_identity,read(ROOT/'strata/flash-next/model-lock.json'),[Path(row['path']) for row in prepared['model_shards']]);post_hash_boundary(read(a.model_identity),result,health)
 require(a.packet_receipt is not None,'Actual four-prefix independent packet check required');packet=read(a.packet_receipt);require(packet.get('passed') and packet.get('plan_sha256')==result['plan_sha256'],'Actual packet evidence absent/wrong generation');require(packet.get('source_value_fields')==31 and packet.get('derived_fields')==2 and packet.get('raw_fused_hidden_proven') is False and packet.get('original_weight_consumers_qualified') is False and packet.get('full_model_math_qualified') is False and [r['prefix'] for r in packet['results']]==PREFIXES,'Packet scope/count differs');require(packet.get('requests_sha256')==sha(a.output/'candidatecombined_on/requests.json'),'Packet inputs differ')
 require(a.lifecycle_receipt is not None,'Current layer0 target allocation/owning context/logical free receipt required');life=read(a.lifecycle_receipt);require(life.get('passed') and life.get('plan_sha256')==result['plan_sha256'],'Layer0 logical lifecycle receipt missing/wrong generation');require(len(life['owners'])==1 and life['owners'][0]['bytes']==7023304 and life.get('postfree_type_gate_used') is False and life.get('log_sha256')==sha(a.output/'candidatecombined_on/engine.combined.log'),'Actual complete256-control target lifecycle differs')
 # Recollect current exact raw fields/producer associations and both native arms.
 recollected={}
 for name,on in [('candidatecombined_off',False),('candidatecombined_on',True)]:
  directory=a.output/name;rows=read(directory/'requests.json');require([row['prefix'] for row in rows]==PREFIXES,'NUM10 current prefix roster differs')
  for row in rows:
   require(row['raw']['ids']==plan['prefixes'][str(row['prefix'])] and row['raw']['fresh']==1,'NUM10 current accepted IDs/fresh differs')
   meta=base.extract(row['raw'],directory/'captures',True,True,expected_stage_ranges(plan['args']));external=strengthen(row['raw'],expected_stage_ranges(plan['args']),True,True)
   require(json.loads(json.dumps(meta))==row['meta'] and json.loads(json.dumps(external))==row['external'],'NUM10 current SFD/actual request data changed')
   if on:require(json.loads(json.dumps(collect_frame(row['raw'],directory,result['plan_sha256'])))==row['layer0'],'NUM10 current L0 fields/nonce/layout/live pointers changed')
   else:require(not any(line.startswith('L0Q8 frame ') for line in row['raw']['stderr']),'NUM10 OFF emitted L0 frame')
  recollected[name]=rows
 fresh=[{'prefix':left['prefix'],'same_source35_off_on':compare_numeric(left,right)} for left,right in zip(recollected['candidatecombined_off'],recollected['candidatecombined_on'])];require(fresh==result['comparisons'],'NUM10 current OFF/ON comparisons changed')
 from verify_layer0_packets_v3 import verify as verify_packets
 current=verify_packets(recollected['candidatecombined_on']);current.update(plan_sha256=result['plan_sha256'],requests_sha256=sha(a.output/'candidatecombined_on/requests.json'));require(current==packet,'NUM10 packet receipt differs from current supplied fields')
 from audit_layer0_capture_lifecycle_v2 import audit_text
 current_life=audit_text((a.output/'candidatecombined_on/engine.combined.log').read_text());current_life.update(plan_sha256=result['plan_sha256'],log_sha256=sha(a.output/'candidatecombined_on/engine.combined.log'));require(json.loads(json.dumps(current_life))==life,'NUM10 current owning-free/lifecycle proof differs')
 result.update(passed=True,post_health_passed=True,post_model_identity=identity,layer0_logical_lifecycle_qualified=True,lifecycle_receipt_sha256=sha(a.lifecycle_receipt),packet_contract_qualified=True,packet_receipt_sha256=sha(a.packet_receipt),original_own_state_fp64_qualified=False,complete_layer_math_qualified=False);write(a.output/'report.json',result)


def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
 p=sub.add_parser('prepare')
 for name in ('prepared','engine-root','model-identity','output'):p.add_argument('--'+name,type=Path,required=True)
 p=sub.add_parser('run')
 for name in ('plan','pre-health','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--group',default='numerical');p.add_argument('--one-card-receipt',type=Path)
 p=sub.add_parser('finalize')
 for name in ('output','post-health','model-identity','lifecycle-receipt','packet-receipt'):p.add_argument('--'+name,type=Path,required=True)
 a=parser.parse_args();{'prepare':prepare,'run':run,'finalize':finalize}[a.mode](a)
if __name__=='__main__':main()
