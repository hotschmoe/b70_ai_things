#!/usr/bin/env python3
"""Combined33-field raw-prefix numerical orchestration versus qualified frozen21.
No API/natural-completion/latency/full-math claim. Parent owns GPU lifecycle.
"""
import argparse,array,collections,copy,hashlib,json,math,os,re,shlex,signal,struct,subprocess,time
from pathlib import Path
import serial_prefix_qualification_v6 as base
import c1_serve_controller_combined_v3 as candidate_c1
import qualify_layer0_numerical_v2 as reference_parent
from audit_layer0_capture_lifecycle_v2 import live_blob_bindings
from audit_v6_actual_request_records import strengthen
from merged_numerical_protocol_v2 import MergedProtocol
from source_page_watchdog_v3 import guard as page_guard
ROOT=Path(__file__).resolve().parents[2]
c1=candidate_c1;sha=base.sha;read=base.read;write=base.write;require=base.require
PLAN_SOURCE='strata/flash-next/combined-numerical-batch-engine-build-plan-v1.json'
PLAN_SOURCE_SHA='8d708bff8f59b1133f5156969e6a59cf8dee4a949637d75e850f58ede63d9d6a'
PATCH21='0021-sycl-bounded-layer0-numerical-capture.patch'
PATCH21_SHA='3c282ced590a5a71e5354c22438728cf326cbf78201497cdd1de708ce0f51081'
PREFIXES=[1,2,4,8]
CONTRACT_SHA='86a83df856829674f03b9eb6c80bac0817cb2cdfa4459b3186f5480600e8dfdd'
expected_stage_ranges=base.expected_stage_ranges
def verify_model_identity(path,lock,shards):
 identity=base.verify_model_identity(path,lock,shards);identity['current_known_pages']=page_guard(shards[2]);return identity


def engine_binding(engine,source_plan_sha):
 source=read(ROOT/PLAN_SOURCE);require(sha(ROOT/PLAN_SOURCE)==PLAN_SOURCE_SHA and source_plan_sha==PLAN_SOURCE_SHA,'Combined source plan identity differs')
 receipt=read(engine/'receipt.json');require(receipt.get('build_rc')==0 and receipt.get('external_source_unchanged') is True and receipt.get('plan_snapshot_unchanged') is True and receipt.get('plan_sha256')==PLAN_SOURCE_SHA and receipt.get('image')==c1.BASE_IMAGE,'Actual combined SDK build receipt absent')
 require(receipt.get('patched_source_sha256')==source['expected_patched_source_sha256'],'Actual complete51 source ledger differs')
 require([(Path(x['path']).name,x['sha256']) for x in receipt['patches']]==[(Path(x['path']).name,x['sha256']) for x in source['patches']],'Actual ordered24 patch chain differs')
 for name,digest in source['expected_patched_source_sha256'].items():require(sha(engine/'source'/name)==digest,'Compiled source changed '+name)
 for header in source['added_header_payloads']:require(header['sha256']==source['expected_patched_source_sha256'][header['path']],'Consumed header manifest contradicts final source')
 require(receipt.get('commands') and 'icpx --version' in receipt.get('container_script','') and all(t in receipt['container_script'] for t in source['build_targets']),'Actual SDK full-target command evidence absent')
 for target in source['build_targets']:
  binary=engine/'build'/target;require(binary.is_file() and receipt['binary_sha256'].get(str(binary))==sha(binary),'Actual rebuilt ABI target missing '+target)
  with binary.open('rb') as f:require(f.read(4)==b'\x7fELF','CPU mocked file is not a genuine rebuilt executable')
 return receipt


API_PYTHON_SOURCES=('serve/server.py','serve/frontend.py','serve/artifact_identity.py','serve/batch_request_identity.py','tools/strata_tokenizer.py','tools/gguf_reader.py')

def api_source_binding(engine,manifest):
 sources=manifest['runtime']['python_sources']
 require(set(sources)==set(API_PYTHON_SOURCES),'Actual complete6 API source fingerprint roster differs')
 for name in API_PYTHON_SOURCES:require(sources[name]==sha(engine/'source'/name),'Actual API22 complete helper/source fingerprint absent '+name)
 return sources


def candidate_binding(plan):
 engine=Path(plan['engine_root']);receipt=engine_binding(engine,PLAN_SOURCE_SHA)
 prepared=c1.validate_prepared(Path(plan['prepared']));base.matched_upload_engine_gate(prepared,engine)
 require(plan['engine_receipt_sha256']==sha(engine/'receipt.json')==prepared['engine_receipt_sha256'] and plan['prepared_sha256']==sha(Path(plan['prepared'])/'prepared.json'),'Genuine candidate SDK/C1/full390 preparation differs')
 path=Path(plan['candidate_c1_qualification']);qualified=read(path)
 require(path.parent.resolve()==Path(plan['prepared']).resolve() and sha(path)==plan['candidate_c1_qualification_sha256'] and qualified.get('passed') is True and qualified.get('teardown_passed') is True and qualified.get('post_health_passed') is True and qualified.get('engine_receipt_sha256')==plan['engine_receipt_sha256'],'Actual combined C1 qualification missing; CPU mock positives do not prepare')
 manifest=read(Path(plan['prepared'])/'artifact-identity.json');sources=api_source_binding(engine,manifest)
 return {'source_plan_sha256':PLAN_SOURCE_SHA,'engine_receipt_sha256':sha(engine/'receipt.json'),'executable_sha256':sha(engine/'build/strata'),'prepared_sha256':plan['prepared_sha256'],'actual_c1_qualification_sha256':sha(path),'api_python_sources':sources,'source_gate':'genuine combined SDK+source390+C1 only; no old or mocked receipt substitution'}


def manifest_binding(plan):
 require(plan['driver_sha256']==sha(Path(__file__)),'V3 driver changed')
 for path,want in plan['dependency_sha256'].items():require(sha(ROOT/path)==want,'Validation dependency changed '+path)
 candidate=candidate_binding(plan);reference=read(Path(plan['reference_plan']));require(sha(Path(plan['reference_plan']))==plan['reference_plan_sha256'],'Frozen21V2 reference changed');reference_parent.prepared_chain_binding(reference)
 qualification=read(Path(plan['reference_qualification']));parent=read(Path(plan['reference_parent_qualification']))
 require(sha(Path(plan['reference_qualification']))==plan['reference_qualification_sha256'] and qualification.get('passed') is True and qualification.get('numerical_and_teardown_passed') is True and qualification.get('post_health_passed') is True and qualification.get('layer0_logical_lifecycle_qualified') is True and qualification['engine_receipt_sha256']==reference['engine_receipt_sha256'],'Actual finalized21V2 child numerical reference absent')
 require(sha(Path(plan['reference_parent_qualification']))==plan['reference_parent_qualification_sha256'] and parent.get('passed') is True and parent.get('post_health_passed') is True and parent.get('owned_containers_terminal') is True and parent['plan_sha256']==plan['reference_plan_sha256'],'Actual21V2 parent health/hash/owned teardown reference absent')
 require(plan['cards']==reference['cards'] and plan['args']==reference['args'],'Matched reference placement/context/precision/cuts differ')
 refenv=read(Path(reference['prepared'])/'server-config.json')['env'];env=read(Path(plan['prepared'])/'server-config.json')['env']
 math_env=lambda values:{k:v for k,v in values.items() if k not in {'STRATA_ARTIFACT_IDENTITY_SHA256','STRATA_BATCH_FIDELITY_DIAG'}}
 require(math_env(refenv)==math_env(env),'Effective matched arithmetic environment differs')
 return {'candidate':candidate,'reference_engine_receipt_sha256':reference['engine_receipt_sha256'],'reference_qualification_sha256':plan['reference_qualification_sha256'],'reference_parent_qualification_sha256':plan['reference_parent_qualification_sha256'],'genuine_prepared':True}


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
  require(file.is_file() and not file.is_symlink() and file.stat().st_size==want['bytes'] and file.name not in seen,'Field missing/truncated/reused');seen.add(file.name);data=file.read_bytes()
  if want['encoding'].startswith('LE_F32'):require(all(math.isfinite(x) for x in struct.unpack('<'+'f'*(len(data)//4),data)),'Nonfinite source/derived observation')
  observed.append({'name':want['name'],'path':str(file),'sha256':sha(file),'bytes':len(data),'encoding':want['encoding'],'provenance':want['provenance']})
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


def prepare(a):
 require(sha(ROOT/PLAN_SOURCE)==PLAN_SOURCE_SHA,'Combined source plan changed');engine=engine_binding(a.engine_root,PLAN_SOURCE_SHA)
 reference=read(a.reference_plan);reference_parent.prepared_chain_binding(reference)
 prepared=c1.validate_prepared(a.prepared);base.matched_upload_engine_gate(prepared,a.engine_root)
 require(prepared['cards']==reference['cards'],'Candidate/reference physical device placement differs')
 plan={'schema':1,'status':'prepared numerical source lane only; no actual execution/full math claim','prepared':str(a.prepared.resolve()),'prepared_sha256':sha(a.prepared/'prepared.json'),'engine_root':str(a.engine_root.resolve()),'engine_receipt_sha256':sha(a.engine_root/'receipt.json'),'reference_plan':str(a.reference_plan.resolve()),'reference_plan_sha256':sha(a.reference_plan),'reference_qualification':str(a.reference_qualification.resolve()),'reference_qualification_sha256':sha(a.reference_qualification),'model_identity':verify_model_identity(a.model_identity,read(ROOT/'strata/flash-next/model-lock.json'),[Path(row['path']) for row in prepared['model_shards']]),'cards':reference['cards'],'args':reference['args'],'tokens':reference['tokens'],'image':prepared['runtime']['image'],'driver_sha256':sha(Path(__file__)),'primary_model_name':'hotschmoe-dd','research_alias':reference['research_alias']+'-numerical31source2derived-combined','prefixes':{str(n):reference['tokens']['cases']['A'][:n] for n in PREFIXES},'source_plan_sha256':PLAN_SOURCE_SHA,'candidate_c1_qualification':str(a.candidate_c1_qualification.resolve()),'candidate_c1_qualification_sha256':sha(a.candidate_c1_qualification),'reference_parent_qualification':str(a.reference_parent_qualification.resolve()),'reference_parent_qualification_sha256':sha(a.reference_parent_qualification),'dependency_sha256':{path:sha(ROOT/path) for path in ['strata/flash-next/serial_prefix_qualification_v6.py','strata/flash-next/qualify_serial_prefix_v6.py','strata/flash-next/audit_v6_actual_request_records.py','strata/flash-next/audit_layer0_capture_lifecycle_v2.py','strata/flash-next/layer0_numerical_contract_v1.hpp','strata/flash-next/verify_layer0_packets_v3.py','strata/flash-next/independent_q8_1_activation_v1.py','strata/flash-next/merged_numerical_protocol_v2.py','strata/flash-next/source_page_watchdog_v3.py','strata/flash-next/qualify_layer0_numerical_v2.py','strata/flash-next/layer0_numerical_qualification_v2.py','strata/flash-next/combined-numerical-batch-engine-build-plan-v1.json','strata/flash-next/c1_serve_controller_combined_v3.py','strata/flash-next/c1-combined-v3-source-plan.json']},'max_new':1,'observed_fields':33,'source_value_fields':31,'derived_value_fields':2,'layout_fields':33,'full_model_math_qualified':False,'natural_completion_or_api_or_latency_claim':False}
 manifest_binding(plan);a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',plan);print('PREPARED',a.output/'plan.json','GPU_EXECUTED false')


def run_process(plan,output,name,engine_root,prepared_directory,layer0):
 directory=output/name;directory.mkdir();(directory/'captures').mkdir();(directory/'layer0').mkdir();prepared=c1.validate_prepared(prepared_directory);env=dict(read(prepared_directory/'server-config.json')['env']);binding=sha(output/'plan.snapshot.json')
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
 plan=read(a.plan);manifest_binding(plan);c1.leased([0,1]);health=read(a.pre_health);require(health['passed'] and set(plan['cards'])<=set(health['cards']) and 0<=time.time()-health['finished_epoch']<=300,'Fresh parent health required')
 require(health.get('files'),'Parent pre-health source commands/logs absent')
 for record in health['files']:require(sha(record['path'])==record['sha256'],'Parent pre-health source evidence changed')
 if len(plan['cards'])>1:
  require(a.one_card_receipt is not None,'Qualified samecombined one-card numerical/lifecycle control required before pair')
  one=read(a.one_card_receipt);require(one.get('passed') and one.get('post_health_passed') and one.get('layer0_logical_lifecycle_qualified') and one.get('engine_receipt_sha256')==plan['engine_receipt_sha256'] and one.get('cards')==[0],'Actual samecombined one-card qualification missing')
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.snapshot.json',plan)
 stopping=[False]
 def stop(sig,frame):
  if not stopping[0]:stopping[0]=True;raise InterruptedError('Parent requested numerical stop '+str(sig))
 for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stop)
 reference=read(Path(plan['reference_plan']));results=[];comparisons=[];error=None
 try:
  for name,engine,prepared,on in [('reference21',Path(reference['engine_root']),Path(reference['prepared']),False),('candidatecombined_off',Path(plan['engine_root']),Path(plan['prepared']),False),('candidatecombined_on',Path(plan['engine_root']),Path(plan['prepared']),True)]:
   result=run_process(plan,a.output,name,engine,prepared,on);results.append(result);require(result['passed'] and result['complete_four_prefix_roster'],'Numerical arm failed: '+str(result['error']))
  old=read(a.output/'reference21/requests.json');off=read(a.output/'candidatecombined_off/requests.json');on=read(a.output/'candidatecombined_on/requests.json')
  for x,y,z in zip(old,off,on):require(x['prefix']==y['prefix']==z['prefix'],'Actual numerical prefix roster differs');comparisons.append({'prefix':x['prefix'],'21_vs_combined_off':compare_numeric(x,y),'same_combined_off_on':compare_numeric(y,z)})
 except Exception as exception:error=str(exception)
 report={'schema':1,'passed':False,'numerical_and_teardown_passed':error is None and len(comparisons)==4,'post_health_passed':False,'error':error,'cards':plan['cards'],'plan_sha256':sha(a.output/'plan.snapshot.json'),'engine_receipt_sha256':plan['engine_receipt_sha256'],'finished_epoch':time.time(),'results':results,'comparisons':comparisons,'full_model_math_qualified':False,'natural_completion_or_api_or_latency_claim':False,'layer0_logical_lifecycle_qualified':False}
 write(a.output/'report.json',report)
 if error:raise SystemExit(1)


def finalize(a):
 result=read(a.output/'report.json');health=read(a.post_health);require(result['numerical_and_teardown_passed'] and health['passed'] and set(result['cards'])<=set(health['cards']) and health['finished_epoch']>=result['finished_epoch'],'Parent numerical/teardown/post-health gate failed')
 plan=read(a.output/'plan.snapshot.json');prepared=c1.validate_prepared(Path(plan['prepared']));identity=verify_model_identity(a.model_identity,read(ROOT/'strata/flash-next/model-lock.json'),[Path(row['path']) for row in prepared['model_shards']]);require(read(a.model_identity)['started']>=int(result['finished_epoch']),'Actual complete post-run hash started too early')
 require(a.packet_receipt is not None,'Actual four-prefix independent packet check required');packet=read(a.packet_receipt);require(packet.get('passed') and packet.get('plan_sha256')==result['plan_sha256'],'Actual packet evidence absent/wrong generation');require(packet.get('source_value_fields')==31 and packet.get('derived_fields')==2 and packet.get('raw_fused_hidden_proven') is False and packet.get('original_weight_consumers_qualified') is False and packet.get('full_model_math_qualified') is False and [r['prefix'] for r in packet['results']]==PREFIXES,'Packet scope/count differs');require(packet.get('requests_sha256')==sha(a.output/'candidatecombined_on/requests.json'),'Packet inputs differ')
 require(a.lifecycle_receipt is not None,'Current layer0 target allocation/owning context/logical free receipt required');life=read(a.lifecycle_receipt);require(life.get('passed') and life.get('plan_sha256')==result['plan_sha256'],'Layer0 logical lifecycle receipt missing/wrong generation');require(len(life['owners'])==1 and life['owners'][0]['bytes']==7023304 and life.get('postfree_type_gate_used') is False and life.get('log_sha256')==sha(a.output/'candidatecombined_on/engine.combined.log'),'Actual complete256-control target lifecycle differs')
 result.update(passed=True,post_health_passed=True,post_model_identity=identity,layer0_logical_lifecycle_qualified=True,lifecycle_receipt_sha256=sha(a.lifecycle_receipt),packet_contract_qualified=True,packet_receipt_sha256=sha(a.packet_receipt),original_own_state_fp64_qualified=False,complete_layer_math_qualified=False);write(a.output/'report.json',result)


def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
 p=sub.add_parser('prepare');p.add_argument('--reference-plan',type=Path,required=True);p.add_argument('--reference-qualification',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--engine-root',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--candidate-c1-qualification',type=Path,required=True);p.add_argument('--reference-parent-qualification',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p=sub.add_parser('run');p.add_argument('--plan',type=Path,required=True);p.add_argument('--pre-health',type=Path,required=True);p.add_argument('--group',default='numerical');p.add_argument('--one-card-receipt',type=Path);p.add_argument('--output',type=Path,required=True)
 p=sub.add_parser('finalize');p.add_argument('--output',type=Path,required=True);p.add_argument('--post-health',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--lifecycle-receipt',type=Path,required=True);p.add_argument('--packet-receipt',type=Path,required=True)
 a=parser.parse_args();{'prepare':prepare,'run':run,'finalize':finalize}[a.mode](a)
if __name__=='__main__':main()
