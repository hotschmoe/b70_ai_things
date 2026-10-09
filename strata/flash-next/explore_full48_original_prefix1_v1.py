#!/usr/bin/env python3
"""Readonly original own-prefix1 exploration against finalized V9 observations.
Metrics/first differences only; NO numerical PASS, tolerance or native math claim.
Actual model payload reads occur only when an operator explicitly executes CLI.
"""
import argparse,contextlib,hashlib,io,json,os,re,time,traceback
from pathlib import Path
import numpy as np
from full48_owned_composition_storage_v1 import BoundOriginalFull48Rows,Full48OwnedComposition
from original_math_scalar import metrics
from source_page_watchdog_v3 import guard,preserve,signature
from run_source_upload_oracle_full_v2 import full_buffered_identity
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
FOUNDATION=HERE/'original-gguf-reference-foundation-plan-v1.json'
SOURCE_PLAN=HERE/'integrated-slot-owner-trace-engine-build-plan-v1.json'
SOURCE_PLAN_SHA='94aef305dbc6a31f023afa55573d52a9ce6af9443cfc0182eb7c324255159094'
DRIVER_SHA='2252f24be9a3511bcb40d5e623caa75259112b9cd07f3d55944f67c67d2a084c'
PARENT_SHA='3900a306b60b0eb4146d5faf61ca6a473d60cfeef32bdfd4626f7e774abd465e'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
 if not ok:raise ValueError(message)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='ascii')


def identity_admission(path,lock_path,shards,after_epoch,current=True):
 value=read(path);lock=read(lock_path);expected=[r for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')]
 require(value.get('passed') is True and value['lock_sha256']==sha(lock_path) and value['model_revision']==lock['revision'],'Complete source identity lock/revision differs')
 require(len(expected)==len(shards)==len(value['rows'])==4 and value['started']>=after_epoch and value['finished']>=value['started'],'Complete source identity roster/chronology differs')
 for row,want,p in zip(value['rows'],expected,shards):
  require(Path(row['path']).resolve()==Path(p).resolve() and Path(p).name==Path(want['path']).name,'Source identity ordered shard association differs')
  require(row.get('passed') is True and row['sha256']==row['expected_sha256']==want['sha256'] and row['bytes']==want['size'],'Complete publisher source identity differs')
  require(isinstance(row['stat_before'],list) and len(row['stat_before'])==5 and row['stat_before']==row['stat_after'],'Canonical dev/inode/size/mtime/ctime source proof differs')
  if current:require(signature(p)==row['stat_after'],'Current source stat differs from complete identity proof')
 return {'path':str(Path(path).resolve()),'sha256':sha(path),'started':value['started'],'finished':value['finished'],'current_stat_verified':current,'complete_four_publisher_hashes_verified':True}


def finalized_binding(run_root):
 import layer0_numerical_qualification_v9 as ctrl
 run_root=Path(run_root).resolve();parent=read(run_root/'parent-qualification.json');child=read(run_root/'child/report.json');plan=read(run_root/'child/plan.snapshot.json');health=read(run_root/'post-health.json')
 require(sha(HERE/'layer0_numerical_qualification_v9.py')==DRIVER_SHA==parent['controller_sha256']==plan['driver_sha256'] and sha(HERE/'qualify_layer0_numerical_v9.py')==PARENT_SHA==parent['wrapper_sha256'],'Frozen V9 source identity differs')
 require(sha(SOURCE_PLAN)==SOURCE_PLAN_SHA==plan['source_plan_sha256'],'Frozen29 capture engine source plan differs')
 require(sha(run_root/'child/plan.snapshot.json')==parent['plan_sha256']==child['plan_sha256']==sha(run_root/'input-plan.snapshot.json'),'Finalized native plan/report identity differs')
 require(parent.get('passed') is True and not parent['errors'] and parent.get('child_return_code')==0 and parent.get('owned_containers_terminal') is True and not parent.get('interrupted') and not parent.get('forced_cleanup') and parent.get('kernel_fault_gate_passed') is True and parent.get('pre_health_passed') is True and parent.get('post_health_passed') is True,'Finalized native parent lifecycle/source/health not qualified')
 require(child.get('passed') is True and child.get('numerical_and_teardown_passed') is True and child.get('post_health_passed') is True and child.get('layer0_logical_lifecycle_qualified') is True and child.get('packet_contract_qualified') is True and child.get('error') is None and child.get('full_model_math_qualified') is False,'Finalized V9 native observation report absent/scope differs')
 require(health['passed'] is True and health['finished_epoch']>=child['finished_epoch'] and parent['finished_epoch']>=health['finished_epoch'],'Native post-health terminal chronology differs')
 for name,key in [('layer0-logical-lifecycle.json','lifecycle_receipt_sha256'),('packet-contract.json','packet_receipt_sha256')]:require(sha(run_root/name)==child[key] and read(run_root/name).get('passed') is True,'Actual native lifecycle/packet association differs')
 require(read(run_root/'packet-contract.json')['requests_sha256']==sha(run_root/'child/candidatecombined_on/requests.json'),'Native four-prefix raw report differs from finalized packet evidence')
 for name,result in zip(('reference21','candidatecombined_off','candidatecombined_on'),child['results']):require(sha(run_root/'child'/name/'engine.combined.log')==result['canonical_combined_log_sha256'],'Finalized canonical actual producer log changed')
 require(child['engine_receipt_sha256']==plan['engine_receipt_sha256'] and [r['passed'] for r in child['results']]==[True]*3 and all(r.get('removed') is True and r.get('complete_four_prefix_roster') is True for r in child['results']),'Finalized native engine/arm roster differs')
 # Actual SDK/source/library/C1/full390 proof, six API fingerprints and frozen
 # mathematical launch flags; this may stat/read both known pages when run.
 chain=ctrl.manifest_binding(plan)
 require(chain==parent['prepared_chain'],'Actual finalized SDK/C1/source gate chain differs')
 post=child['post_model_identity'];require(Path(post['path']).resolve()==run_root/'post-model-identity.json' and sha(post['path'])==post['sha256'],'Native terminal all4 source proof association differs')
 return parent,child,plan,{'parent_sha256':sha(run_root/'parent-qualification.json'),'child_sha256':sha(run_root/'child/report.json'),'plan_sha256':sha(run_root/'child/plan.snapshot.json'),'source_plan_sha256':SOURCE_PLAN_SHA,'chain':chain,'native_terminal_postidentity':post,'posthealth_finished':health['finished_epoch']}


def native_prefix1(run_root,plan):
 from verify_layer0_ffn_original_v1 import load_frame
 requests_path=Path(run_root)/'child/candidatecombined_on/requests.json';rows=read(requests_path)
 require([r['prefix'] for r in rows]==[1,2,4,8],'Exact finalized V9 four-prefix observation roster differs')
 row=rows[0];ids=row['raw']['ids'];require(len(ids)==1 and ids==plan['prefixes']['1'] and row['raw']['fresh']==1,'Actual prefix1 GEN IDs/fresh binding differs')
 _,_,l0binding=load_frame(row) # Native metadata/packets/provenance only, no original payload.
 frame=row['layer0']['frame'];pid=frame['pid'];ordinal=frame['request'];require(frame['binding_sha256']==sha(Path(run_root)/'child/plan.snapshot.json'),'Actual frame source binding differs')
 meta=row['meta'];require(meta['coverage']['passed'] is True and meta['ledger']['cancelled'] is False and meta['ledger']['actual_reused']==0 and meta['ledger']['evaluated_prompt_rows']==1 and row['external']['passed'] is True,'Native complete fresh prefix1 coverage differs')
 vectors=meta['residuals']+meta['logits'];require(len(meta['logits'])==1 and len(meta['residuals'])==48 and sorted(int(r['layer']) for r in meta['residuals'])==list(range(48)),'Native fullhead/all48 observation scope incomplete')
 actualmarkers=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in row['raw']['stderr'] if line.startswith('SFD vector ')]
 require(len(actualmarkers)==49,'Native source vector marker roster differs')
 values={};bindings={};seen=set();stages=meta['coverage']['required_stage_ranges']
 from serial_prefix_qualification_v6 import expected_stage_ranges
 require(list(stages.values())==[list(bounds) for bounds in expected_stage_ranges(plan['args'])],'Native configured/observed stage roster differs')
 for vector in vectors:
  layer=int(vector['layer']);phase='first_logits_before_sampler' if layer==-1 else 'first_window_residual';size=993280 if layer==-1 else 40960;p=Path(vector['path'])
  require(p.resolve().parent==(Path(run_root)/'child/candidatecombined_on/captures').resolve() and not p.is_symlink() and p.name==Path(vector['file']).name and str(p) not in seen,'Native raw file path/reuse differs');seen.add(str(p))
  require(int(vector['pid'])==pid and int(vector['request'])==ordinal and int(vector['pos'])==0 and int(vector['token'])==ids[0] and vector['phase']==phase and vector['canonical']=='le_f32' and int(vector['bytes'])==size and int(vector['reused'])==0,'Native row/PID/source/extent binding differs')
  bounds=stages[str(vector['stage'])];require(bounds==[int(vector['lb']),int(vector['le'])] and (bounds[0]<=layer<bounds[1] if layer>=0 else bounds[1]==48),'Native stage/head ownership differs')
  require(sum(all(marker.get(k)==str(vector[k]) for k in ('pid','request','stage','lb','le','phase','layer','pos','token','bytes','file','canonical','reused','observed_only')) for marker in actualmarkers)==1,'Native vector producer marker differs')
  raw=p.read_bytes();require(len(raw)==size and hashlib.sha256(raw).hexdigest()==vector['sha256'],'Native raw file SHA/extent differs');value=np.frombuffer(raw,dtype='<f4');require(np.isfinite(value).all(),'Native raw nonfinite vector')
  key='head' if layer==-1 else 'post_ffn_%d'%layer;values[key]=value.copy();bindings[key]=dict(vector)
 for name,key in [('residual_input','embedding'),('residual_after_attn','post_attention_0')]:
  field=next(f for f in row['layer0']['observed'] if f['name']==name);p=Path(field['path']);raw=p.read_bytes();require(len(raw)==40960 and sha(p)==field['sha256'] and not p.is_symlink(),'Native L0 input/postattn binding differs');values[key]=np.frombuffer(raw,dtype='<f4').copy();bindings[key]=field
 return ids,values,{'requests_path':str(requests_path),'requests_sha256':sha(requests_path),'native_vectors':bindings,'layer0_binding':l0binding,'prefix':1,'nativePID':pid,'request':ordinal}


def compare_observation(own,native):
 own=np.asarray(own,dtype='<f4').reshape(-1);native=np.asarray(native,dtype='<f4').reshape(-1);require(own.shape==native.shape and np.isfinite(own).all() and np.isfinite(native).all(),'Comparison finite/shape differs')
 differs=np.flatnonzero(own.view('<u4')!=native.view('<u4'));first=None if not len(differs) else int(differs[0]);return {**metrics(own.astype(np.float64),native.astype(np.float64)),'bitwise_equal':first is None,'first_bitwise_differing_index':first,'own_value_at_first_difference':None if first is None else float(own[first]),'native_value_at_first_difference':None if first is None else float(native[first]),'numeric_gate_assigned':False}


def save_array(output,leaf,value,encoding='LE_F32'):
 p=Path(output)/leaf;raw=np.asarray(value,dtype='<f4').tobytes();require(not p.exists() and np.isfinite(np.frombuffer(raw,dtype='<f4')).all(),'Preserve output/nonfinite own array');p.write_bytes(raw);return {'path':str(p),'sha256':sha(p),'bytes':len(raw),'shape':list(np.shape(value)),'encoding':encoding}


def save_owned_and_compare(output,owned,native):
 require(owned['ids'] and len(owned['ids'])==1 and owned['captured_inputs_used'] is False and owned['captured_states_or_selected_ids_used'] is False and owned['full_model_math_qualified'] is False,'Owned prefix1/input scope differs')
 layers=owned['trace'][0]['layers'];require([v['layer'] for v in layers]==list(range(48)),'Owned complete48 trace absent');files={};checks=[]
 files['head']=save_array(output,'own-first-head.f32',owned['first_generated_logits']);checks.append(('embedding',compare_observation(layers[0]['input'],native['embedding'])));checks.append(('post_attention_0',compare_observation(layers[0]['attention'],native['post_attention_0'])))
 for layer in layers:
  l=layer['layer']
  for phase in ('input','attention','ffn'):files['layer%d_%s'%(l,phase)]=save_array(output,'own-l%d-%s.f32'%(l,phase),layer[phase])
  checks.append(('post_ffn_%d'%l,compare_observation(layer['ffn'],native['post_ffn_%d'%l])))
 checks.append(('head',compare_observation(owned['first_generated_logits'],native['head'])))
 for l,state in owned['gdn_states_owned'].items():
  for role,value in state.items():files['gdn%d_%s'%(l,role)]=save_array(output,'own-gdn%d-%s.f32'%(l,role),value)
 for l,state in owned['qsa_states_owned'].items():
  for role in ('keys','values','tail','dead','pooled'):files['qsa%d_%s'%(l,role)]=save_array(output,'own-qsa%d-%s.f32'%(l,role),getattr(state,role))
 files['ple_history']=save_array(output,'own-ple-history.f32',owned['ple_history_owned'])
 return {'qsa_logical_scalars':{str(l):{'block_pos':state.block_pos,'position_base':state.pos_base,'history_tokens':[r['token'] for r in state.records],'records_are_owned_estimates':True} for l,state in owned['qsa_states_owned'].items()},'arrays':files,'comparisons':dict(checks),'first_bitwise_difference':next(({'scope':key,**value} for key,value in checks if not value['bitwise_equal']),None),'owned_last_two':owned['last_two_owned'],'owned_routes':[r['route'] for r in owned['trace']],'numeric_pass_claim':False,'tolerance_gate':None,'full_model_math_qualified':False}


def dependency_binding():
 manifest=read(HERE/'full48-owned-composition-storage-ready-manifest-v1.json');plan=read(HERE/'full48-owned-composition-storage-source-plan-v1.json')
 for name,want in {**manifest['files'],**plan['frozen_dependencies']}.items():require(sha(ROOT/name)==want,'Owned frozen helper source changed '+name)
 return {str(path):sha(path) for path in [Path(__file__),FOUNDATION,HERE/'full48-owned-composition-storage-ready-manifest-v1.json',HERE/'full48-owned-composition-storage-source-plan-v1.json',HERE/'source_page_watchdog_v3.py',HERE/'run_source_upload_oracle_full_v2.py',HERE/'verify_layer0_ffn_original_v1.py',HERE/'layer0_numerical_qualification_v9.py',HERE/'qualify_layer0_numerical_v9.py']}


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--native-run-root',type=Path,required=True);ap.add_argument('--model-identity',type=Path,required=True);ap.add_argument('--prefix',type=int,choices=[1],default=1);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 report={'schema':1,'status':'incomplete','started_epoch':time.time(),'numeric_pass_claim':False,'full_model_math_qualified':False,'nativebitwise_qualified':False,'tolerance_gate':None,'errors':[]};lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];computation_terminal=None;source_admitted=False
 def watch(label):
  try:return guard(shards[2])
  except Exception:
   report['preserved_'+label]=preserve(shards[2],out,label);raise
 try:
  report['dependency_sha256']=dependency_binding();report['initial_known_pages']=watch('initial-failure');parent,child,plan,binding=finalized_binding(a.native_run_root);report['native_finalized_binding']=binding
  boundary=max(parent['child_terminal_epoch'],child['finished_epoch'],binding['posthealth_finished']);report['pre_original_identity']=identity_admission(a.model_identity,lock_path,shards,boundary);report['pre_known_pages']=watch('pre-failure');write(out/'report.json',report)
  ids,native,native_binding=native_prefix1(a.native_run_root,plan);report['native_observation_binding']=native_binding;source_admitted=True
  numpy_config=io.StringIO()
  with contextlib.redirect_stdout(numpy_config):np.show_config()
  report['runtime']={'numpy_version':np.__version__,'numpy_module':np.__file__,'numpy_module_sha256':sha(np.__file__),'numpy_configuration':numpy_config.getvalue(),'numpy_binary_sha256':{str(p):sha(p) for p in sorted(Path(np.__file__).parent.rglob('*.so'))},'native_config_env':read(Path(plan['prepared'])/'server-config.json')['env'],'env':{k:v for k,v in os.environ.items() if k.startswith(('STRATA_','OMP_','MKL_','OPENBLAS_','NUMEXPR_','SYCL_','ONEAPI_'))},'effective_math_profile':{'earlier_prefill_rows':0,'last_verifier_rows':1,'nativeHC':True,'source_exactPLE':True,'prefill_noMMQ_noFUSED_BF16X2':True,'original_state_inputs':'zero; embeddings only'},'numeric_runtime_qualified':False};write(out/'report.json',report)
  provider=BoundOriginalFull48Rows(FOUNDATION,a.model_identity);report['original_role_binding']={name:{'type':tensor['type'],'shape':tensor['shape_ggml_order'],'offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'source_path':file['path'],'source_stat':sig} for name,(file,tensor,sig) in provider.reader.tensors.items()};write(out/'report.json',report);composition=Full48OwnedComposition(provider,sha(a.model_identity),tile_bytes=64<<20);owned=composition.tokens(ids);computation_terminal=time.time();report['computation_terminal_epoch']=computation_terminal;report['exploration']=save_owned_and_compare(out,owned,native);report['packet_events_owned']=owned['packet_events'];report['unsupported']=owned['unsupported'];report['post_known_pages']=watch('post-failure');identity_admission(a.model_identity,lock_path,shards,boundary);report['status']='computed; final full4 source proof pending';write(out/'report.json',report)
 except Exception as error:
  computation_terminal=time.time();report['errors'].append(str(error));report['traceback']=traceback.format_exc()
  if 'initial_known_pages' in report:
   try:report['failure_page_views']=preserve(shards[2],out,'admission-or-computation-failure')
   except Exception as preservation_error:report['errors'].append('Page preservation: '+str(preservation_error))
 finally:
  # New complete all4 scan after CPU terminal, independent of original stat proof.
  try:
   if not source_admitted:raise RuntimeError('Fullscan skipped: original payload admission not reached')
   after=computation_terminal or time.time();identity=full_buffered_identity(lock_path,lock,shards,out/'post-original-model-identity.json',after);require(identity['passed'],'NEW post-CPU complete publisher identity failed');report['post_original_identity']=identity_admission(out/'post-original-model-identity.json',lock_path,shards,after);report['final_known_pages']=watch('final-failure')
  except Exception as error:
   report['errors'].append('Post-source proof: '+str(error))
   try:
    if source_admitted:report['preserved_source_pages']=preserve(shards[2],out,'failure')
   except Exception as preserve_error:report['errors'].append('Source preservation: '+str(preserve_error))
  report['finished_epoch']=time.time();report['status']='exploratory complete; NO numerical qualification' if not report['errors'] else 'FAILED source/admission/execution; preserve evidence';write(out/'report.json',report)
 print(json.dumps({'status':report['status'],'errors':report['errors'],'first_bitwise_difference':report.get('exploration',{}).get('first_bitwise_difference'),'numeric_pass_claim':False,'full_model_math_qualified':False}));return 1 if report['errors'] else 0
if __name__=='__main__':raise SystemExit(main())
