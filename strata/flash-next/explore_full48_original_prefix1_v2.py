#!/usr/bin/env python3
"""Readonly original own-prefix1 exploration against finalized source33/P30V2 observations.
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
SOURCE_PLAN=HERE/'current-ple-input33-engine-build-plan-v1.json'
SOURCE_PLAN_SHA='81798dff3fd012f7b95984403ea1088a1f157d37ee2d2e6e1b612711052cd34c'


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
 from validate_prefix_residual30_final_v2 import finalized_binding as validate
 return validate(run_root)


def native_prefix1(run_root,plan):
 from validate_prefix_residual30_final_v2 import finalized_binding as validate
 _,_,_,admitted=validate(run_root)
 root=Path(run_root).resolve();directory=root/'child/p30_on';requests_path=directory/'requests.json';rows=read(requests_path)
 require([r['prefix'] for r in rows]==[1,2,4,8],'Exact native four-prefix roster differs')
 row=rows[0];ids=row['raw']['ids'];require(len(ids)==1 and ids==plan['prefixes']['1'],'Actual source33 prefix1 differs')
 pid=int(row['meta']['logits'][0]['pid']);ordinal=int(row['meta']['logits'][0]['request']);values={};bindings={}
 frames=[f for f in admitted['capture']['frames'] if f['binding']['request']==ordinal]
 require(len(frames)==len(row['meta']['coverage']['required_stage_ranges']),'Exact prefix1 P30 stage roster differs')
 for frame in frames:
  d=frame['binding'];require(d['pid']==pid and d['gen_ids']==ids and d['route']=='verifier' and d['rows']==1 and d['first_position']==0,'Current P30 prefix1 producer/row binding differs')
  for field in frame['fields']:
   p=Path(field['path']);require(p.resolve().parent==(directory/'p30').resolve() and not p.is_symlink() and sha(p)==field['sha256'] and field['bytes']==40960,'Current P30 raw path/SHA/extent differs')
   key='layer%d_%s'%(field['layer'],field['phase']);require(key not in values,'Duplicate P30 layer/phase');values[key]=np.frombuffer(p.read_bytes(),dtype='<f4').reshape(4,2560).copy();bindings[key]=dict(field)
 require(set(values)=={'layer%d_%s'%(layer,phase) for layer in range(48) for phase in ('input','attention','ffn')},'All144 native P30 phase matrices required')
 head=row['meta']['logits'][0];p=Path(head['path']);require(p.resolve().parent==(directory/'captures').resolve() and not p.is_symlink() and sha(p)==head['sha256'] and p.stat().st_size==993280,'Current full SFD head binding differs');values['head']=np.frombuffer(p.read_bytes(),dtype='<f4').copy();bindings['head']=head
 require(all(np.isfinite(v).all() for v in values.values()),'Native nonfinite observation')
 return ids,values,{'requests_path':str(requests_path),'requests_sha256':sha(requests_path),'native_vectors':bindings,'prefix':1,'nativePID':pid,'request':ordinal,'all144_P30_phases':True,'native_inputs_used_for_own_computation':False}


def compare_observation(own,native):
 own=np.asarray(own,dtype='<f4').reshape(-1);native=np.asarray(native,dtype='<f4').reshape(-1);require(own.shape==native.shape and np.isfinite(own).all() and np.isfinite(native).all(),'Comparison finite/shape differs')
 differs=np.flatnonzero(own.view('<u4')!=native.view('<u4'));first=None if not len(differs) else int(differs[0]);return {**metrics(own.astype(np.float64),native.astype(np.float64)),'bitwise_equal':first is None,'first_bitwise_differing_index':first,'own_value_at_first_difference':None if first is None else float(own[first]),'native_value_at_first_difference':None if first is None else float(native[first]),'numeric_gate_assigned':False}


def save_array(output,leaf,value,encoding='LE_F32'):
 p=Path(output)/leaf;raw=np.asarray(value,dtype='<f4').tobytes();require(not p.exists() and np.isfinite(np.frombuffer(raw,dtype='<f4')).all(),'Preserve output/nonfinite own array');p.write_bytes(raw);return {'path':str(p),'sha256':sha(p),'bytes':len(raw),'shape':list(np.shape(value)),'encoding':encoding}


def save_owned_and_compare(output,owned,native):
 require(owned['ids'] and len(owned['ids'])==1 and owned['captured_inputs_used'] is False and owned['captured_states_or_selected_ids_used'] is False and owned['full_model_math_qualified'] is False,'Owned prefix1/input scope differs')
 layers=owned['trace'][0]['layers'];require([v['layer'] for v in layers]==list(range(48)),'Owned complete48 trace absent');files={};checks=[]
 files['head']=save_array(output,'own-first-head.f32',owned['first_generated_logits'])
 for layer in layers:
  l=layer['layer']
  for phase in ('input','attention','ffn'):
   key='layer%d_%s'%(l,phase);files[key]=save_array(output,'own-l%d-%s.f32'%(l,phase),layer[phase]);checks.append((key,compare_observation(layer[phase],native[key])))
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
 source=read(HERE/'full48-original-prefix1-p30-source-plan-v2.json')
 for name,want in source['files'].items():require(sha(ROOT/name)==want,'New source33 exploration/admission dependency changed '+name)
 return {**source['files'],str(FOUNDATION):sha(FOUNDATION)}


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
  provider=BoundOriginalFull48Rows(FOUNDATION,a.model_identity);report['original_role_binding']={name:{'type':tensor['type'],'shape':tensor['shape_ggml_order'],'offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'source_path':file['path'],'source_stat':sig} for name,(file,tensor,sig) in provider.reader.tensors.items()};write(out/'report.json',report);math_keys=('STRATA_SYCL_NATIVE_HC','STRATA_PREFILL_MMQ','STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD','STRATA_GDN_REC_HEADS');native_env=read(Path(plan['prepared'])/'server-config.json')['env'];math_options={key:native_env[key] for key in math_keys if key in native_env};composition=Full48OwnedComposition(provider,sha(a.model_identity),tile_bytes=64<<20,runtime_options=math_options);owned=composition.tokens(ids);computation_terminal=time.time();report['computation_terminal_epoch']=computation_terminal;report['exploration']=save_owned_and_compare(out,owned,native);report['packet_events_owned']=owned['packet_events'];report['unsupported']=owned['unsupported'];report['post_known_pages']=watch('post-failure');identity_admission(a.model_identity,lock_path,shards,boundary);report['status']='computed; final full4 source proof pending';write(out/'report.json',report)
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
