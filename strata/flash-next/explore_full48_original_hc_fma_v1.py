#!/usr/bin/env python3
"""Original own-state prefix1/2/4/8 exploration from finalized source35/P30V4.
Native observations are comparison targets only; no numerical PASS/tolerance.
Actual original payload execution belongs to separately authorized root CLI use.
"""
import argparse,contextlib,hashlib,io,json,os,time,traceback
from pathlib import Path
import numpy as np
from full48_owned_composition_storage_v2 import BoundOriginalFull48Rows,derive_schedule,bind_dispatch_source
from full48_owned_hc_fma_refinement_v1 import OwnedHcFmaRefinement,source_binding as refinement_binding
import qualify_hc35_host_bulk_runtime_v1 as bulk_qualification
from explore_full48_original_prefix1_v4 import identity_admission,compare_observation,save_array,sha,read,require,write
from explore_full48_original_prefix1_v4 import dependency_binding as frozen_prefix1_dependencies
from validate_prefix_residual30_final_v4 import finalized_binding
from collect_prefix_residual30_v2 import window_nonce
from source_page_watchdog_v3 import guard,preserve
from run_source_upload_oracle_full_v2 import full_buffered_identity
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
FOUNDATION=HERE/'original-gguf-reference-foundation-plan-v1.json'

def dependency_binding():
 binding=frozen_prefix1_dependencies();plan=read(HERE/'full48-original-route-reference-source-plan-v2.json')
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Route reference source dependency changed '+name)
 return {**binding,**plan['files']}

def native_prefix(run_root,plan,prefix,schedule):
 _,_,_,admitted=finalized_binding(run_root);root=Path(run_root).resolve();directory=root/'child/p30_on';requests_path=directory/'requests.json';rows=read(requests_path)
 require([row['prefix'] for row in rows]==[1,2,4,8],'Exact finalized native four-prefix roster differs');row=next(row for row in rows if row['prefix']==prefix);ids=row['raw']['ids'];require(ids==plan['prefixes'][str(prefix)]==schedule['accepted_ids'] and len(ids)==prefix and row['raw']['fresh']==1,'Actual fresh accepted prefix differs')
 pid=int(row['meta']['logits'][0]['pid']);ordinal=int(row['meta']['logits'][0]['request']);values={};bindings={};seen_windows=set()
 frames=[frame for frame in admitted['capture']['frames'] if frame['binding']['request']==ordinal]
 stages=row['meta']['coverage']['required_stage_ranges'];expected_windows={(window['position'],window['rows'],stage) for window in schedule['windows'] for stage in map(int,stages)}
 for frame in frames:
  data=frame['binding'];position=data['first_position'];count=data['rows'];stage=data['stage'];key=(position,count,stage);require(key in expected_windows and key not in seen_windows,'Native window/stage route differs from config-derived schedule');seen_windows.add(key)
  window=next(window for window in schedule['windows'] if window['position']==position and window['rows']==count);route='prompt_verifier' if window['normal_dispatch']=='prompt_verify' else 'verifier'
  require(data['pid']==pid and data['gen_ids']==ids and data['schema']==2 and data['route']==route and data['normal_dispatch']==window['normal_dispatch'] and data['nonce']==window_nonce(pid,ordinal,position,count,route),'Native source35 route/current window differs from independently derived config')
  for field in frame['fields']:
   path=Path(field['path']);require(path.resolve().parent==(directory/'p30').resolve() and not path.is_symlink() and sha(path)==field['sha256'] and field['bytes']==count*40960,'Native full matrix raw path/SHA/extent differs')
   raw=path.read_bytes();require(len(raw)==field['bytes'] and hashlib.sha256(raw).hexdigest()==field['sha256'],'Native matrix changed during read');matrix=np.frombuffer(raw,dtype='<f4').reshape(count,4,2560);require(np.isfinite(matrix).all(),'Native nonfinite matrix')
   for offset in range(count):
    key='p%d_l%d_%s'%(position+offset,field['layer'],field['phase']);require(key not in values,'Duplicate native row/layer/phase');values[key]=matrix[offset].copy();bindings[key]={**field,'row_offset':offset,'position':position+offset,'source_rows':count,'source_dispatch':data['normal_dispatch']}
 require(seen_windows==expected_windows and set(values)=={'p%d_l%d_%s'%(position,layer,phase) for position in range(prefix) for layer in range(48) for phase in ('input','attention','ffn')},'All native prefix rows/all48/three phases required')
 head=row['meta']['logits'][0];path=Path(head['path']);require(path.resolve().parent==(directory/'captures').resolve() and not path.is_symlink() and sha(path)==head['sha256'] and path.stat().st_size==993280,'Current native SFD fullhead binding differs');raw=path.read_bytes();require(len(raw)==993280 and hashlib.sha256(raw).hexdigest()==head['sha256'],'Native fullhead changed during read');values['head']=np.frombuffer(raw,dtype='<f4').copy();bindings['head']=head;require(np.isfinite(values['head']).all(),'Nonfinite actual fullhead')
 return ids,values,{'requests_path':str(requests_path),'requests_sha256':sha(requests_path),'native_vectors':bindings,'prefix':prefix,'nativePID':pid,'request':ordinal,'comparison_vector_count':144*prefix+1,'native_inputs_or_routes_used_for_own_computation':False}

def save_owned_and_compare(output,owned,native):
 require(len(owned['ids']) in (1,2,4,8) and owned['captured_inputs_used'] is False and owned['captured_states_or_selected_ids_used'] is False and owned['native_routes_used_as_math_inputs'] is False and owned['state_initialized_from_zero'] is True and owned['full_model_math_qualified'] is False,'Independent owned input/state scope differs')
 files={};checks=[]
 for row in owned['trace']:
  position=row['position'];require([layer['layer'] for layer in row['layers']]==list(range(48)),'Independent complete48 trace required')
  for layer in row['layers']:
   for phase in ('input','attention','ffn'):
    key='p%d_l%d_%s'%(position,layer['layer'],phase);files[key]=save_array(output,'own-'+key+'.f32',layer[phase]);checks.append((key,compare_observation(layer[phase],native[key])))
 files['native_head']=save_array(output,'native-first-head.f32',native['head']);files['head']=save_array(output,'own-first-head.f32',owned['first_generated_logits']);checks.append(('head',compare_observation(owned['first_generated_logits'],native['head'])))
 for layer,state in owned['gdn_states_owned'].items():
  for role,value in state.items():files['gdn%d_%s'%(layer,role)]=save_array(output,'own-gdn%d-%s.f32'%(layer,role),value)
 for layer,state in owned['qsa_states_owned'].items():
  for role in ('keys','values','tail','dead','pooled'):files['qsa%d_%s'%(layer,role)]=save_array(output,'own-qsa%d-%s.f32'%(layer,role),getattr(state,role))
 files['ple_history']=save_array(output,'own-ple-history.f32',owned['ple_history_owned'])
 require(len(checks)==144*len(owned['ids'])+1,'Exact independent/native comparison roster differs')
 return {'arrays':files,'comparisons':dict(checks),'first_bitwise_difference':next(({'scope':key,**value} for key,value in checks if not value['bitwise_equal']),None),'owned_last_two':owned['last_two_owned'],'declared_schedule':owned['declared_schedule'],'numeric_pass_claim':False,'tolerance_gate':None,'actual_window_group_rounding_qualified':False,'full_model_math_qualified':False}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--native-run-root',type=Path,required=True);ap.add_argument('--model-identity',type=Path,required=True);ap.add_argument('--prefix',type=int,choices=[1,2,4,8],default=2);ap.add_argument('--bulk-build-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 report={'schema':1,'status':'incomplete','started_epoch':time.time(),'numeric_pass_claim':False,'full_model_math_qualified':False,'nativebitwise_qualified':False,'tolerance_gate':None,'errors':[]};lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];computation_terminal=None;source_admitted=False
 def watch(label):
  try:return guard(shards[2])
  except Exception:
   report['preserved_'+label]=preserve(shards[2],out,label);raise
 try:
  report['bulk_qualification_pre']=bulk_qualification.finalized_binding(a.bulk_build_root);report['dependency_sha256']=dependency_binding();report['initial_known_pages']=watch('initial-failure');parent,child,plan,binding=finalized_binding(a.native_run_root);report['native_finalized_binding']=binding
  boundary=max(parent['child_terminal_epoch'],child['finished_epoch'],binding['posthealth_finished']);report['pre_original_identity']=identity_admission(a.model_identity,lock_path,shards,boundary);report['pre_known_pages']=watch('pre-failure');write(out/'report.json',report)
  cfg=read(Path(plan['prepared'])/'server-config.json');source_root=Path(plan['engine_root'])/'source';report['HC_refinement_source_pre']=refinement_binding(source_root);report['source_dispatch_binding']=bind_dispatch_source(source_root);schedule=derive_schedule(plan['prefixes'][str(a.prefix)],plan['args'],cfg['env']);ids,native,native_binding=native_prefix(a.native_run_root,plan,a.prefix,schedule);report['native_observation_binding']=native_binding;report['declared_schedule']=schedule;source_admitted=True
  numpy_config=io.StringIO()
  with contextlib.redirect_stdout(numpy_config):np.show_config()
  report['runtime']={'numpy_version':np.__version__,'numpy_module':np.__file__,'numpy_module_sha256':sha(np.__file__),'numpy_configuration':numpy_config.getvalue(),'numpy_binary_sha256':{str(p):sha(p) for p in sorted(Path(np.__file__).parent.rglob('*.so'))},'native_config_env':read(Path(plan['prepared'])/'server-config.json')['env'],'env':{k:v for k,v in os.environ.items() if k.startswith(('STRATA_','OMP_','MKL_','OPENBLAS_','NUMEXPR_','SYCL_','ONEAPI_'))},'effective_math_profile':{'earlier_prefill_rows':0,'earlier_prompt_verifier_rows':len(ids)-1,'last_verifier_rows':1,'actual_T_group_rounding_qualified':False,'nativeHC':True,'source_exactPLE':True,'prefill_noMMQ_noFUSED_BF16X2':True,'original_state_inputs':'zero; embeddings only'},'numeric_runtime_qualified':False};write(out/'report.json',report)
  provider=BoundOriginalFull48Rows(FOUNDATION,a.model_identity);report['original_role_binding']={name:{'type':tensor['type'],'shape':tensor['shape_ggml_order'],'offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'source_path':file['path'],'source_stat':sig} for name,(file,tensor,sig) in provider.reader.tensors.items()};write(out/'report.json',report);composition=OwnedHcFmaRefinement(provider,sha(a.model_identity),plan['args'],cfg['env'],source_root,a.bulk_build_root,tile_bytes=64<<20);owned=composition.tokens(ids);report['HC_refinement_provenance']={k:v for k,v in owned.items() if k.startswith('HC_') or k=='non_HC_primitives_changed'};report['bulk_qualification_post']=bulk_qualification.finalized_binding(a.bulk_build_root);require(report['bulk_qualification_post']==report['bulk_qualification_pre'],'Bulk helper/runtime changed');computation_terminal=time.time();report['computation_terminal_epoch']=computation_terminal;report['exploration']=save_owned_and_compare(out,owned,native);report['packet_events_owned']=owned['packet_events'];report['unsupported']=owned['unsupported'];report['post_dispatch_binding']=bind_dispatch_source(source_root);require(report['post_dispatch_binding']==report['source_dispatch_binding'],'Source35 dispatch code/stat changed during own computation');report['post_known_pages']=watch('post-failure');identity_admission(a.model_identity,lock_path,shards,boundary);report['status']='computed; final full4 source proof pending';write(out/'report.json',report)
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
