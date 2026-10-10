#!/usr/bin/env python3
"""Original-owned L0 GDN replay against finalized NUM10 source35 output targets.
Root-only actual payload CLI. No numeric PASS/tolerance or native math inputs.
"""
import argparse,contextlib,hashlib,io,json,os,time,traceback
from pathlib import Path
import numpy as np
from owned_layer0_gdn_replay_v1 import OwnedLayer0GdnReplay,F32_SHAPES,PACKET_SHAPES
from full48_owned_composition_storage_v2 import BoundOriginalFull48Rows,derive_schedule,bind_dispatch_source
import explore_full48_original_routes_v1 as original
from explore_full48_original_prefix1_v4 import identity_admission,compare_observation,save_array,sha,read,require,write
from source_page_watchdog_v3 import guard,preserve
from run_source_upload_oracle_full_v2 import full_buffered_identity
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
FOUNDATION=HERE/'original-gguf-reference-foundation-plan-v1.json'
SOURCE_PLAN=HERE/'owned-layer0-gdn-num10-source-plan-v1.json'

def dependencies():
 plan=read(SOURCE_PLAN)
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Owned L0 immutable source changed: '+name)
 return plan['files']

def num10_admission(run_root):
 # Explicit public finalized NUM10 source/lifecycle/packet/raw recollection.
 # Import deferred: no fallback to older generation when NUM10 is unavailable.
 from validate_layer0_numerical_final_v10 import finalized_binding
 return finalized_binding(run_root)

def load_target(run_root,plan,prefix,admitted_requests):
 root=Path(run_root).resolve();directory=root/'child/candidatecombined_on';requests=read(directory/'requests.json');require(requests==admitted_requests,'Native requests changed after finalized admission');require([row['prefix'] for row in requests]==[1,2,4,8],'Final NUM10 exact fresh prefix roster required')
 row=next(row for row in requests if row['prefix']==prefix);ids=row['raw']['ids'];frame=row['layer0']['frame'];require(ids==plan['prefixes'][str(prefix)] and len(ids)==prefix and row['raw']['fresh']==1 and frame['position']==prefix-1 and frame['token']==ids[-1] and frame['rows']==1 and frame['layer']==0,'Actual final T1 accepted IDs/source position differs')
 require(frame['binding_sha256']==sha(root/'child/plan.snapshot.json') and frame['gen_ids']==ids,'Current NUM10 frame/plan binding differs')
 roster={field['name']:field for field in row['layer0']['observed']};require(len(roster)==33 and sum(f['provenance']=='actual_buffer' for f in roster.values())==31,'Exact31actual source fields plus2derived required');values={};packets={};bindings={}
 for name in list(F32_SHAPES)+list(PACKET_SHAPES):
  f=roster[name];path=Path(f['path']);require(path.resolve().parent==(directory/'layer0').resolve() and not path.is_symlink() and f['provenance']=='actual_buffer','Native target path/source provenance differs '+name)
  before=path.stat();raw=path.read_bytes();after=path.stat();sig=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns];require(sig(before)==sig(after) and len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'],'Native target changed/extent differs '+name)
  if name in F32_SHAPES:
   require(f['encoding'].startswith('LE_F32') and len(raw)==int(np.prod(F32_SHAPES[name]))*4,'Native F32 target geometry differs '+name);value=np.frombuffer(raw,dtype='<f4').reshape(F32_SHAPES[name]).copy();require(np.isfinite(value).all(),'Native nonfinite comparison target');values[name]=value
  else:require(f['encoding'].startswith('Q8_1_LE') and len(raw)==PACKET_SHAPES[name],'Native packet target extent differs');packets[name]=raw
  bindings[name]=f
 return ids,values,packets,{'request_sha256':sha(directory/'requests.json'),'prefix':prefix,'pid':frame['pid'],'request':frame['request'],'frame_sha256':row['layer0']['metadata_sha256'],'field_bindings':bindings,'captured_values_used_only_as_output_comparison_targets':True}

def save_owned(output,owned):
 arrays={};packet_files={}
 for row in owned['rows']:
  position=row['position']
  for name,value in row['fields_owned'].items():arrays['p%d_%s'%(position,name)]=save_array(output,'own-p%d-%s.f32'%(position,name),value)
  for name,value in {'embedding':row['embedding'],**{key:row['mixer_details_owned'][key] for key in ('alpha','beta_pre','core_scaled','conv_silu')}}.items():arrays['p%d_%s'%(position,name)]=save_array(output,'own-p%d-%s.f32'%(position,name),value)
  for name,raw in row['packets_owned'].items():
   path=output/('own-p%d-%s.q81'%(position,name));require(not path.exists(),'Preserve own packet evidence');path.write_bytes(raw);packet_files['p%d_%s'%(position,name)]={'path':str(path),'sha256':sha(path),'bytes':len(raw),'encoding':'Q8_1_LE'}
 return arrays,packet_files

def compare_final(owned,values,packets):
 row=owned['rows'][-1];require(set(values)==set(F32_SHAPES) and set(packets)==set(PACKET_SHAPES),'Complete L0 GDN output target roster differs')
 checks={name:compare_observation(row['fields_owned'][name],values[name]) for name in F32_SHAPES};packet_checks={name:{'own_sha256':hashlib.sha256(row['packets_owned'][name]).hexdigest(),'native_sha256':hashlib.sha256(packets[name]).hexdigest(),'bytes_equal':row['packets_owned'][name]==packets[name],'bytes':len(packets[name]),'numeric_gate_assigned':False} for name in PACKET_SHAPES}
 return {'final_position':row['position'],'field_comparisons':checks,'packet_comparisons':packet_checks,'first_field_bitwise_difference':next((name for name,result in checks.items() if not result['bitwise_equal']),None),'numeric_pass_claim':False,'tolerance_gate':None,'captured_inputs_or_state_used_for_reference':False}

def load_own_array(root,row):
 path=Path(row['path']);require(path.resolve().parent==root.resolve() and not path.is_symlink(),'Original replay target escaped report root');raw=path.read_bytes();require(row['encoding']=='LE_F32' and len(raw)==row['bytes']==int(np.prod(row['shape']))*4 and hashlib.sha256(raw).hexdigest()==row['sha256'],'Original replay target changed/geometry differs');value=np.frombuffer(raw,dtype='<f4').reshape(row['shape']).copy();require(np.isfinite(value).all(),'Original replay target nonfinite');return value

def compare_completed_original(owned,original_root,report_sha,original_native_root,lock_path,shards,plan,env):
 root=Path(original_root).resolve();path=root/'report.json';require(sha(path)==report_sha,'Completed original report SHA changed');report=read(path)
 require(report.get('status')=='exploratory complete; NO numerical qualification' and report.get('errors')==[] and report.get('numeric_pass_claim') is False and report.get('full_model_math_qualified') is False and report.get('tolerance_gate') is None,'Completed original route V2 report required')
 require(report['dependency_sha256']==original.dependency_binding(),'Completed original frozen source closure differs')
 _,_,original_plan,binding=original.finalized_binding(original_native_root)
 for key in ('parent_sha256','child_sha256','plan_sha256','source_plan_sha256'):require(binding[key]==report['native_finalized_binding'][key],'Original P30 finalized association differs '+key)
 require(original_plan['engine_receipt_sha256']==plan['engine_receipt_sha256'] and original_plan['args']==plan['args'],'Original/new NUM10 source35 math/topology arguments differ')
 require(read(Path(original_plan['prepared'])/'server-config.json')['env']==env,'Original/new NUM10 baseline math env differs')
 require(report['declared_schedule']==owned['declared_schedule'] and report['source_dispatch_binding']==owned['source_dispatch_binding']==report['post_dispatch_binding'],'Original/new own route/source3 binding differs')
 post=identity_admission(root/'post-original-model-identity.json',lock_path,shards,report['computation_terminal_epoch']);require(post==report['post_original_identity'] and post['finished']<=report['finished_epoch'],'Completed original new postCPU full4 identity changed')
 guard(shards[2]);arrays=report['exploration']['arrays'];checks=[]
 for row in owned['rows']:
  for phase,name in [('input','residual_input'),('attention','residual_after_attn')]:
   key='p%d_l0_%s'%(row['position'],phase);require(key in arrays,'Completed original L0 allrow phase absent');target=load_own_array(root,arrays[key]);comparison=compare_observation(row['fields_owned'][name],target);require(comparison['bitwise_equal'],'Own replay differs from completed original L0 phase '+key);checks.append({'scope':key,**comparison})
 for name,key in [('recurrent','gdn0_recurrent'),('conv','gdn0_conv')]:
  if key in arrays:
   target=load_own_array(root,arrays[key]);comparison=compare_observation(owned['state_owned'][name],target);require(comparison['bitwise_equal'],'Own replay differs from completed original L0 finalstate '+name);checks.append({'scope':key,**comparison})
 require(len(checks)==2*len(owned['ids'])+2,'Complete original L0 allrow input/attention+both finalstate replay evidence required')
 return {'report_path':str(path),'report_sha256':report_sha,'post_identity':post,'allrows_and_finalstates_replay_bitwise_equal':True,'checks':checks,'full_model_math_qualified':False}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--native-run-root',type=Path,required=True);ap.add_argument('--model-identity',type=Path,required=True);ap.add_argument('--prefix',type=int,choices=[1,2,4,8],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--original-root',type=Path);ap.add_argument('--original-report-sha256');ap.add_argument('--original-native-run-root',type=Path);a=ap.parse_args();require(all(v is not None for v in [a.original_root,a.original_report_sha256,a.original_native_run_root]) or all(v is None for v in [a.original_root,a.original_report_sha256,a.original_native_run_root]),'Original replay association arguments must all be supplied or omitted');out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 report={'schema':1,'status':'incomplete','started_epoch':time.time(),'numeric_pass_claim':False,'tolerance_gate':None,'full_model_math_qualified':False,'captured_inputs_used':False,'errors':[]};lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[ROOT/lock['destination']/row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')];admitted=False;terminal=None
 def watch(label):
  try:return guard(shards[2])
  except Exception:report['preserved_'+label]=preserve(shards[2],out,label);raise
 try:
  report['dependency_sha256']=dependencies();report['initial_known_pages']=watch('initial-failure');parent,child,plan,binding=num10_admission(a.native_run_root);report['native_finalized_binding']=binding
  boundary=max(parent['child_terminal_epoch'],child['finished_epoch'],read(Path(a.native_run_root)/'post-health.json')['finished_epoch']);report['pre_original_identity']=identity_admission(a.model_identity,lock_path,shards,boundary);report['pre_known_pages']=watch('pre-failure');env=read(Path(plan['prepared'])/'server-config.json')['env'];source=Path(plan['engine_root'])/'source';ids,targets,packets,target_binding=load_target(a.native_run_root,plan,a.prefix,binding['requests']);schedule=derive_schedule(ids,plan['args'],env);report['declared_schedule']=schedule;report['native_observation_binding']=target_binding;report['source_dispatch_binding']=bind_dispatch_source(source);write(out/'report.json',report);admitted=True
  cfg=io.StringIO()
  with contextlib.redirect_stdout(cfg):np.show_config()
  report['runtime']={'numpy_version':np.__version__,'numpy_module_sha256':sha(np.__file__),'numpy_configuration':cfg.getvalue(),'numpy_binary_sha256':{str(p):sha(p) for p in sorted(Path(np.__file__).parent.rglob('*.so'))},'native_config_env':env,'env':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','MKL_','OPENBLAS_','NUMEXPR_','STRATA_','SYCL_','ONEAPI_'))},'numeric_runtime_qualified':False};write(out/'report.json',report)
  provider=BoundOriginalFull48Rows(FOUNDATION,a.model_identity);model=OwnedLayer0GdnReplay(provider,sha(a.model_identity),plan['args'],env,source,tile_bytes=64<<20)
  report['original_role_binding']={name:{'type':tensor['type'],'shape':tensor['shape_ggml_order'],'offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'source_path':file['path'],'source_stat':sig} for name,(file,tensor,sig) in provider.reader.tensors.items() if name=='token_embd.weight' or name.startswith('blk.0.hc_attn_') or name in {'blk.0.'+role for role in ('attn_qkv.weight','attn_gate.weight','ssm_out.weight','ssm_alpha.weight','ssm_beta.weight','ssm_dt.bias','ssm_a','ssm_norm.weight','ssm_conv1d.weight')}};write(out/'report.json',report)
  owned=model.tokens(ids);report['owned_provenance']={key:owned[key] for key in ('lane','source_identity_sha256','captured_inputs_used','captured_states_or_routes_used','state_initialized_from_zero','actual_original_payload_used','actual_T_group_rounding_qualified','full_model_math_qualified','numerical_tolerance_assigned')};terminal=time.time();report['computation_terminal_epoch']=terminal;arrays,packet_files=save_owned(out,owned);report['owned_arrays']=arrays;report['owned_packets']=packet_files;report['exploration']=compare_final(owned,targets,packets);report['packet_events_owned']=[event for row in owned['rows'] for event in row['packet_events_owned']];report['unsupported']=owned['unsupported'];report['layout_contract']={key:owned[key] for key in ('logical_recurrent_layout','native_recurrent_layout','logical_history_layout','native_history_layout')}
  if a.original_root:report['completed_original_replay']=compare_completed_original(owned,a.original_root,a.original_report_sha256,a.original_native_run_root,lock_path,shards,plan,env)
  else:report['completed_original_replay']={'observed':False,'reason':'No completed full48 routeV2 report supplied'}
  _,_,post_plan,post_native=num10_admission(a.native_run_root);require(post_plan==plan and post_native==binding,'Final NUM10 native evidence changed during own replay');report['post_native_finalized_binding']=post_native;dependencies();report['post_dispatch_binding']=bind_dispatch_source(source);require(report['post_dispatch_binding']==report['source_dispatch_binding'],'Current source3 code/stat changed');report['post_known_pages']=watch('post-failure');identity_admission(a.model_identity,lock_path,shards,boundary);write(out/'report.json',report)
 except Exception as error:
  terminal=time.time();report['errors'].append(str(error));report['traceback']=traceback.format_exc()
  if 'initial_known_pages' in report:
   try:report['failure_page_views']=preserve(shards[2],out,'admission-or-computation-failure')
   except Exception as preservation_error:report['errors'].append('Page preservation: '+str(preservation_error))
 finally:
  try:
   require(admitted,'Fullscan skipped: original payload admission not reached');after=terminal or time.time();report['before_full4_pages']=watch('before-full4-failure');identity=full_buffered_identity(lock_path,lock,shards,out/'post-original-model-identity.json',after);require(identity['passed'],'NEW postCPU all4 publisher hash failed');report['post_original_identity']=identity_admission(out/'post-original-model-identity.json',lock_path,shards,after);report['final_known_pages']=watch('final-failure')
  except Exception as error:
   report['errors'].append('Post-source proof: '+str(error))
   if admitted:
    try:report['preserved_post_failure']=preserve(shards[2],out,'post-failure')
    except Exception as preserve_error:report['errors'].append('Source preservation: '+str(preserve_error))
  report['finished_epoch']=time.time();report['status']='exploratory complete; NO numerical qualification' if not report['errors'] else 'FAILED source/admission/execution; preserve evidence';write(out/'report.json',report)
 print(json.dumps({'status':report['status'],'errors':report['errors'],'prefix':a.prefix,'first_field_bitwise_difference':report.get('exploration',{}).get('first_field_bitwise_difference'),'numeric_pass_claim':False}));return 1 if report['errors'] else 0
if __name__=='__main__':raise SystemExit(main())
