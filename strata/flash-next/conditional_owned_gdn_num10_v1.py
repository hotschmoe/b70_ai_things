#!/usr/bin/env python3
"""Explicit CONDITIONAL final-row original GDN seams; never full owned math."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import explore_owned_layer0_gdn_num10_v1 as own
from owned_layer0_gdn_replay_v1 import OwnedLayer0GdnReplay,physical_recurrent,physical_conv
from full48_owned_composition_storage_v2 import BoundOriginalFull48Rows,derive_schedule,bind_dispatch_source
from independent_q8_1_activation_v1 import encode
sha,read,write,require=own.sha,own.read,own.write,own.require
HERE=Path(__file__).resolve().parent

def dependencies():
 plan=read(HERE/'conditional-owned-gdn-num10-source-plan-v1.json')
 for name,want in plan['files'].items():require(sha(own.ROOT/name)==want,'Conditional immutable source changed '+name)
 return plan['files']

def preflight(native_root,original_root,report_sha,lock_path,shards):
 path=original_root/'report.json';raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==report_sha==sha(path),'Completed owned L0 report changed during read');report=json.loads(raw)
 require(report.get('status')=='exploratory complete; NO numerical qualification' and report.get('errors')==[] and report.get('numeric_pass_claim') is False and report.get('full_model_math_qualified') is False and report.get('captured_inputs_used') is False,'Completed independently owned L0 report required')
 require(report['dependency_sha256']==own.dependencies(),'Completed owned source closure changed')
 parent,child,plan,binding=own.num10_admission(native_root)
 require(binding==report['native_finalized_binding']==report['post_native_finalized_binding'],'Current NUM10 raw/source/parent association differs')
 env=read(Path(plan['prepared'])/'server-config.json')['env'];ids=report['declared_schedule']['accepted_ids'];require(len(ids)==4,'This diagnostic requires completed prefix4 final row3')
 schedule=derive_schedule(ids,plan['args'],env);require(schedule==report['declared_schedule'],'Config-derived normal route changed')
 dispatch=bind_dispatch_source(Path(plan['engine_root'])/'source');require(dispatch==report['source_dispatch_binding']==report['post_dispatch_binding'],'Current source3 dispatch changed')
 got,targets,packets,target_binding=own.load_target(native_root,plan,4,binding['requests']);require(got==ids and target_binding==report['native_observation_binding'],'Current native target raw binding changed')
 identity_path=original_root/'post-original-model-identity.json';identity=own.identity_admission(identity_path,lock_path,shards,report['computation_terminal_epoch']);require(identity==report['post_original_identity'] and identity['finished']<=report['finished_epoch'],'Completed NEW postCPU full4 proof changed')
 runtime=report['runtime'];require(runtime['native_config_env']==env and runtime['numpy_version']==np.__version__ and runtime['numpy_module_sha256']==sha(np.__file__) and runtime['numpy_binary_sha256']=={str(p):sha(p) for p in sorted(Path(np.__file__).parent.rglob('*.so'))},'NumPy/native config identity changed')
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):require(runtime['env'].get(name)==os.environ.get(name)=='1','Matched single BLAS thread required '+name)
 saved={name:own.load_own_array(original_root,row) for name,row in report['owned_arrays'].items() if name.startswith('p3_')}
 require(report['completed_original_replay'].get('allrows_and_finalstates_replay_bitwise_equal') is True,'Completed whole own reference replay checks required')
 return report,plan,binding,targets,packets,saved,identity_path

def state_from_physical(recurrent,conv):
 require(np.shape(recurrent)==(128,48,128) and np.shape(conv)==(10240,3),'Explicit native physical state shape differs')
 return {'recurrent':np.asarray(recurrent,dtype='<f4').transpose(1,0,2).copy(),'conv':np.asarray(conv,dtype='<f4').T.copy()}

def mixer_case(model,mixed,state,route):
 require(route=='verifier' and np.shape(mixed)==(2560,) and np.isfinite(mixed).all(),'Actual final T1 verifier finite input required')
 model.projector.events=[]
 after,d=model.gdn.mixer(np.asarray(mixed,dtype='<f4').copy(),{k:v.copy() for k,v in state.items()},route)
 fields={'gdn_qkv':d['qkv'],'gdn_z':d['z'].reshape(-1),'gdn_decay_beta':np.stack([d['log_decay'],d['beta']]),'gdn_normalized_qkv':np.concatenate([d['q'].reshape(-1),d['k'].reshape(-1),d['v'].reshape(-1)]),'gdn_output_gated':d['normalized_gated_mathematical_F32'].reshape(-1),'gdn_block_output':d['output'],'gdn_state_after':physical_recurrent(after['recurrent']),'gdn_conv_after':physical_conv(after['conv'])}
 packets={'attn_input_q81':encode(np.asarray(mixed,dtype='<f4').tobytes(),1,2560),'gdn_output_q81':encode(np.asarray(fields['gdn_output_gated'],dtype='<f4').tobytes(),1,6144)}
 return fields,packets,list(model.projector.events)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('native-run-root','original-run-root','output'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--original-report-sha256',required=True);ap.add_argument('--native-beforestate-case',action='store_true');a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 report={'schema':1,'scope':'CONDITIONAL original GDN final row3; never full owned reference','captured_inputs_used':True,'captured_inputs_in_full_reference':False,'numeric_pass_claim':False,'tolerance_gate':None,'full_model_math_qualified':False,'started_epoch':time.time(),'errors':[],'cases':[]};admitted=False;terminal=None
 lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[own.ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')]
 def watch(label):
  try:return own.guard(shards[2])
  except Exception:report['preserved_'+label]=own.preserve(shards[2],out,label);raise
 try:
  report['dependencies']=dependencies();prior,plan,binding,targets,target_packets,saved,identity_path=preflight(a.native_run_root,a.original_run_root.resolve(),a.original_report_sha256,lock_path,shards);report['original_report_sha256']=a.original_report_sha256;report['native_finalized_binding']=binding;report['pre_original_identity']=prior['post_original_identity'];report['initial_pages']=watch('initial');write(out/'report.json',report);admitted=True
  provider=BoundOriginalFull48Rows(own.FOUNDATION,identity_path);roles={name:{'type':tensor['type'],'shape':tensor['shape_ggml_order'],'offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'source_path':file['path'],'source_stat':sig} for name,(file,tensor,sig) in provider.reader.tensors.items() if name in prior['original_role_binding']};require(roles==prior['original_role_binding'],'Current original L0 roles/shard stats changed')
  env=prior['runtime']['native_config_env'];source=Path(plan['engine_root'])/'source';model=OwnedLayer0GdnReplay(provider,sha(identity_path),plan['args'],env,source,tile_bytes=64<<20);schedule=derive_schedule(prior['declared_schedule']['accepted_ids'],plan['args'],env);window=next(w for w in schedule['windows'] if w['position']<=3<w['position']+w['rows']);require(window['rows']==1 and window['position']==3,'Actual final T1 row3 window required');route=window['math_route'];state=state_from_physical(saved['p3_gdn_state_before'],saved['p3_gdn_conv_before'])
  lanes=[('saved_own_input_and_state_replay',saved['p3_attn_mixed'],state),('native_mixed_own_beforestate_CONDITIONAL',targets['attn_mixed'],state)]
  if a.native_beforestate_case:lanes.append(('native_mixed_native_beforestate_CONDITIONAL',targets['attn_mixed'],state_from_physical(targets['gdn_state_before'],targets['gdn_conv_before'])))
  for lane,mixed,before in lanes:
   fields,packets,events=mixer_case(model,mixed,before,route);baseline=lane=='saved_own_input_and_state_replay';checks={name:own.compare_observation(value,targets[name]) for name,value in fields.items()};replay={name:own.compare_observation(value,saved['p3_'+name]) for name,value in fields.items()};require(not baseline or all(v['bitwise_equal'] for v in replay.values()),'Saved unchanged own GDN baseline replay differs')
   own_packets=prior['owned_packets'];packet_checks={}
   for name,raw in packets.items():
    saved_packet=own_packets['p3_'+name];path=Path(saved_packet['path']);require(path.resolve().parent==a.original_run_root.resolve() and not path.is_symlink(),'Saved own packet path escaped');old=path.read_bytes();require(len(old)==saved_packet['bytes'] and hashlib.sha256(old).hexdigest()==saved_packet['sha256']==sha(path),'Saved own packet changed');require(not baseline or raw==old,'Saved own baseline packet differs');file=out/(lane+'-'+name+'.q81');file.write_bytes(raw);packet_checks[name]={'native_bytes_equal':raw==target_packets[name],'saved_own_bytes_equal':raw==old,'path':str(file),'sha256':sha(file),'bytes':len(raw)}
   arrays={name:own.save_array(out,lane+'-'+name+'.f32',value) for name,value in fields.items()};report['cases'].append({'lane':lane,'position':3,'source_config_window':window,'native_route_used_as_input':False,'native_mixed_used_as_conditional_input':not baseline,'native_beforestate_used_as_conditional_input':lane.endswith('native_beforestate_CONDITIONAL'),'physical_state_permutation':'native[i,head,j] -> logical[head,i,j]; conv[channel,history] -> logical[history,channel]','target_comparisons':checks,'saved_own_comparisons':replay,'packet_comparisons':packet_checks,'arrays':arrays,'projection_events':events,'complete_owned_reference':False,'actual_T_group_rounding_qualified':False});write(out/'report.json',report)
  require(len(report['cases'])==2+int(a.native_beforestate_case),'Exact conditional case count differs');terminal=time.time();report['computation_terminal_epoch']=terminal
  _,_,post_plan,post_binding=own.num10_admission(a.native_run_root);require(post_plan==plan and post_binding==binding,'Post diagnostic NUM10 evidence changed');dependencies();require(bind_dispatch_source(source)==prior['post_dispatch_binding'],'Post source3 code/stat changed');preflight(a.native_run_root,a.original_run_root.resolve(),a.original_report_sha256,lock_path,shards);report['post_pages']=watch('post')
 except Exception as error:terminal=time.time();report['errors'].append(str(error));report['traceback']=traceback.format_exc()
 finally:
  if admitted:
   try:
    after=terminal or time.time();report['computation_terminal_epoch']=after;report['before_full4_pages']=watch('before-full4');identity=own.full_buffered_identity(lock_path,lock,shards,out/'post-conditional-model-identity.json',after);require(identity['passed'],'NEW conditional postCPU full4 hash failed');report['post_CPU_source4']=own.identity_admission(out/'post-conditional-model-identity.json',lock_path,shards,after);report['final_pages']=watch('final')
   except Exception as error:report['errors'].append('Post source proof: '+str(error));report['preserved_failure']=own.preserve(shards[2],out,'failure')
  report['finished_epoch']=time.time();report['status']='CONDITIONAL exploratory complete; NO numerical qualification' if not report['errors'] else 'FAILED; preserve evidence';write(out/'report.json',report)
 print(json.dumps({'status':report['status'],'errors':report['errors'],'cases':len(report['cases']),'numeric_pass_claim':False}));return int(bool(report['errors']))
if __name__=='__main__':raise SystemExit(main())
