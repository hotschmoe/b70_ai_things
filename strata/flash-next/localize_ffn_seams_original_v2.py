#!/usr/bin/env python3
"""Conditional original FFN seam diagnostic; no owned-model or numerical PASS.
Native attention is an explicitly captured conditional input only. No native
router IDs, mixed activations or packets are observed or fed to original math.
"""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import explore_full48_original_routes_v1 as original
from full48_owned_composition_storage_v2 import BoundOriginalFull48Rows,RouteAwareFull48OwnedComposition,derive_schedule,bind_dispatch_source
from independent_q8_1_activation_v1 import encode

HERE=Path(__file__).resolve().parent
LAYERS=(1,2,4,10)
POSITION=1
sha,read,write,require=original.sha,original.read,original.write,original.require

def case_roster():
 return {(POSITION,layer,lane) for layer in LAYERS for lane in ('native_attention_conditional','owned_attention_replay')}

def source_binding():
 plan=read(HERE/'ffn-seam-localization-original-source-plan-v2.json')
 for name,digest in plan['files'].items():require(sha(original.ROOT/name)==digest,'Frozen seam source changed '+name)
 return {'seam_dependencies':plan['files'],'original_dependencies':original.dependency_binding()}


def own_array(root,row):
 path=Path(row['path']);require(path.resolve().parent==root.resolve() and not path.is_symlink(),'Own array escaped original run')
 require(row['encoding']=='LE_F32' and row['bytes']==int(np.prod(row['shape']))*4 and sha(path)==row['sha256'],'Own array original byte/shape binding differs')
 raw=path.read_bytes();require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Own array changed/truncated during read');value=np.frombuffer(raw,dtype='<f4').reshape(row['shape']).copy()
 require(np.isfinite(value).all(),'Own array nonfinite');return value


def preflight(native_root,original_root,report_sha,lock_path,shards):
 dependencies=source_binding();report_path=original_root/'report.json';require(sha(report_path)==report_sha,'Completed original report SHA changed')
 raw=report_path.read_bytes();require(hashlib.sha256(raw).hexdigest()==report_sha,'Completed original report changed during read');report=json.loads(raw)
 require(report.get('status')=='exploratory complete; NO numerical qualification' and report.get('errors')==[] and report.get('numeric_pass_claim') is False and report.get('full_model_math_qualified') is False and report.get('tolerance_gate') is None,'Completed original exploratory/post-CPU proof required')
 require(report['dependency_sha256']==dependencies['original_dependencies'],'Original computation frozen dependency binding differs')
 parent,child,plan,binding=original.finalized_binding(native_root)
 for key in ('parent_sha256','child_sha256','plan_sha256','source_plan_sha256'):
  require(binding[key]==report['native_finalized_binding'][key],'Original/native finalized source association differs '+key)
 env=read(Path(plan['prepared'])/'server-config.json')['env'];ids=report['declared_schedule']['accepted_ids'];prefix=len(ids);require(prefix in (2,4,8),'Selected absolute row1 requires completed prefix2/4/8')
 schedule=derive_schedule(ids,plan['args'],env);require(schedule==report['declared_schedule'],'Saved route schedule differs from admitted source config')
 dispatch=bind_dispatch_source(Path(plan['engine_root'])/'source');require(dispatch==report['source_dispatch_binding']==report['post_dispatch_binding'],'Original source3 code/stat binding changed')
 ids,native,observed=original.native_prefix(native_root,plan,prefix,schedule)
 require(observed==report['native_observation_binding'],'Current native observation differs from original run')
 terminal=report['computation_terminal_epoch'];require(0<terminal<=report['finished_epoch'],'Original computation chronology invalid')
 identity_path=original_root/'post-original-model-identity.json';identity=original.identity_admission(identity_path,lock_path,shards,terminal)
 require(identity==report['post_original_identity'] and identity['finished']<=report['finished_epoch'],'Original NEW post-CPU complete source4 receipt differs')
 arrays=report['exploration']['arrays'];require(set(arrays)>={'head','native_head'}|{'p%d_l%d_%s'%(pos,layer,phase) for pos in range(prefix) for layer in range(48) for phase in ('input','attention','ffn')},'Original complete prefix/all48 saved phase roster absent')
 owned={name:own_array(original_root,row) for name,row in arrays.items()}
 for layer in LAYERS:
  for phase in ('attention','ffn'):require(owned['p%d_l%d_%s'%(POSITION,layer,phase)].shape==(4,2560),'Original FFN seam geometry differs')
 require(report['runtime']['native_config_env']==read(Path(plan['prepared'])/'server-config.json')['env'],'Original/native math environment changed')
 require(report['runtime']['numpy_version']==np.__version__ and report['runtime']['numpy_module_sha256']==sha(np.__file__),'Original/current NumPy module identity differs')
 binaries={str(path):sha(path) for path in sorted(Path(np.__file__).parent.rglob('*.so'))};require(report['runtime']['numpy_binary_sha256']==binaries,'Original/current complete NumPy binary roster changed')
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):require(report['runtime']['env'].get(name)==os.environ.get(name)=='1','Matched original/seam single BLAS thread required '+name)
 return report,plan,native,owned,identity_path,{'dependencies':dependencies,'original_report_sha256':report_sha,'native_finalized':binding,'original_post_CPU_source4':identity}


def conditional_case(model,ids,layer,attention,target):
 require(layer in LAYERS and len(ids) in (2,4,8),'Only absolute row1 layers1/2/4/10 prefix2/4/8 supported')
 schedule=derive_schedule(ids,model.args,dict(model.env));window=next(w for w in schedule['windows'] if w['position']<=POSITION<w['position']+w['rows']);route=window['math_route']
 attention=np.asarray(attention,dtype='<f4');target=np.asarray(target,dtype='<f4')
 require(attention.shape==target.shape==(4,2560) and np.isfinite(attention).all() and np.isfinite(target).all(),'Conditional finite seam shape differs')
 model.projector.events=[];hc=model.hc_read(attention,'blk.%d.hc_ffn_'%layer)
 mixed=np.asarray(hc['mixed'],dtype='<f4');packet=encode(mixed.tobytes(),1,len(mixed));ffn=model.ffn[layer].run(mixed,route);output=model.hc_write(attention,ffn['block_output'],hc['inject'])
 return output,{'position':POSITION,'layer':layer,'route':route,'config_derived_window':window,'native_route_used_as_math_input':False,'actual_window_group_shape_rounding_qualified':False,'attention_sha256':hashlib.sha256(attention.tobytes()).hexdigest(),'independently_derived_mixed_sha256':hashlib.sha256(mixed.tobytes()).hexdigest(),'independently_derived_mixed_q8_packet_sha256':hashlib.sha256(packet).hexdigest(),'independently_derived_router_ids':ffn['router_ids_owned'].tolist(),'independently_derived_router_weights':ffn['router_weights_owned'].tolist(),'packet_events_independently_derived':list(model.projector.events),'output_metrics':original.compare_observation(output,target),'native_mixed_or_packet_observed':False,'native_selected_ids_used':False,'complete_owned_reference':False,'full_model_math_qualified':False,'numeric_pass_claim':False,'tolerance_gate':None}


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('native-run-root','original-run-root','output'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--original-report-sha256',required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 report={'schema':2,'scope':'Conditional captured attention FFN seam localization; NEVER complete owned reference','captured_inputs_used':True,'native_selected_ids_used':False,'native_mixed_or_packet_observed':False,'numeric_pass_claim':False,'full_model_math_qualified':False,'tolerance_gate':None,'started_epoch':time.time(),'errors':[],'cases':[]}
 lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[original.ROOT/lock['destination']/row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')];admitted=False;terminal=None
 def watch(label):
  try:return original.guard(shards[2])
  except Exception:
   report['preserved_'+label]=original.preserve(shards[2],out,label);raise
 try:
  prior,plan,native,owned,identity_path,binding=preflight(a.native_run_root,a.original_run_root.resolve(),a.original_report_sha256,lock_path,shards);report['bindings']=binding;report['initial_known_pages']=watch('admission');write(out/'report.json',report)
  admitted=True;provider=BoundOriginalFull48Rows(original.FOUNDATION,identity_path)
  roles={name:{'type':tensor['type'],'shape':tensor['shape_ggml_order'],'offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'source_path':file['path'],'source_stat':sig} for name,(file,tensor,sig) in provider.reader.tensors.items()};require(roles==prior['original_role_binding'],'Original complete role identity changed before payload')
  env=prior['runtime']['native_config_env'];source_root=Path(plan['engine_root'])/'source';model=RouteAwareFull48OwnedComposition(provider,sha(identity_path),plan['args'],env,source_root,tile_bytes=64<<20);ids=prior['declared_schedule']['accepted_ids']
  report['runtime']={'numpy_version':np.__version__,'numpy_module_sha256':sha(np.__file__),'env':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','MKL_','OPENBLAS_','NUMEXPR_'))},'numeric_runtime_qualified':False}
  for layer in LAYERS:
   for lane,inputs,targets in [('native_attention_conditional',native,native),('owned_attention_replay',owned,owned)]:
    predicted,case=conditional_case(model,ids,layer,inputs['p%d_l%d_attention'%(POSITION,layer)],targets['p%d_l%d_ffn'%(POSITION,layer)]);case.update(input_lane=lane,captured_inputs_used=lane=='native_attention_conditional',attention_input_comparison=original.compare_observation(native['p%d_l%d_attention'%(POSITION,layer)],owned['p%d_l%d_attention'%(POSITION,layer)]),output=original.save_array(out,'l%d-%s.f32'%(layer,lane),predicted));require(lane!='owned_attention_replay' or case['output_metrics']['bitwise_equal'],'Saved independent own FFN replay changed');report['cases'].append(case);write(out/'report.json',report)
  require(len(report['cases'])==8 and {(case['position'],case['layer'],case['input_lane']) for case in report['cases']}==case_roster(),'Exact conditional8 case roster required');report['post_dispatch_binding']=bind_dispatch_source(source_root);require(report['post_dispatch_binding']==prior['post_dispatch_binding'],'Source3 changed during seam computation');terminal=time.time();report['computation_terminal_epoch']=terminal;report['post_known_pages']=watch('post-computation')
 except Exception as error:
  terminal=time.time();report['errors'].append(str(error));report['traceback']=traceback.format_exc()
 finally:
  if admitted:
   try:
    terminal=terminal or time.time();report['computation_terminal_epoch']=terminal;identity=original.full_buffered_identity(lock_path,lock,shards,out/'post-seam-model-identity.json',terminal);require(identity['passed'],'NEW post-seam full4 publisher scan failed');report['post_CPU_source4']=original.identity_admission(out/'post-seam-model-identity.json',lock_path,shards,terminal);report['final_known_pages']=watch('final')
   except Exception as error:
    report['errors'].append('Post-source proof: '+str(error))
    try:report['preserved_source_pages']=original.preserve(shards[2],out,'failure')
    except Exception as error:report['errors'].append('Preservation: '+str(error))
  report['finished_epoch']=time.time();report['status']='conditional exploratory complete; NO numerical qualification' if not report['errors'] else 'FAILED prerequisite/execution/source proof';write(out/'report.json',report)
 print(json.dumps({'status':report['status'],'errors':report['errors'],'cases':len(report['cases']),'full_model_math_qualified':False}));return 1 if report['errors'] else 0
if __name__=='__main__':raise SystemExit(main())
