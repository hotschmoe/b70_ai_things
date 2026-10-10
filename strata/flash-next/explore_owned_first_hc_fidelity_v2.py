#!/usr/bin/env python3
"""Root-only original token HC block control. Native fields are targets only."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import owned_first_hc_fidelity_v2 as localization
import first_hc_computation_chronology_v2 as chronology
import full48_owned_hc_fma_refinement_v1 as refined
import qualify_hc35_host_bulk_runtime_v1 as bulk
from independent_q8_1_activation_v1 import encode
from explore_owned_layer0_gdn_num10_v1 import num10_admission,load_target,FOUNDATION
from full48_owned_composition_storage_v2 import BoundOriginalFull48Rows
from conditional_hc_projection35_v1 import host_admission
from explore_full48_original_prefix1_v4 import identity_admission,compare_observation,save_array,sha,read,write,require
from source_page_watchdog_v3 import guard,preserve
from run_source_upload_oracle_full_v2 import full_buffered_identity
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def compare_targets(rows,targets,packets):
 return {lane:{name:compare_observation(values[name],targets[field]) for name,field in localization.TARGET_FIELDS.items()} for lane,values in rows.items()}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-run-root',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--bulk-build-root',type=Path,required=True);p.add_argument('--conditional-gdn-seam',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 report={'schema':1,'control_generation':2,'status':'incomplete','started_epoch':time.time(),'captured_inputs_used':False,'conditional_captured_inputs_requested':a.conditional_gdn_seam,'numeric_pass_claim':False,'full_model_math_qualified':False,'tolerance_gate':None,'errors':[]};lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];admitted=False;terminal=None
 def watch(label):
  try:return guard(shards[2])
  except Exception:report['preserved_'+label]=preserve(shards[2],out,label);raise
 def host_current():return bulk.finalized_binding(a.bulk_build_root)
 try:
  report['chronology_source_contract']=chronology.source_contract(__file__);report['localization_source_pre']=localization.source_binding();report['initial_pages']=watch('initial-failure');parent,child,plan,binding=num10_admission(a.native_run_root);report['native_finalized_binding']=binding;boundary=max(parent['child_terminal_epoch'],child['finished_epoch'],read(a.native_run_root/'post-health.json')['finished_epoch']);report['pre_original_identity']=identity_admission(a.model_identity,lock_path,shards,boundary)
  ids,targets,packets,target_binding=load_target(a.native_run_root,plan,1,binding['requests']);report['native_output_targets']=target_binding;env=read(Path(plan['prepared'])/'server-config.json')['env'];source=Path(plan['engine_root'])/'source';report['source_binding']=refined.source_binding(source);report['host_binding']=host_current();report['pre_pages']=watch('pre-failure');admitted=True
  report['CPU_runtime']={'numpy_version':np.__version__,'numpy_module_sha256':sha(np.__file__),'numpy_binary_sha256':{str(q):sha(q) for q in sorted(Path(np.__file__).parent.rglob('*.so'))},'native_math_env':env,'compiled_helper_executed_by_this_control':True}
  provider=BoundOriginalFull48Rows(FOUNDATION,a.model_identity);model=refined.OwnedHcFmaRefinement(provider,sha(a.model_identity),plan['args'],env,source,a.bulk_build_root);owned=localization.compute_owned(model,ids);report['owned_HC_terminal_epoch']=time.time()
  report['original_role_binding']={name:{'type':t['type'],'shape':t['shape_ggml_order'],'offset':t['absolute_offset'],'packed_bytes':t['packed_bytes'],'source_path':f['path'],'source_stat5':sig} for name,(f,t,sig) in provider.reader.tensors.items() if name=='token_embd.weight' or name.startswith('blk.0.hc_attn_') or a.conditional_gdn_seam and name.startswith('blk.0.')};report['provenance']={k:v for k,v in owned.items() if k not in ('embedding_owned','residual_owned','candidate','prior')};report['owned_arrays']={name:save_array(out,'own-'+name+'.f32',owned[name]) for name in ('embedding_owned','residual_owned')};report['comparisons']=compare_targets({lane:owned[lane] for lane in ('candidate','prior')},targets,packets);report['owned_packets']={}
  require(np.asarray(targets['residual_input'],dtype='<f4').tobytes()==owned['residual_owned'].tobytes(),'Actual first-row target residual differs from own original embedding')
  for lane in ('candidate','prior'):
   for name,value in owned[lane].items():report['owned_arrays'][lane+'_'+name]=save_array(out,'own-'+lane+'-'+name+'.f32',value)
   packet=encode(np.asarray(owned[lane]['mixed'],dtype='<f4').tobytes(),1,2560);path=out/('own-'+lane+'-mixed.q81');path.write_bytes(packet);report['owned_packets'][lane]={'path':str(path),'sha256':sha(path),'bytes':len(packet),'native_packet_sha256':hashlib.sha256(packets['attn_input_q81']).hexdigest(),'bytes_equal':packet==packets['attn_input_q81'],'different_bytes':sum(a!=b for a,b in zip(packet,packets['attn_input_q81']))}
  if a.conditional_gdn_seam:
   seam=localization.conditional_gdn(model,targets['attn_mixed']);report['conditional_gdn_seam']={k:v for k,v in seam.items() if k not in ('output','state','detail')};report['conditional_gdn_seam']['output']=save_array(out,'conditional-gdn-output.f32',seam['output']);report['conditional_gdn_seam']['comparison']=compare_observation(seam['output'],targets['gdn_block_output']);report['conditional_gdn_seam']['states']={k:save_array(out,'conditional-gdn-state-'+k+'.f32',v) for k,v in seam['state'].items()};report['conditional_GDN_terminal_epoch']=time.time()
  terminal=time.time();report['computation_terminal_epoch']=terminal;report['observed_computation_chronology']=chronology.observed(report)
  report['localization_source_post']=localization.source_binding();require(report['localization_source_post']==report['localization_source_pre'],'Localization source changed');report['post_host_binding']=host_current();require(report['post_host_binding']==report['host_binding'],'Host runtime changed');_,_,postplan,post=num10_admission(a.native_run_root);require(postplan==plan and post==binding,'Native finalized target changed');require(refined.source_binding(source)==report['source_binding'],'Current source changed');report['post_pages']=watch('post-failure');identity_admission(a.model_identity,lock_path,shards,boundary)
 except Exception as error:terminal=time.time();report['errors'].append(str(error));report['traceback']=traceback.format_exc()
 finally:
  try:
   require(admitted,'Full4 skipped before original admission');after=terminal or time.time();report['before_full4_pages']=watch('before-full4-failure');identity=full_buffered_identity(lock_path,lock,shards,out/'post-original-model-identity.json',after);require(identity['passed'],'NEW postCPU all4 source proof failed');report['post_original_identity']=identity_admission(out/'post-original-model-identity.json',lock_path,shards,after);report['final_pages']=watch('final-failure')
  except Exception as error:
   report['errors'].append('Post-source proof: '+str(error))
   if admitted:
    try:report['preserved_post_failure']=preserve(shards[2],out,'post-failure')
    except Exception as more:report['errors'].append(str(more))
  report['finished_epoch']=time.time();report['status']='owned exploratory complete; NO device/math qualification' if not report['errors'] else 'FAILED source/admission/computation';write(out/'report.json',report)
 print(json.dumps({'status':report['status'],'errors':report['errors'],'numeric_pass_claim':False}));return int(bool(report['errors']))
if __name__=='__main__':raise SystemExit(main())
