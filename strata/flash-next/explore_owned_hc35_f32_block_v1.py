#!/usr/bin/env python3
"""Root-only original token HC block control. Native fields are targets only."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import owned_hc35_f32_block_control_v1 as control
from explore_owned_layer0_gdn_num10_v1 import num10_admission,load_target,FOUNDATION
from full48_owned_composition_storage_v2 import BoundOriginalFull48Rows
from conditional_hc_projection35_v1 import host_admission
from explore_full48_original_prefix1_v4 import identity_admission,compare_observation,save_array,sha,read,write,require
from source_page_watchdog_v3 import guard,preserve
from run_source_upload_oracle_full_v2 import full_buffered_identity
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def compare_targets(row,targets,packets):
 columns=list(control.COLUMNS);checks={'normalized':compare_observation(row['normalized_owned_candidate'],targets['attn_hc_normalized']),'low':compare_observation(row['post_silu_owned_candidate'],targets['attn_hc_low']),'gate_block':compare_observation(row['gate_owned_candidate'],targets['attn_hc_gate'][:,columns]),'inject':compare_observation(row['inject_owned_candidate'],targets['attn_hc_inject'])}
 packet=packets['attn_input_q81'][control.BLOCK*36:(control.BLOCK+1)*36]
 for name,value in row['mixed_block_owned_candidates'].items():
  checks[name]=compare_observation(value,targets['attn_mixed'][columns]);own=row['q81_block_owned_candidates'][name];checks[name+'_q81']={'bytes_equal':own==packet,'own_sha256':hashlib.sha256(own).hexdigest(),'native_sha256':hashlib.sha256(packet).hexdigest(),'code_differences':sum(a!=b for a,b in zip(own[4:],packet[4:])),'numeric_gate_assigned':False}
 return checks

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-run-root',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--prefix',type=int,choices=[1,2,4,8],default=4);p.add_argument('--helper',type=Path,required=True);p.add_argument('--helper-build-root',type=Path,required=True);p.add_argument('--host-runtime-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 report={'schema':1,'status':'incomplete','started_epoch':time.time(),'captured_inputs_used':False,'numeric_pass_claim':False,'full_model_math_qualified':False,'tolerance_gate':None,'errors':[]};lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];admitted=False;terminal=None
 def watch(label):
  try:return guard(shards[2])
  except Exception:report['preserved_'+label]=preserve(shards[2],out,label);raise
 def host_current():return host_admission(a.helper,a.helper_build_root,a.host_runtime_root)
 try:
  report['initial_pages']=watch('initial-failure');parent,child,plan,binding=num10_admission(a.native_run_root);report['native_finalized_binding']=binding;boundary=max(parent['child_terminal_epoch'],child['finished_epoch'],read(a.native_run_root/'post-health.json')['finished_epoch']);report['pre_original_identity']=identity_admission(a.model_identity,lock_path,shards,boundary)
  ids,targets,packets,target_binding=load_target(a.native_run_root,plan,a.prefix,binding['requests']);report['native_output_targets']=target_binding;env=read(Path(plan['prepared'])/'server-config.json')['env'];source=Path(plan['engine_root'])/'source';report['source_binding']=control.source_binding(source);report['host_binding']=host_current();report['pre_pages']=watch('pre-failure');admitted=True
  report['CPU_runtime']={'numpy_version':np.__version__,'numpy_module_sha256':sha(np.__file__),'numpy_binary_sha256':{str(q):sha(q) for q in sorted(Path(np.__file__).parent.rglob('*.so'))},'native_math_env':env,'compiled_helper_executed_by_this_control':False}
  provider=BoundOriginalFull48Rows(FOUNDATION,a.model_identity);model=control.OwnedHcBlockControl(provider,sha(a.model_identity),plan['args'],env,source,host_current);owned=model.tokens(ids);terminal=time.time();report['computation_terminal_epoch']=terminal
  report['original_role_binding']={name:{'type':t['type'],'shape':t['shape_ggml_order'],'offset':t['absolute_offset'],'packed_bytes':t['packed_bytes'],'source_path':f['path'],'source_stat5':sig} for name,(f,t,sig) in provider.reader.tensors.items() if name=='token_embd.weight' or name.startswith('blk.0.hc_attn_')};report['provenance']={k:v for k,v in owned.items() if k!='rows'};arrays={}
  for row in owned['rows']:
   pos=row['position'];report.setdefault('rms_owned',{})[str(pos)]=row['rms_owned']
   for name in ('embedding_owned','rsqrt_host_candidate','normalized_owned_candidate','down_owned_candidate','post_silu_owned_candidate','gate_owned_candidate','inject_owned_candidate'):arrays['p%d_%s'%(pos,name)]=save_array(out,'own-p%d-%s.f32'%(pos,name),row[name])
   for name,value in row['mixed_block_owned_candidates'].items():
    arrays['p%d_%s'%(pos,name)]=save_array(out,'own-p%d-%s.f32'%(pos,name),value);packet=row['q81_block_owned_candidates'][name];path=out/('own-p%d-%s.q81'%(pos,name));path.write_bytes(packet);arrays['p%d_%s_q81'%(pos,name)]={'path':str(path),'sha256':sha(path),'bytes':len(packet),'encoding':'Q8_1_LE'}
  report['owned_arrays']=arrays;report['comparisons']=compare_targets(owned['rows'][-1],targets,packets);report['post_host_binding']=host_current();require(report['post_host_binding']==report['host_binding'],'Host runtime changed');_,_,postplan,post=num10_admission(a.native_run_root);require(postplan==plan and post==binding,'Native finalized target changed');require(control.source_binding(source)==report['source_binding'],'Current source changed');report['post_pages']=watch('post-failure');identity_admission(a.model_identity,lock_path,shards,boundary)
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
