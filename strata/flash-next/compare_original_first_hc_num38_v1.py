#!/usr/bin/env python3
"""Root-only bounded original embedding->HC->packet; historical targets only."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
from serial37_canonical_json_v3 import read_unique,canonical
from original_first_gdn_layer_v1 import OriginalTensorRows
from full48_owned_composition_storage_v1 import OriginalStorageProjector
import original_native_hc_f32_v1 as hc
from source43_num10_original_reference_v3 import metrics,consume
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'original-first-hc-num38-source-plan-v1.json';FOUNDATION=HERE/'original-gguf-reference-foundation-plan-v1.json'
REPORT_SHA='23e5c2f9c5d09195ae859e99a103517e6b371b8821028dfff503323a59d75248'
SOURCE_SHA='97170326d6e09770ce572a2618afe49a194c4d8a3f2a3222fded1464d5411da9'
SHAPES={'residual_input':(4,2560),'attn_hc_normalized':(4,2560),'attn_hc_low':(320,),'attn_hc_gate':(4,2560),'attn_hc_inject':(4,),'attn_mixed':(2560,)}
require=hc.require

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_binding():
 value=read_unique(PLAN)
 for name,want in value['files'].items():require(sha(ROOT/name)==want,'Bounded comparison source changed '+name)
 return sha(PLAN)
def artifact(root,report,path):
 path=Path(path);relative=str(path.relative_to(root));require(path.is_file()and not path.is_symlink()and not any(p.is_symlink()for p in path.parents)and sha(path)==report['artifacts'][relative],'Original consumed target artifact changed '+relative);return path

def targets(root):
 root=Path(root).resolve();report=read_unique(root/'report.json');require(sha(root/'report.json')==REPORT_SHA and report['passed']is True and report['errors']==[]and report['source_plan_sha256']==SOURCE_SHA,'Exact historical passed NUM38 producer report required')
 require(sha(artifact(root,report,root/'source-plan.snapshot.json'))==SOURCE_SHA and sha(artifact(root,report,root/'plan.snapshot.json'))==report['plan_sha256'],'Original source/corpus snapshots changed')
 source=read_unique(root/'source-plan.snapshot.json');require(sha(artifact(root,report,root/'producer.snapshot.py'))==source['files']['strata/flash-next/run_source44_num38_zero_v2.py'],'Truthful recorded source44 producer required')
 directory=root/'source44-num38-ON';recipe=read_unique(artifact(root,report,directory/'recipe.json'));result=read_unique(artifact(root,report,directory/'result.json'));phase=read_unique(artifact(root,report,directory/'phase_receipt.json'));artifact(root,report,directory/'command.json')
 require(type(recipe['engine_generation'])is int and recipe['engine_generation']==44 and type(recipe['materialized_mode'])is int and recipe['materialized_mode']==1 and recipe['NUM38_enabled']is True and recipe['observer_enabled']is True,'Recorded actual44 mode1 NUM38 recipe required')
 require(result['passed']is True and result['errors']==[]and result['error']is None and type(result['engine_rc'])is int and result['engine_rc']==result['return_code']==0 and result['forced_cleanup']is False and result['removal']['removed']is True and canonical(result)==canonical(report['results']['source44-num38-ON']),'Original recorded normal capture result required')
 require(phase['passed']is True and phase['error']is None and phase['return_code']==0 and phase['result_sha256']==sha(directory/'result.json')and phase['command_sha256']==sha(directory/'command.json'),'Original recorded phase receipt required')
 rows=read_unique(artifact(root,report,directory/'requests.json'));require(len(rows)==4 and [r['ordinal']for r in rows]==[1,2,3,4],'Four original request rows required');out=[]
 for row in rows:
  raw=row['raw'];require(raw['ids']==[248045]and type(raw['fresh'])is int and raw['fresh']==1 and raw['pin']is None and raw['stop_sent']is False and raw['cancel_requested']is None,'Original four fresh literalprefix1 targets only')
  require(canonical(raw)==canonical(read_unique(artifact(root,report,directory/('request-'+str(row['ordinal'])+'.json')))),'Individual original request changed')
  files=list((directory/'layer0').glob('l0q8-*-r'+str(row['ordinal'])+'-s0.json'));require(len(files)==1,'Unique original frame metadata required');path=artifact(root,report,files[0]);frame=read_unique(path)
  for key,want in(('schema',1),('request',row['ordinal']),('stage',0),('layer',0),('rows',1),('position',0),('token',248045),('reused',0),('numerical_source_generation',44)):
   require(type(frame[key])is int and frame[key]==want,'Exact typed historical frame differs '+key)
  pid=frame['pid'];require(type(pid)is int and pid==1 and frame['completed_nonce']==((pid<<32)|row['ordinal'])and frame['gen_ids']==[248045]and frame['binding_sha256']==SOURCE_SHA and frame['gdn_core_output_observed']is True and frame['producer_mapping_verified']is True and frame['request_replay_marker_verified']is True and frame['complete_preregistered_layout']is True,'Original metadata owner/layout/source mapping differs')
  fields=frame['fields'];require(len(fields)==38 and len({f['name']for f in fields})==38,'Complete unique recorded raw38 roster required');values={};packet=None
  for field in fields:
   relative=Path(field['file']);require(relative.is_absolute()and str(relative.parent)=='/results/layer0'and '..'not in relative.parts,'Original raw38 target path escaped');path=artifact(root,report,directory/'layer0'/relative.name)
   if field['name']in SHAPES:
    shape=SHAPES[field['name']];require(field['bytes']==int(np.prod(shape))*4 and field['encoding']=='LE_F32['+','.join(map(str,shape))+']','Original HC target encoding/extent differs');data=consume(path,field['bytes'],report['artifacts'][str(path.relative_to(root))]);values[field['name']]=np.frombuffer(data,dtype='<f4').reshape(shape).copy();require(np.isfinite(values[field['name']]).all(),'Original target nonfinite')
   elif field['name']=='attn_input_q81':
    require(field['bytes']==2880,'Original first packet extent differs');packet=consume(path,2880,report['artifacts'][str(path.relative_to(root))])
   elif field['name']in('gdn_state_before','gdn_conv_before'):
    data=consume(path,field['bytes'],report['artifacts'][str(path.relative_to(root))]);require(not any(data),'Recorded literalprefix1 recurrent/conv state must be bytezero')
  require(set(values)==set(SHAPES)and packet is not None,'All first HC targets and packet required');out.append({'request':row['ordinal'],'values':values,'packet':packet})
 return out,{'producer_report_sha256':REPORT_SHA,'recorded_source_plan_sha256':SOURCE_SHA,'comparison_target_only':True,'current_native_qualification_transferred':False,'current_registry_gate_reexecuted':False}

def execute(args):
 output=args.output.resolve();require(not output.exists(),'New external comparison output required');output.mkdir();started=time.time();report={'schema':1,'passed':False,'errors':[],'started_epoch':started,'full_model_math_qualified':False,'GPU_execution':False,'captured_inputs_used':False,'full48_computation_performed':False};error=None
 try:
  source=source_binding();require(sha(args.model_identity)==args.expected_identity_sha256,'Exact freshly supplied full4 receipt required');captured,binding=targets(args.actor_root);report['historical_targets']=binding
  provider=OriginalTensorRows(FOUNDATION,args.model_identity);projector=OriginalStorageProjector(provider,1<<20);projector.require('token_embd.weight','Q8_0',(2560,248320));embedding=np.asarray(provider.rows('token_embd.weight',[248045])[0],dtype='<f4');residual=np.broadcast_to(embedding,(4,2560)).copy()
  own=hc.hc_read_original(projector,residual,'blk.0.hc_attn_',provider.source_contract['original_rms_epsilon']);packet=hc.owned_first_packet(own['mixed']);fields={'residual_input':residual,'attn_hc_normalized':own['normalized'],'attn_hc_low':own['low_silu'],'attn_hc_gate':own['gate'],'attn_hc_inject':own['inject'],'attn_mixed':own['mixed']}
  report['independent_computations']=1;report['actual_literal_target_comparisons']=4;report['same_stateless_HC_result_compared_to_four_identical_fresh_inputs']=True;report['comparisons']=[{'request':row['request'],'fields':{name:metrics(value,row['values'][name])for name,value in fields.items()},'first_packet_bitwise_equal':packet==row['packet']}for row in captured]
  for name,value in fields.items():(output/(name+'.f32')).write_bytes(np.asarray(value,dtype='<f4').tobytes())
  (output/'own-first-packet.q81').write_bytes(packet);report['source_plan_sha256']=source;report['model_identity_sha256']=args.expected_identity_sha256;report['owned_original_parameters_only']=True;report['host_rsqrt_expf_device_intrinsics_qualified']=False;targets(args.actor_root);require(source_binding()==source and sha(args.model_identity)==args.expected_identity_sha256,'Closing small input/source bindings changed')
 except BaseException as exc:error=exc;report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  # Fresh close after all target/provider reads, including failure; no oldcapture readmission.
  try:
   from run_source44_num38_original_reference_v2 import closing_identity
   report['postCPU_full4']=closing_identity(output,time.time())
  except BaseException as exc:report['errors'].append('fresh-closing: '+type(exc).__name__+': '+str(exc))
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['artifacts']={str(p.relative_to(output)):sha(p)for p in sorted(output.rglob('*'))if p.is_file()and p.name!='report.json'};(output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='ascii')
 require(report['passed'],'Bounded firstHC diagnostic or fresh source closing failed');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--actor-root',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--expected-identity-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();require(__debug__,'Unoptimized Python required')
 for sig in(signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,lambda *_:(_ for _ in()).throw(InterruptedError('Controlled bounded CPU reference interruption')))
 execute(args)
if __name__=='__main__':main()
