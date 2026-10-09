#!/usr/bin/env python3
"""Conditional original FFN consumers from actual33 fields; no own-state/fullmath.
DERIVED hidden is never original raw-fused proof. Per-expert down outputs absent.
"""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from original_first_gdn_layer_v1 import OriginalTensorRows,FirstGdnLayer
from original_math_scalar import metrics,sigmoid
from native_storage_first_gdn_estimate_v1 import bf16_rne
from independent_q8_1_activation_v1 import require_packet_exact,decode
from audit_layer0_capture_lifecycle_v2 import live_blob_bindings
from verify_layer0_packets_v3 import hidden_metrics
ROOT=Path(__file__).resolve().parents[2]
NMSE=1e-6;MAX_NORMALIZED=1e-4;TILE=64<<20
CONTRACT_SHA='86a83df856829674f03b9eb6c80bac0817cb2cdfa4459b3186f5480600e8dfdd'
ROLES={'ffn_gate_shexp.weight':('Q8_0',(2560,640)),'ffn_up_shexp.weight':('Q8_0',(2560,640)),'ffn_down_shexp.weight':('Q8_0',(640,2560)),'ffn_gate_exps.weight':('Q4_K',(2560,640,512)),'ffn_up_exps.weight':('Q4_K',(2560,640,512)),'ffn_down_exps.weight':('Q5_1',(640,2560,512)),'ffn_gate_inp.weight':('F32',(2560,512)),'ffn_gate_inp_shexp.weight':('F32',(2560,)),'hc_ffn_norm.weight':('F32',(10240,)),'hc_ffn_down.weight':('Q8_0',(10240,320)),'hc_ffn_up.weight':('Q8_0',(320,10240)),'hc_ffn_inject.weight':('F32',(10240,4))}
NEEDED={'ffn_mixed','router_logits','router_ids','router_weights','shared_gate_up','shared_hidden_DERIVED','expert_entry_map','expert_gate_up','expert_hidden_DERIVED','ffn_block_output','residual_after_attn','residual_after_ffn'}
PACKETS={'ffn_input_q81':(1,2560),'shared_hq81':(1,640),'expert_hq81':(10,640)}
DEPENDENCIES=['original_first_gdn_layer_v1.py','original_math_scalar.py','original_gguf_reference.py','original_gguf_vector_decoder_v2.py','independent_q8_1_activation_v1.py','native_storage_first_gdn_estimate_v1.py','audit_layer0_capture_lifecycle_v2.py','parse_usm_logical_free_trace.py','layer0_numerical_contract_v1.hpp','verify_layer0_packets_v3.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(actual,expected):
 result=metrics(np.asarray(actual,dtype=np.float64).reshape(-1),np.asarray(expected,dtype=np.float64).reshape(-1));result['passed']=result['nmse']<=NMSE and result['max_normalized']<=MAX_NORMALIZED;return result


def original_roles(provider):
 result={}
 for role,(kind,shape) in ROLES.items():
  name='blk.0.'+role;file,tensor,signature=provider.reader.tensors[name]
  if tensor['type']!=kind or tuple(tensor['shape_ggml_order'])!=shape:raise ValueError('Selected original consumer kind/shape differs '+name)
  result[name]={'type':kind,'shape_ggml_order':list(shape),'path':file['path'],'absolute_offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'original_file_stat':signature}
 return result


def load_frame(row):
 raw=row['raw'];frame=row['layer0']['frame'];metadata=Path(row['layer0']['metadata']);ids=raw['ids']
 if row['prefix'] not in (1,2,4,8) or len(ids)!=row['prefix'] or raw['fresh']!=1:raise ValueError('Exact fresh prefix scope differs')
 if metadata.is_symlink() or sha(metadata)!=row['layer0']['metadata_sha256'] or json.loads(metadata.read_text())!=frame:raise ValueError('Actual metadata SHA/content differs')
 starts=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in raw['stderr'] if line.startswith('SFD request ')]
 markers=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in raw['stderr'] if line.startswith('L0Q8 frame ')]
 if len(starts)!=1 or len(markers)!=1:raise ValueError('Actual SFD/L0 marker roster differs')
 pid=int(starts[0]['pid']);request=int(starts[0]['request']);marker=markers[0]
 if frame['pid']!=pid or frame['request']!=request or int(marker['pid'])!=pid or int(marker['request'])!=request or Path(marker['metadata']).name!=metadata.name:raise ValueError('Actual producer PID/request metadata crossbinding differs')
 if frame['schema']!=1 or frame['stage']!=0 or frame['layer']!=0 or frame['rows']!=1 or frame['gen_ids']!=ids or frame['position']!=len(ids)-1 or frame['token']!=ids[-1] or frame['reused']!=0:raise ValueError('Actual source/token/row boundary differs')
 if frame['request_replay_marker_verified'] is not True or frame['completed_nonce']!=(((pid&0xffffffff)<<32)|request) or frame['graph_key'] not in (2,3):raise ValueError('Actual current GPU graph nonce/key differs')
 if not re.fullmatch('[0-9a-f]{64}',frame['binding_sha256']) or frame['producer_mapping_verified'] is not True or frame['full_model_math_qualified'] is not False or frame['raw_fused_hidden_observed'] is not False or frame['complete_preregistered_layout'] is not True:raise ValueError('Actual33 source/derived/partial-math scope differs')
 contract=Path(__file__).with_name('layer0_numerical_contract_v1.hpp')
 if sha(contract)!=CONTRACT_SHA:raise ValueError('Frozen field source contract changed')
 rows=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',contract.read_text());observed=row['layer0']['observed'];records={field['name']:field for field in observed}
 if len(rows)!=33 or len(frame['fields'])!=33 or len(records)!=33 or len(observed)!=33 or set(records)!={name for name,_,_ in rows}:raise ValueError('Complete33 fields required')
 values={};packets={};bindings={};seen=set();raw_fields={}
 for index,(source,(name,encoding,size)) in enumerate(zip(frame['fields'],rows)):
  provenance='DERIVED_gpu_native_exp_from_actual_gate_up' if index in (25,29) else 'actual_buffer'
  if (source['name'],source['encoding'],source['bytes'],source['observed'],source['provenance'])!=(name,encoding,int(size),True,provenance):raise ValueError('Actual source/DERIVED field contract differs '+name)
  field=records[name];path=Path(field['path']);data=path.read_bytes()
  if path.is_symlink() or str(path) in seen or path.name!=Path(source['file']).name or field['encoding']!=encoding or field['bytes']!=int(size) or len(data)!=int(size) or sha(path)!=field['sha256'] or field['provenance']!=provenance:raise ValueError('Actual field path/SHA/size/provenance differs '+name)
  seen.add(str(path));bindings[name]={key:field[key] for key in ('path','sha256','bytes','encoding','provenance')}
  if encoding.startswith('LE_F32'):
   floats=np.frombuffer(data,dtype='<f4')
   if not np.isfinite(floats).all():raise ValueError('Nonfinite actual/DERIVED F32 '+name)
   if name in NEEDED:values[name]=floats.astype(np.float64)
  elif name in NEEDED:values[name]=np.frombuffer(data,dtype='<i4').astype(np.int64)
  if name in PACKETS:raw_fields[name]=data
 input_raw=np.asarray(values['ffn_mixed'],dtype='<f4').tobytes();input_contract=require_packet_exact(input_raw,raw_fields['ffn_input_q81'],1,2560)
 for name,(count,cols) in PACKETS.items():
  image=decode(raw_fields[name],count,cols)
  if not np.isfinite(image['reconstructed_activation']).all() or not np.isfinite(image['stored_sum']).all():raise ValueError('Nonfinite supplied native packet')
  packets[name]=image['reconstructed_activation']
 derived_contract={}
 for derived,name,count in [('shared_hidden_DERIVED','shared_hq81',1),('expert_hidden_DERIVED','expert_hq81',10)]:
  derived_contract[name]=require_packet_exact(np.asarray(values[derived],dtype='<f4').tobytes(),raw_fields[name],count,640)
 ids=values['router_ids'];mapping=values['expert_entry_map'].reshape(10,3)
 if len(set(ids.tolist()))!=10 or np.any(ids<0) or np.any(ids>=512) or not np.array_equal(mapping,np.column_stack([ids,np.zeros(10,dtype=np.int64),np.arange(10)])):raise ValueError('Actual expert ID/rank/ent_tok/dst binding differs')
 addresses=frame['expert_blob_addresses'];tiers=frame['expert_tiers']
 if len(addresses)!=len(tiers) or len(addresses)!=10 or len(set(addresses))!=10 or any(type(p) is not int or p<=0 or p%4 for p in addresses) or any(type(t) is not int or t not in (1,2) for t in tiers):raise ValueError('Actual selected expert blob/tier roster differs')
 log=metadata.parent.parent/'engine.combined.log';live=live_blob_bindings(log.read_text(),pid,request,addresses,tiers)
 if row['layer0']['source_live_binding']!=live:raise ValueError('Saved live producer allocation binding differs')
 derived_math={name:hidden_metrics(np.asarray(values[gu],dtype='<f4').tobytes(),np.asarray(values[derived],dtype='<f4').tobytes(),count) for name,gu,derived,count in [('shared','shared_gate_up','shared_hidden_DERIVED',1),('experts','expert_gate_up','expert_hidden_DERIVED',10)]}
 return values,packets,{'derived_math_conditional_on_supplied_GU':derived_math,'fields':bindings,'input_packet_contract':input_contract,'derived_HQ_packet_contracts':derived_contract,'raw_hidden_proven':False,'log_sha256':sha(log),'source_live_binding':live}


def bf16_projection(provider,role,x):
 name='blk.0.'+role;shape=provider.shape(name);count=shape[1] if len(shape)>1 else 1;tile=max(1,TILE//(shape[0]*8));output=np.empty(count)
 for start in range(0,count,tile):
  amount=min(tile,count-start);weights=bf16_rne(provider.rows(name,range(start,start+amount)));output[start:start+amount]=weights@np.asarray(x,dtype=np.float64)
 return output


def conditional_ffn(layer,provider,values,packets):
 x=packets['ffn_input_q81'].reshape(-1);ids=values['router_ids'].astype(np.int64);checks={};shared_gu=values['shared_gate_up'].reshape(2,640);expert_gu=values['expert_gate_up'].reshape(10,2,640)
 for index,role in enumerate(('ffn_gate_shexp.weight','ffn_up_shexp.weight')):checks['shared_'+('gate' if index==0 else 'up')+'_original_Q8_0_supplied_packet']=check(shared_gu[index],layer.project('blk.0.'+role,x))
 for rank,expert in enumerate(ids):
  for index,role in enumerate(('ffn_gate_exps.weight','ffn_up_exps.weight')):checks['expert_rank%d_%s_original_Q4_K_supplied_packet'%(rank,'gate' if index==0 else 'up')]=check(expert_gu[rank,index],layer.project('blk.0.'+role,x,expert=int(expert)))
 logits=bf16_projection(provider,'ffn_gate_inp.weight',values['ffn_mixed']);checks['router_original_F32_to_BF16_RNE_weight_supplied_F32_activation']=check(values['router_logits'],logits)
 # Supplied logits/IDs are used here: native exp/tie rounding is not emulated.
 supplied=values['router_logits'];probabilities=np.exp(supplied-supplied.max());probabilities/=probabilities.sum();selected=probabilities[ids];weights=selected/max(float(selected.sum()),6.103515625e-5)
 checks['router_weights_supplied_actual_logits_ids_FP64_softmax']=check(values['router_weights'],weights)
 shared_down=layer.project('blk.0.ffn_down_shexp.weight',packets['shared_hq81'].reshape(-1));shared_gate=float(sigmoid(bf16_projection(provider,'ffn_gate_inp_shexp.weight',values['ffn_mixed']))[0])
 expert_down=np.stack([layer.project('blk.0.ffn_down_exps.weight',packets['expert_hq81'][rank],expert=int(expert)) for rank,expert in enumerate(ids)])
 combined=np.sum(expert_down*values['router_weights'][:,None],axis=0)+shared_down*shared_gate
 checks['aggregate_original_Q5_1_Q8_0_down_supplied_HQ_weights_sharedgate_BF16_seam']=check(values['ffn_block_output'],combined)
 read=layer.hc_read(values['residual_after_attn'].reshape(4,2560),'ffn');checks['FFN_HC_read_original_weights_supplied_after_attn']=check(values['ffn_mixed'],read['mixed'])
 checks['FFN_HC_write_original_injection_supplied_block_after_attn']=check(values['residual_after_ffn'].reshape(4,2560),layer.hc_write(values['residual_after_attn'].reshape(4,2560),values['ffn_block_output'],read['inject']))
 return {'checks':checks,'passed':all(check['passed'] for check in checks.values()),'theoretical_top10_from_supplied_logits':np.lexsort((np.arange(512),-supplied))[:10].tolist(),'actual_rank_ids':ids.tolist(),'top10_native_rounding_qualified':False,'individual_down_outputs_observed':False,'aggregate_cancellation_can_mask_individual_down_errors':True,'shared_scalar_gate_observed':False,'native_exp_or_reduction_emulated':False}


def verify(rows,identity):
 if [row['prefix'] for row in rows]!=[1,2,4,8]:raise ValueError('Exact actual1/2/4/8 roster required')
 frames=[row['layer0']['frame'] for row in rows]
 if len({f['pid'] for f in frames})!=1 or len({f['binding_sha256'] for f in frames})!=1 or len({f['request'] for f in frames})!=4:raise ValueError('Actual source/run request roster differs')
 allpaths=[field['path'] for row in rows for field in row['layer0']['observed']]
 if len(set(allpaths))!=len(allpaths):raise ValueError('Cross-request source field reused')
 # Bind captures before reading any original model payload.
 loaded=[load_frame(row) for row in rows]
 provider=OriginalTensorRows(ROOT/'strata/flash-next/original-gguf-reference-foundation-plan-v1.json',identity);roles=original_roles(provider);layer=FirstGdnLayer(provider,tile_bytes=TILE);results=[]
 for row,(values,packets,binding) in zip(rows,loaded):results.append({'prefix':row['prefix'],'source_bindings':binding,**conditional_ffn(layer,provider,values,packets)})
 return {'passed':all(row['passed'] for row in results),'results':results,'original_weight_bindings':roles,'operator_lane':'original_FP64_conditional_native_Q8_1_code_storage_BF16_RNE_weight_seam','thresholds':{'nmse':NMSE,'max_normalized':MAX_NORMALIZED,'normalizer_floor':1e-6},'affine_contract':'selected Q4_K/Q5_1 affine terms use d_x*integer_code_sum; stored Q8_1 s unused; no other affine format admitted','raw_fused_hidden_observed':False,'per_expert_down_outputs_observed':False,'original_own_state_reference_qualified':False,'complete_layer_math_qualified':False,'full_model_math_qualified':False,'source_or_capture_lifecycle_qualified':False,'scope':'Supplied last-row FFN inputs/GU/HQ/logits/IDs/weights/residuals only; conditional original selected consumers, aggregate down+sharedgate and HC write. No upstream/prefill/ownstate/native-intrinsic/reducer/per-expert down/fullmodel proof'}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--requests',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve evidence; use new output')
 result=verify(json.loads(a.requests.read_text()),a.model_identity);result.update(requests_sha256=sha(a.requests),model_identity_sha256=sha(a.model_identity),checker_sha256=sha(Path(__file__)),dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in DEPENDENCIES},numpy_version=np.__version__)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii');print(json.dumps({'passed':result['passed'],'prefixes':len(result['results']),'conditional_checks':sum(len(row['checks']) for row in result['results']),'full_model_math_qualified':False}))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
