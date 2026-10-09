#!/usr/bin/env python3
"""Conditional supplied-input/state GDN and original-Q8_0 packet projections.
Never an independent own-state/model oracle, graph lifecycle or affine-format gate.
"""
import argparse,hashlib,json,math,re
from pathlib import Path
import numpy as np
from original_first_gdn_layer_v1 import OriginalTensorRows,FirstGdnLayer
from original_math_scalar import gdn_step,sigmoid,metrics
from independent_q8_1_activation_v1 import require_packet_exact,decode
ROOT=Path(__file__).resolve().parents[2]
NMSE=1e-6;MAX_NORMALIZED=1e-4;EPS=float(np.float32(1e-6));TILE=64<<20
F32={'attn_mixed':2560,'gdn_qkv':10240,'gdn_z':6144,'gdn_normalized_qkv':10240,'gdn_decay_beta':96,'gdn_state_before':128*48*128,'gdn_state_after':128*48*128,'gdn_conv_before':10240*3,'gdn_conv_after':10240*3,'gdn_output_gated':6144,'gdn_block_output':2560}
PACKETS={'attn_input_q81':('attn_mixed',2560),'gdn_output_q81':('gdn_output_gated',6144)}
ROLES={'blk.0.attn_qkv.weight':('Q8_0',(2560,10240)),'blk.0.attn_gate.weight':('Q8_0',(2560,6144)),'blk.0.ssm_out.weight':('Q8_0',(6144,2560)),'blk.0.ssm_norm.weight':('F32',(128,)),'blk.0.ssm_conv1d.weight':('F32',(4,10240))}
CONTRACT_SHA='86a83df856829674f03b9eb6c80bac0817cb2cdfa4459b3186f5480600e8dfdd'
DEPENDENCIES=['original_first_gdn_layer_v1.py','original_math_scalar.py','original_gguf_reference.py','original_gguf_vector_decoder_v2.py','independent_q8_1_activation_v1.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check(actual,expected):
 result=metrics(np.asarray(actual,dtype=np.float64).reshape(-1),np.asarray(expected,dtype=np.float64).reshape(-1));result['passed']=result['nmse']<=NMSE and result['max_normalized']<=MAX_NORMALIZED;return result


def role_binding(provider):
 result={}
 for name,(kind,shape) in ROLES.items():
  file,tensor,sig=provider.reader.tensors[name]
  if tensor['type']!=kind or tuple(tensor['shape_ggml_order'])!=shape:raise ValueError('Only exact original Q8_0/F32 role/shape admitted: '+name)
  result[name]={'type':kind,'shape_ggml_order':list(shape),'path':file['path'],'absolute_offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'original_file_stat':sig}
 return result


def load_fields(row):
 if row['prefix'] not in [1,2,4,8] or len(row['raw']['ids'])!=row['prefix'] or row['raw']['fresh']!=1:raise ValueError('Actual fresh raw prefix scope differs')
 frame=row['layer0']['frame'];ids=row['raw']['ids']
 metadata=Path(row['layer0']['metadata'])
 if metadata.is_symlink() or sha(metadata)!=row['layer0']['metadata_sha256'] or json.loads(metadata.read_text())!=frame:raise ValueError('Metadata SHA/content differs')
 starts=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in row['raw']['stderr'] if line.startswith('SFD request ')]
 markers=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in row['raw']['stderr'] if line.startswith('L0Q8 frame ')]
 if len(starts)!=1 or len(markers)!=1:raise ValueError('Actual SFD/L0 frame roster ambiguous')
 pid=int(starts[0]['pid']);ordinal=int(starts[0]['request']);marker=markers[0]
 if frame['pid']!=pid or frame['request']!=ordinal or int(marker['pid'])!=pid or int(marker['request'])!=ordinal or Path(marker['metadata']).name!=metadata.name:raise ValueError('Actual frame PID/request/path crossbinding differs')
 if frame['schema']!=1 or frame['rows']!=1 or frame['graph_key'] not in (2,3) or frame['completed_nonce']!=(((pid&0xffffffff)<<32)|ordinal):raise ValueError('Actual graph nonce/layout differs')
 if not re.fullmatch('[0-9a-f]{64}',frame['binding_sha256']) or row['layer0']['full_model_math_qualified'] is not False:raise ValueError('Partial frame binding/scope differs')
 contract=Path(__file__).with_name('layer0_numerical_contract_v1.hpp')
 if sha(contract)!=CONTRACT_SHA:raise ValueError('Frozen source field contract changed')
 wanted=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',contract.read_text())
 if len(wanted)!=33 or len(frame['fields'])!=33:raise ValueError('Actual field quota differs')
 for index,(actual,(name,encoding,size)) in enumerate(zip(frame['fields'],wanted)):
  observed=index<24 or index>=31
  if (actual['name'],actual['encoding'],actual['bytes'],actual['observed'])!=(name,encoding,int(size),observed):raise ValueError('Actual field source contract differs')
  if actual['provenance']!=('actual_buffer' if observed else 'UNOBSERVED_no_producer_hook'):raise ValueError('Actual field provenance differs')

 if frame['gen_ids']!=ids or frame['position']!=len(ids)-1 or frame['token']!=ids[-1] or frame['reused']!=0 or frame['layer']!=0 or frame['stage']!=0 or not frame['request_replay_marker_verified']:raise ValueError('Actual source frame/token/state boundary binding differs')
 if frame['full_model_math_qualified'] is not False or frame['raw_fused_hidden_observed'] is not False or frame['complete_preregistered_layout'] is not False:raise ValueError('Partial native frame cannot claim full model math')
 records={item['name']:item for item in row['layer0']['observed']}
 if len(records)!=26 or set(records)!={f['name'] for f in frame['fields'] if f['observed']}:raise ValueError('Actual 26 observed field roster differs')
 if len(records)!=len(row['layer0']['observed']):raise ValueError('Duplicate actual source field')
 values={};raw_fields={};bindings={}
 for name in list(F32)+list(PACKETS):
  field=records[name];path=Path(field['path']);source=next(f for f in frame['fields'] if f['name']==name)
  if path.name!=Path(source['file']).name or field['encoding']!=source['encoding']:raise ValueError('Observed field/frame file binding differs')
  raw=path.read_bytes();expected=F32[name]*4 if name in F32 else PACKETS[name][1]//32*36
  if field['bytes']!=expected or len(raw)!=expected or sha(path)!=field['sha256'] or path.is_symlink():raise ValueError('Actual field SHA/size/provenance changed: '+name)
  raw_fields[name]=raw;bindings[name]={'path':str(path),'sha256':field['sha256'],'bytes':expected,'encoding':field['encoding']}
  if name in F32:
   if not field['encoding'].startswith('LE_F32'):raise ValueError('Actual F32 encoding differs')
   values[name]=np.frombuffer(raw,dtype='<f4').astype(np.float64)
   if not np.isfinite(values[name]).all():raise ValueError('Actual supplied F32 field nonfinite: '+name)
  elif not field['encoding'].startswith('Q8_1_LE'):raise ValueError('Actual supplied packet encoding differs')
 packets={};contracts={}
 for name,(source,cols) in PACKETS.items():
  contracts[name]=require_packet_exact(raw_fields[source],raw_fields[name],1,cols)
  packets[name]=decode(raw_fields[name],1,cols)['reconstructed_activation'].reshape(-1)
  if not np.isfinite(packets[name]).all():raise ValueError('Actual Q8_1 d*codes reconstruction nonfinite')
 return values,packets,bindings,contracts


def conditional_gdn(values,gamma):
 gamma=np.asarray(gamma,dtype=np.float64)
 if gamma.shape!=(128,) or not np.isfinite(gamma).all():raise ValueError('Original F32 ssm_norm gamma differs')
 normalized=values['gdn_normalized_qkv'];q=normalized[:2048].reshape(16,128);k=normalized[2048:4096].reshape(16,128);v=normalized[4096:].reshape(48,128)
 physical=values['gdn_state_before'].reshape(128,48,128);state=physical.transpose(1,0,2);decay_beta=values['gdn_decay_beta'].reshape(2,48)
 next_state,unscaled=gdn_step(state,q,k,v,decay_beta[0],decay_beta[1],lane='original_fp64');core=unscaled/math.sqrt(128)
 normalized_core=core*gamma[None,:]/np.sqrt(np.mean(core*core,axis=1,keepdims=True)+EPS);gated=normalized_core*sigmoid(values['gdn_z'].reshape(48,128))
 expected_physical=next_state.transpose(1,0,2).reshape(-1)
 return {'gdn_state_after_supplied_normalized_qkv_decay_beta_state':check(values['gdn_state_after'],expected_physical),'gdn_output_gated_supplied_state_qkv_beta_z_original_gamma':check(values['gdn_output_gated'],gated.reshape(-1))}


def conditional_convolution(values,weights):
 weights=np.asarray(weights,dtype=np.float64)
 if weights.shape!=(10240,4) or not np.isfinite(weights).all():raise ValueError('Original convolution channel/tap layout differs')
 history=values['gdn_conv_before'].reshape(10240,3)
 full=np.concatenate([history,values['gdn_qkv'][:,None]],axis=1)
 conv=np.sum(full*weights,axis=1);activated=conv*sigmoid(conv)
 normalized=activated.copy()
 for start in (0,2048):
  heads=activated[start:start+2048].reshape(16,128)
  normalized[start:start+2048]=(heads/np.sqrt(np.sum(heads*heads,axis=1,keepdims=True)+EPS)).reshape(-1)
 return {'conv_history_shift_supplied_history_qkv':check(values['gdn_conv_after'],full[:,1:].reshape(-1)),'conv_silu_qk_sum_l2_supplied_history_qkv_original_F32_taps':check(values['gdn_normalized_qkv'],normalized)}


def conditional_projections(layer,values,packets):
 # Specific original Q8_0 rows are symmetric d_w*codes_w. Activation Q8_1
 # stored-sum s is UNUSED here. Do not apply this to Q5_1/Q4_K affine weights.
 return {'attn_qkv_verified_packet_original_Q8_0':check(values['gdn_qkv'],layer.project('blk.0.attn_qkv.weight',packets['attn_input_q81'])),'attn_gate_verified_packet_original_Q8_0':check(values['gdn_z'],layer.project('blk.0.attn_gate.weight',packets['attn_input_q81'])),'ssm_out_verified_packet_original_Q8_0':check(values['gdn_block_output'],layer.project('blk.0.ssm_out.weight',packets['gdn_output_q81']))}


def validate_roster(requests):
 if [row['prefix'] for row in requests]!=[1,2,4,8]:raise ValueError('Exact actual prefix1/2/4/8 roster required')
 frames=[row['layer0']['frame'] for row in requests]
 if len({f['binding_sha256'] for f in frames})!=1 or len({f['pid'] for f in frames})!=1 or len({f['request'] for f in frames})!=4:raise ValueError('Actual capture run/request roster differs')
 paths=[f['path'] for row in requests for f in row['layer0']['observed']]
 if len(set(paths))!=len(paths):raise ValueError('Actual cross-request capture file reused')
 return [load_fields(row) for row in requests]


def verify(requests,identity):
 loaded=validate_roster(requests)
 provider=OriginalTensorRows(ROOT/'strata/flash-next/original-gguf-reference-foundation-plan-v1.json',identity);weights=role_binding(provider);layer=FirstGdnLayer(provider,tile_bytes=TILE);gamma=layer.vector('blk.0.ssm_norm.weight');conv_weights=provider.rows('blk.0.ssm_conv1d.weight',range(10240));results=[]
 for row,(values,packets,bindings,contracts) in zip(requests,loaded):
  checks={**conditional_convolution(values,conv_weights),**conditional_gdn(values,gamma),**conditional_projections(layer,values,packets)}
  results.append({'prefix':row['prefix'],'field_bindings':bindings,'packet_contracts':contracts,'checks':checks,'passed':all(result['passed'] for result in checks.values()),'state_mapping':'captured physical[128,48,128] ->transpose(1,0,2) logical[48,128,128]','incoming_state_supplied':True,'normalized_qkv_logdecay_beta_z_supplied':True})
 return {'passed':all(row['passed'] for row in results),'scope':'Conditional native supplied normalized qkv/logdecay-beta/incoming state/z plus original F32 gamma; verified actual native Q8_1 d*codes and original symmetric Q8_0 projections only. No original own-state/alpha-beta/upstream/MoE/all-layer math or source/capture lifecycle qualification','operator_lane':'original_fp64_conditional','thresholds':{'nmse':NMSE,'max_normalized':MAX_NORMALIZED,'normalizer_floor':1e-6},'rms_epsilon_original_F32':EPS,'core_query_scale':'1/sqrt128 before per-head RMS gamma and sigmoid z','original_weight_bindings':weights,'results':results,'projection_affine_generalization':False,'Q8_1_stored_sum_used_for_Q8_0':False,'full_model_math_qualified':False,'original_own_state_reference_qualified':False,'source_or_capture_lifecycle_qualified':False}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--requests',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve existing evidence; choose a new output')
 result=verify(json.loads(a.requests.read_text()),a.model_identity);result.update(requests_sha256=sha(a.requests),model_identity_sha256=sha(a.model_identity),checker_sha256=sha(Path(__file__)),dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in DEPENDENCIES},numpy_version=np.__version__)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii');print(json.dumps({'passed':result['passed'],'prefixes':len(result['results']),'operator_checks':sum(len(row['checks']) for row in result['results']),'full_model_math_qualified':False}))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
