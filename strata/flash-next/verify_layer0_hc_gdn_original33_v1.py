#!/usr/bin/env python3
"""Conditional original HC/GDN consumers for genuine31source+2DERIVED captures.
No upstream, independently reconstructed incoming state, complete layer/model,
or source/capture lifecycle qualification is inferred from these equations.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import verify_layer0_ffn_original_v1 as frame33
import verify_layer0_gdn_original_v1 as gdn
import verify_layer0_hc_original_v1 as hc
from original_first_gdn_layer_v1 import OriginalTensorRows,FirstGdnLayer
from independent_q8_1_activation_v1 import require_packet_exact,decode
ROOT=Path(__file__).resolve().parents[2]
NMSE=gdn.NMSE
MAX_NORMALIZED=gdn.MAX_NORMALIZED
HC_F32={'residual_input':10240,'attn_hc_normalized':10240,'attn_hc_low':320,'attn_hc_gate':10240,'attn_hc_inject':4,'residual_after_attn':10240,'ffn_mixed':2560,'ffn_block_output':2560,'residual_after_ffn':10240}
HC_ROLES={f'blk.0.hc_{half}_{role}.weight':(kind,shape) for half in ('attn','ffn') for role,kind,shape in [('norm','F32',(10240,)),('down','Q8_0',(10240,320)),('up','Q8_0',(320,10240)),('inject','F32',(10240,4))]}
DEPENDENCY_SHA={'audit_layer0_capture_lifecycle_v2.py': 'cd5441854c7bf11250080c7b109564e91f204422c4b790623132999641a7abfe', 'independent_q8_1_activation_v1.py': 'b58d73282051bfa301ebb6f781d5d59c63128d9074a42d53cf4d9ff05d75a714', 'layer0_numerical_contract_v1.hpp': '86a83df856829674f03b9eb6c80bac0817cb2cdfa4459b3186f5480600e8dfdd', 'native_storage_first_gdn_estimate_v1.py': '43a2beeafabcd5a8ed040139a5122b7b18d0518b94b3700d4f6601d685e93d2a', 'original_first_gdn_layer_v1.py': 'c6352dc69f22e5940e0c6799fe83137225b8126fe7ce771983de682e00abd67f', 'original_gguf_reference.py': '6498bfd01ad06b7439a6dc07c753306bbf1d0b173f8ba1ab9763da7d11a3ddb5', 'original_gguf_vector_decoder_v2.py': '130f1a525e366e5d33fcb67a3c765ee50d5c4eaedb2b4a0495ae2386cd18c7b9', 'original_math_scalar.py': 'a6e9efac1f298bc9c1867a83b26d76e747ee35bbae3b88858817dcb44adec1c7', 'parse_usm_logical_free_trace.py': '71714821736fe98790b80f85d1ab72ac59f58ee6336298721f0aaee2753ca665', 'verify_layer0_ffn_original_v1.py': '3f433f105c260e5760d54942c6c9e8f46f6ff97793238e310754ffd55ebf8d70', 'verify_layer0_gdn_original_v1.py': 'c96ef510fd23b50956e22675ac36c3050c4f4fe6a4362c444f67819311c3b9fa', 'verify_layer0_hc_original_v1.py': '8f11969cd1f43bc29b8369446a975d77a00ad5fc9a08ba0182938d32617eb059', 'verify_layer0_packets_v3.py': '620f18e8aed625bb3c40f451d94ed66a8b1aa6aa247ec7638423426f38e9b329'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def require(condition,message):
 if not condition:raise ValueError(message)


def source_pins():
 for name,digest in DEPENDENCY_SHA.items():require(sha(Path(__file__).with_name(name))==digest,'Frozen scalar/schema dependency changed: '+name)
 require(NMSE==hc.NMSE==1e-6 and MAX_NORMALIZED==hc.MAX_NORMALIZED==1e-4,'Unchanged original tolerances required')


def run_admission(root):
 """Receipt/source metadata only; source payload admission is separately enforced."""
 root=Path(root).resolve();parent=read(root/'parent-qualification.json');report=read(root/'child/report.json')
 require(parent.get('passed') is True and parent.get('child_return_code')==0 and parent.get('owned_containers_terminal') is True and parent.get('post_health_passed') is True and parent.get('kernel_fault_gate_passed') is True and parent.get('errors')==[] and parent.get('interrupted') is False and parent.get('forced_cleanup') is False,'Actual completed numerical parent required')
 require(report.get('passed') is True and report.get('numerical_and_teardown_passed') is True and report.get('post_health_passed') is True and report.get('layer0_logical_lifecycle_qualified') is True and report.get('packet_contract_qualified') is True and report.get('full_model_math_qualified') is False,'Actual33 numerical/packet/lifecycle report required')
 plan_path=Path(parent['plan']);plan=read(plan_path);digest=sha(plan_path)
 require(parent['plan_sha256']==report['plan_sha256']==digest and plan['observed_fields']==33 and plan['source_value_fields']==31 and plan['derived_value_fields']==2 and plan['layout_fields']==33,'Exact original33 plan association required')
 engine=Path(plan['engine_root']);require(sha(engine/'receipt.json')==plan['engine_receipt_sha256']==report['engine_receipt_sha256'],'Actual selected native engine association differs')
 identity_path=root/'post-model-identity.json';identity=read(identity_path)
 require(report['post_model_identity']['path']==str(identity_path) and report['post_model_identity']['sha256']==sha(identity_path),'Original postidentity association differs')
 require(identity.get('passed') is True and len(identity['rows'])==4,'Complete original postidentity required')
 for name in ('started','finished','after_child_terminal_epoch'):
  require(type(identity[name]) in (int,float) and math.isfinite(identity[name]) and identity[name]>0,'Invalid source chronology')
 require(identity['started']>=identity['after_child_terminal_epoch']>=parent['child_terminal_epoch'] and identity['finished']>=identity['started'] and parent['finished_epoch']>=identity['finished'],'Original postidentity must follow child terminal')
 requests_path=root/'child/candidatecombined_on/requests.json';requests=read(requests_path)
 require([r['prefix'] for r in requests]==[1,2,4,8] and all(r['layer0']['frame']['binding_sha256']==digest for r in requests),'Supplied capture must bind actual candidate33 plan')
 return requests,identity_path,{'parent_path':str(root/'parent-qualification.json'),'parent_sha256':sha(root/'parent-qualification.json'),'report_sha256':sha(root/'child/report.json'),'plan_sha256':digest,'engine_receipt_sha256':sha(engine/'receipt.json'),'requests_sha256':sha(requests_path),'post_identity_sha256':sha(identity_path)}


def load33(row):
 # The genuine33 parser validates ALL fields/live allocations/DERIVED seams.
 # No modified frame or26-field projection is passed into old admission code.
 _,_,binding=frame33.load_frame(row)
 fields={f['name']:f for f in row['layer0']['observed']};values={};packets={};contracts={}
 require(len(fields)==33 and sum(f['provenance']=='actual_buffer' for f in fields.values())==31 and sum(f['provenance']=='DERIVED_gpu_native_exp_from_actual_gate_up' for f in fields.values())==2,'Exact31source2DERIVED roster required')
 raws={}
 for name,size in {**gdn.F32,**HC_F32}.items():
  field=fields[name];path=Path(field['path']);raw=path.read_bytes()
  require(not path.is_symlink() and sha(path)==field['sha256'] and field['bytes']==size*4 and len(raw)==size*4 and field['provenance']=='actual_buffer' and field['encoding'].startswith('LE_F32'),'Exact original input bytes changed: '+name)
  values[name]=np.frombuffer(raw,dtype='<f4').astype(np.float64);raws[name]=raw
  require(np.isfinite(values[name]).all(),'Nonfinite supplied source: '+name)
 for name,(source,cols) in gdn.PACKETS.items():
  field=fields[name];raw=Path(field['path']).read_bytes()
  require(sha(field['path'])==field['sha256'] and field['provenance']=='actual_buffer' and field['encoding'].startswith('Q8_1_LE') and len(raw)==cols//32*36 and field['bytes']==len(raw),'Exact original native packet bytes changed: '+name)
  contracts[name]=require_packet_exact(raws[source],raw,1,cols)
  packets[name]=decode(raw,1,cols)['reconstructed_activation'].reshape(-1)
 return values,packets,{'complete33_source_bindings':binding,'gdn_native_Q8_1_contracts':contracts}


def validate_roster(requests):
 require([r['prefix'] for r in requests]==[1,2,4,8],'Exact four-prefix roster required')
 frames=[r['layer0']['frame'] for r in requests]
 require(len({f['pid'] for f in frames})==1 and len({f['request'] for f in frames})==4 and len({f['binding_sha256'] for f in frames})==1,'Actual run/request source association differs')
 paths=[f['path'] for r in requests for f in r['layer0']['observed']]
 require(len(set(paths))==len(paths),'Cross-request source field reused')
 return [load33(r) for r in requests]


def role_bindings(provider):
 result=gdn.role_binding(provider)
 for name,(kind,shape) in HC_ROLES.items():
  file,tensor,sig=provider.reader.tensors[name]
  require(tensor['type']==kind and tuple(tensor['shape_ggml_order'])==shape,'Exact selected original HC type/shape required: '+name)
  result[name]={'type':kind,'shape_ggml_order':list(shape),'path':file['path'],'absolute_offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'original_file_stat':sig}
 return result


def conditional_hc(layer,v):
 residual=v['residual_input'].reshape(4,2560);attn=layer.hc_read(residual,'attn');checks={}
 for name,key in [('attn_hc_normalized','normalized'),('attn_hc_low','low_silu'),('attn_hc_gate','gate'),('attn_hc_inject','inject'),('attn_mixed','mixed')]:checks[name]=hc.check(v[name],attn[key])
 after=v['residual_after_attn'].reshape(4,2560)
 checks['attn_hc_write']=hc.check(after,layer.hc_write(residual,v['gdn_block_output'],v['attn_hc_inject']))
 ffn=layer.hc_read(after,'ffn');checks['ffn_mixed']=hc.check(v['ffn_mixed'],ffn['mixed'])
 checks['ffn_hc_read_write_conditional']=hc.check(v['residual_after_ffn'].reshape(4,2560),layer.hc_write(after,v['ffn_block_output'],ffn['inject']))
 return checks


def verify(root):
 source_pins();requests,identity,binding=run_admission(root);loaded=validate_roster(requests)
 provider=OriginalTensorRows(ROOT/'strata/flash-next/original-gguf-reference-foundation-plan-v1.json',identity);roles=role_bindings(provider);layer=FirstGdnLayer(provider,tile_bytes=gdn.TILE)
 gamma=layer.vector('blk.0.ssm_norm.weight');conv=provider.rows('blk.0.ssm_conv1d.weight',range(10240));results=[]
 for row,(values,packets,fields) in zip(requests,loaded):
  checks={'hc':conditional_hc(layer,values),'gdn':{**gdn.conditional_convolution(values,conv),**gdn.conditional_gdn(values,gamma),**gdn.conditional_projections(layer,values,packets)}}
  results.append({'prefix':row['prefix'],'checks':checks,'field_bindings':fields,'passed':all(c['passed'] for group in checks.values() for c in group.values()),'incoming_state_supplied':True,'normalized_qkv_decay_beta_z_supplied':True})
 return {'schema':1,'passed':all(r['passed'] for r in results),'results':results,'capture_schema':'31actualsource+2DERIVED; all33 admitted','source_value_fields':31,'derived_fields':2,'observed_layout_fields':33,'run_bindings':binding,'original_weight_bindings':roles,'operator_lane':'frozen original_FP64_conditional selected_Q8_0/F32_HC_GDN_consumers','thresholds':{'nmse':NMSE,'max_normalized':MAX_NORMALIZED,'normalizer_floor':1e-6},'full_model_math_qualified':False,'complete_layer_math_qualified':False,'original_own_state_reference_qualified':False,'source_or_capture_lifecycle_qualified':False,'raw_fused_hidden_observed':False,'projection_affine_generalization':False,'Q8_1_stored_sum_used_for_Q8_0':False,'scope':'Conditional supplied activation/incoming recurrent state/normalized qkv/decay-beta and block outputs only. Original selected HC Q8_0/F32, GDN original Q8_0 projections with verified supplied Q8_1 d*codes, F32 convolution/norm. No upstream/own-state/alpha-beta/prefill/FFN/fullmodel inference.'}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 require(not a.output.exists(),'Preserve evidence; new output required')
 result=verify(a.run_root);result['checker_sha256']=sha(Path(__file__));result['dependency_sha256']=DEPENDENCY_SHA
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
 print(json.dumps({'passed':result['passed'],'prefixes':len(result['results']),'operator_checks':sum(len(g) for r in result['results'] for g in r['checks'].values()),'full_model_math_qualified':False}))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
