#!/usr/bin/env python3
"""Readonly owned-prefix1 L0 input/packet hash localization; no original weights.
Packet equality only, no native reduction/intrinsic/full-model qualification.
"""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from verify_layer0_ffn_original_v1 import load_frame

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)

def compare_events(events,fields,expert_ids):
 events=[e for e in events if e['role'].startswith('blk.0.')];require(len(events)==36,'Complete owned L0 packet event roster differs');require(len(expert_ids)==10 and len(set(expert_ids))==10,'Native expert rank roster differs')
 for event in events:require(event.get('stored_s_used') is False and all(re.fullmatch('[0-9a-f]{64}',event[key]) for key in ('input_sha256','packet_sha256')),'Owned packet input/format hash contract differs')
 expected=[]
 for role,packet,inputfield in [('attn_qkv','attn_input_q81','attn_mixed'),('attn_gate','attn_input_q81','attn_mixed'),('ssm_out','gdn_output_q81','gdn_output_gated'),('ffn_gate_shexp','ffn_input_q81','ffn_mixed'),('ffn_up_shexp','ffn_input_q81','ffn_mixed'),('ffn_down_shexp','shared_hq81','shared_hidden_DERIVED')]:expected.append((role,None,None,packet,inputfield))
 own_ids=[e['expert'] for e in events if e['role']=='blk.0.ffn_down_exps.weight'];require(own_ids==expert_ids,'Owned/native selected expert ID/rank ordering differs')
 for rank,expert in enumerate(expert_ids):
  for role in ('ffn_gate_exps','ffn_up_exps'):expected.append((role,expert,None,'ffn_input_q81','ffn_mixed'))
  expected.append(('ffn_down_exps',expert,rank,'expert_hq81','expert_hidden_DERIVED'))
 result=[]
 for role,expert,rank,packet,inputfield in expected:
  matches=[e for e in events if e['role']=='blk.0.'+role+'.weight' and e['expert']==expert];require(len(matches)==1,'Owned consumer event absent/duplicated');event=matches[0];native_packet=fields[packet];native_input=fields[inputfield]
  if rank is not None:
   require(len(native_packet)==7200 and len(native_input)==25600,'Native ten-row HQ/derived input extent differs');native_packet=native_packet[rank*720:(rank+1)*720];native_input=native_input[rank*2560:(rank+1)*2560]
  input_sha=hashlib.sha256(native_input).hexdigest();packet_sha=hashlib.sha256(native_packet).hexdigest();derived='DERIVED' in inputfield
  result.append({'role':event['role'],'expert':expert,'native_rank':rank,'native_packet_field':packet,'native_input_field':inputfield,'native_input_provenance':'DERIVED_native_exp_from_actual_GU; raw_fused_hidden_NOTobserved' if derived else 'actual_buffer','own_input_sha256':event['input_sha256'],'native_input_sha256':input_sha,'own_packet_sha256':event['packet_sha256'],'native_packet_sha256':packet_sha,'input_F32_bytes_equal':event['input_sha256']==input_sha,'packet_bytes_equal':event['packet_sha256']==packet_sha,'packet_byte_diff_positions_available':False})
 require(len(result)==36,'Complete36 consumer mapping absent');return result


def audit(own_report,native_requests):
 own=json.loads(Path(own_report).read_text());rows=json.loads(Path(native_requests).read_text());require([r['prefix'] for r in rows]==[1,2,4,8],'Actual native prefix roster differs');row=rows[0];binding=own['native_observation_binding']
 require(not own['errors'] and own['status']=='exploratory complete; NO numerical qualification' and own['numeric_pass_claim'] is False and own['full_model_math_qualified'] is False and own['post_original_identity']['complete_four_publisher_hashes_verified'] is True,'Final owned exploration/source receipt absent')
 require(binding['prefix']==1 and binding['requests_sha256']==sha(native_requests) and binding['nativePID']==row['layer0']['frame']['pid'] and binding['request']==row['layer0']['frame']['request'],'Owned/native source/request identity differs')
 _,_,native_binding=load_frame(row) # Native capture/packet proof only; no original reader.
 names={'attn_input_q81','attn_mixed','gdn_output_q81','gdn_output_gated','ffn_input_q81','ffn_mixed','shared_hq81','shared_hidden_DERIVED','expert_hq81','expert_hidden_DERIVED','router_ids'};fields={};source={}
 for field in row['layer0']['observed']:
  if field['name'] not in names:continue
  p=Path(field['path']);require(not p.is_symlink() and sha(p)==field['sha256'],'Actual native packet/input field hash differs');fields[field['name']]=p.read_bytes();source[field['name']]=field
 require(set(fields)==names,'Complete native L0 packet/input source scope absent');ids=np.frombuffer(fields['router_ids'],dtype='<i4').astype(int).tolist();result=compare_events(own['packet_events_owned'],fields,ids)
 return {'schema':1,'scope':'Owned original prefix1 L0 only; actual native33 field admission, exact36consumer packetSHA mapping','own_report':str(Path(own_report).resolve()),'own_report_sha256':sha(own_report),'native_requests_sha256':sha(native_requests),'native_binding':native_binding,'field_bindings':source,'results':result,'all_packet_bytes_equal':all(r['packet_bytes_equal'] for r in result),'all_raw_F32_input_bytes_equal':all(r['input_F32_bytes_equal'] for r in result),'input_difference_changed_any_observed_packet':any(not r['packet_bytes_equal'] for r in result),'actual_native_FMA_exp_or_reductions_qualified':False,'raw_hidden_proven':False,'layer2_scope_observed':False,'full_model_math_qualified':False,'numerical_tolerance_or_pass_assigned':False,'reference_ablation':own.get('reference_ablation')}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--own-report',type=Path,required=True);p.add_argument('--native-requests',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Preserve evidence; new output required');result=audit(a.own_report,a.native_requests);result['auditor_sha256']=sha(Path(__file__));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii');print(json.dumps({'consumer_mappings':len(result['results']),'all_packet_bytes_equal':result['all_packet_bytes_equal'],'all_raw_F32_input_bytes_equal':result['all_raw_F32_input_bytes_equal'],'full_model_math_qualified':False}))
if __name__=='__main__':main()
