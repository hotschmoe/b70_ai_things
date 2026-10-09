#!/usr/bin/env python3
"""Actual33-field supplied-input Q8_1 checks; DERIVED hidden is never raw proof."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from independent_q8_1_activation_v1 import require_packet_exact,quantization_cost
PAIRS=[('attn_mixed','attn_input_q81',1,2560),('gdn_output_gated','gdn_output_q81',1,6144),('ffn_mixed','ffn_input_q81',1,2560),('shared_hidden_DERIVED','shared_hq81',1,640),('expert_hidden_DERIVED','expert_hq81',10,640)]


def bytes_for(field):
 p=Path(field['path']);raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=field['sha256']:raise ValueError('Captured source bytes changed')
 return raw


def hidden_metrics(gate_up,derived,rows):
 gu=np.frombuffer(gate_up,dtype='<f4').astype(np.float64).reshape(rows,2,640);got=np.frombuffer(derived,dtype='<f4').astype(np.float64).reshape(rows,640)
 g=gu[:,0];up=gu[:,1];reference=(g*np.exp(-np.logaddexp(0,-g)))*up
 if not np.isfinite(reference).all() or not np.isfinite(got).all():raise ValueError('Nonfinite hidden reconstruction')
 err=got-reference;nmse=float(np.sum(err*err)/max(1e-30,np.sum(reference*reference)));maximum=float(np.max(np.abs(err))/max(1e-6,np.max(np.abs(reference))))
 if nmse>1e-6 or maximum>1e-4:raise ValueError('DERIVED SiLU/FP64 supplied GU numerical gate failed')
 return {'nmse':nmse,'max_normalized':maximum,'scope':'DERIVED hardware-intrinsic evaluation from actual GU versus independent FP64; original fused hidden UNOBSERVED','raw_hidden_proven':False}


def verify(rows):
 result=[]
 for row in rows:
  fields={f['name']:f for f in row['layer0']['observed']}
  if len(fields)!=33:raise ValueError('Complete33 layout absent')
  packets=[]
  for source,packet,count,cols in PAIRS:
   x=bytes_for(fields[source]);q=bytes_for(fields[packet]);checked=require_packet_exact(x,q,count,cols)
   derived=source.endswith('_DERIVED')
   packets.append({'source':source,'packet':packet,'packet_exact':checked['packet_exact'],'source_provenance':'DERIVED_gpu_native_exp_from_actual_gate_up' if derived else 'actual_F32_buffer','raw_fused_hidden_proven':False,'intrinsic_rounding_packet_mismatch_policy':'FAIL exact packet; no tolerance relaxation','quantization_cost':quantization_cost(x,q,count,cols)})
  silu={'shared':hidden_metrics(bytes_for(fields['shared_gate_up']),bytes_for(fields['shared_hidden_DERIVED']),1),'experts':hidden_metrics(bytes_for(fields['expert_gate_up']),bytes_for(fields['expert_hidden_DERIVED']),10)}
  result.append({'prefix':row['prefix'],'packets':packets,'silu_reconstruction':silu,'raw_fused_hidden_observed':False,'original_weight_consumers_qualified':False})
 if [r['prefix'] for r in result]!=[1,2,4,8]:raise ValueError('Four actual numerical prefixes required')
 return {'passed':True,'results':result,'source_value_fields':31,'derived_fields':2,'raw_fused_hidden_proven':False,'original_weight_consumers_qualified':False,'full_model_math_qualified':False,'scope':'three actual native input packets plus shared/expert HQ against DERIVED inputs only; not raw hidden or original own-state/weight-consumer proof'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=verify(json.loads(a.requests.read_bytes()));r.update(plan_sha256=a.plan_sha256,requests_sha256=hashlib.sha256(a.requests.read_bytes()).hexdigest());a.output.write_text(json.dumps(r,indent=2)+'\n');print('PASS strict supplied-input and DERIVED HQ packet checks; rawhidden/fullmathfalse')
if __name__=='__main__':main()
