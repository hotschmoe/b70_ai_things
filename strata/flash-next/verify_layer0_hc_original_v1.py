#!/usr/bin/env python3
"""Conditional original-weight HC math; supplied inputs do not prove upstream/model math."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from original_first_gdn_layer_v1 import OriginalTensorRows,FirstGdnLayer
from original_math_scalar import metrics
ROOT=Path(__file__).resolve().parents[2]
NMSE=1e-6
MAX_NORMALIZED=1e-4

def check(actual,expected):
 result=metrics(np.asarray(actual).reshape(-1),np.asarray(expected).reshape(-1))
 result['passed']=result['nmse']<=NMSE and result['max_normalized']<=MAX_NORMALIZED
 return result

def verify(requests,identity):
 provider=OriginalTensorRows(ROOT/'strata/flash-next/original-gguf-reference-foundation-plan-v1.json',identity)
 layer=FirstGdnLayer(provider);results=[]
 for row in requests:
  fields={item['name']:item for item in row['layer0']['observed']}
  def load(name):
   field=fields[name];path=Path(field['path']);raw=path.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=field['sha256'] or len(raw)!=field['bytes']:raise ValueError('Captured source field changed: '+name)
   values=np.frombuffer(raw,dtype='<f4').astype(np.float64)
   if not np.isfinite(values).all():raise ValueError('Nonfinite captured input: '+name)
   return values
  residual=load('residual_input').reshape(4,2560)
  attn=layer.hc_read(residual,'attn');checks={}
  for name,key in [('attn_hc_normalized','normalized'),('attn_hc_low','low_silu'),('attn_hc_gate','gate'),('attn_hc_inject','inject'),('attn_mixed','mixed')]:checks[name]=check(load(name),attn[key])
  after_attn=load('residual_after_attn').reshape(4,2560)
  checks['attn_hc_write']=check(after_attn,layer.hc_write(residual,load('gdn_block_output'),load('attn_hc_inject')))
  ffn=layer.hc_read(after_attn,'ffn');checks['ffn_mixed']=check(load('ffn_mixed'),ffn['mixed'])
  checks['ffn_hc_read_write_conditional']=check(load('residual_after_ffn').reshape(4,2560),layer.hc_write(after_attn,load('ffn_block_output'),ffn['inject']))
  results.append({'prefix':row['prefix'],'checks':checks,'passed':all(c['passed'] for c in checks.values())})
 if [r['prefix'] for r in results]!=[1,2,4,8]:raise ValueError('Exact four-prefix HC roster required')
 return {'passed':all(r['passed'] for r in results),'scope':'Conditional original GGUF HC weights + supplied actual input/block output; FFN read-write has independently computed injection. No upstream/GDN/MoE/model/lifecycle proof','operator_lane':'original_fp64','thresholds':{'nmse':NMSE,'max_normalized':MAX_NORMALIZED},'results':results,'full_model_math_qualified':False,'source_or_capture_lifecycle_qualified':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=verify(json.loads(a.requests.read_text()),a.model_identity)
 result['requests_sha256']=hashlib.sha256(a.requests.read_bytes()).hexdigest();result['model_identity_sha256']=hashlib.sha256(a.model_identity.read_bytes()).hexdigest();result['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists():raise ValueError('Preserve existing evidence; use a new output')
 a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
 print(json.dumps({'passed':result['passed'],'prefixes':len(result['results']),'operator_checks':sum(len(r['checks']) for r in result['results']),'full_model_math_qualified':False}))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
