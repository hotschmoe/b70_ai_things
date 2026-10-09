#!/usr/bin/env python3
"""Independent native packet contract on actual26-field frames; full math false."""
import argparse,hashlib,json
from pathlib import Path
from independent_q8_1_activation_v1 import require_packet_exact,quantization_cost
PAIRS=[('attn_mixed','attn_input_q81',2560),('gdn_output_gated','gdn_output_q81',6144),('ffn_mixed','ffn_input_q81',2560)]

def verify(rows):
 results=[]
 for row in rows:
  fields={field['name']:field for field in row['layer0']['observed']};packet_results=[]
  for source,packet,cols in PAIRS:
   x=Path(fields[source]['path']);q=Path(fields[packet]['path']);raw=x.read_bytes();packed=q.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=fields[source]['sha256'] or hashlib.sha256(packed).hexdigest()!=fields[packet]['sha256']:raise ValueError('Actual raw field source changed after capture binding')
   packet_results.append({'source':source,'packet':packet,'contract':require_packet_exact(raw,packed,1,cols),'activation_quantization_cost':quantization_cost(raw,packed,1,cols)})
  results.append({'prefix':row['prefix'],'packets':packet_results,'raw_fused_hidden_observed':False,'complete_layer_operators_qualified':False})
 if [r['prefix'] for r in results]!=[1,2,4,8]:raise ValueError('Actual four-prefix packet roster missing')
 return {'passed':True,'results':results,'full_model_math_qualified':False,'scope':'three supplied actual-F32-to-native-Q8_1 inputs only; not original own-state FP64 or accumulated/full-layer math'}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--requests',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=verify(json.loads(a.requests.read_text()));result['plan_sha256']=a.plan_sha256;result['requests_sha256']=hashlib.sha256(a.requests.read_bytes()).hexdigest();a.output.write_text(json.dumps(result,indent=2)+'\n');print('PASS independent three native packets at four actual prefixes; fullmathfalse')
if __name__=='__main__':main()
