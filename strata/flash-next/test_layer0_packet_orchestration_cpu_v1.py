#!/usr/bin/env python3
"""Independent native packet orchestration on explicit synthetic F32 inputs."""
import hashlib,json,tempfile
from pathlib import Path
import numpy as np
from independent_q8_1_activation_v1 import encode
from verify_layer0_packets_v1 import verify,PAIRS


def main():
 with tempfile.TemporaryDirectory(prefix='l0-packet-orchestration-cpu-') as name:
  root=Path(name);rows=[]
  for prefix in [1,2,4,8]:
   fields=[]
   for source,packet,cols in PAIRS:
    raw=np.linspace(-1,1,cols,dtype=np.float32).astype('<f4').tobytes();packed=encode(raw,1,cols);a=root/f'{prefix}-{source}.bin';b=root/f'{prefix}-{packet}.bin';a.write_bytes(raw);b.write_bytes(packed)
    fields.extend([{'name':source,'path':str(a),'sha256':hashlib.sha256(raw).hexdigest()},{'name':packet,'path':str(b),'sha256':hashlib.sha256(packed).hexdigest()}])
   rows.append({'prefix':prefix,'layer0':{'observed':fields}})
  assert verify(rows)['passed'];field=rows[0]['layer0']['observed'][1];path=Path(field['path']);raw=bytearray(path.read_bytes());raw[4]^=1;path.write_bytes(raw);field['sha256']=hashlib.sha256(raw).hexdigest()
  try:verify(rows)
  except ValueError:pass
  else:raise AssertionError('Corrupted actual packet accepted with matching file SHA')
 result={'CONFIG':'synthetic supplied F32 native packets, four-prefix12packet lane; no GPU/model','COMMAND':'python3 strata/flash-next/test_layer0_packet_orchestration_cpu_v1.py','RESULT':{'exact12packet_inputs':True,'packet_code_corruption_even_rehashed_rejected':True,'numpy':np.__version__},'VERDICT':'PASS conditional source-packet checker controls only; actual21 packets and own-state math unqualified','full_model_math_qualified':False}
 Path(__file__).with_name('layer0-packet-orchestration-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS independent12packet orchestration and bytecorruption control; no GPU/fullmath')
if __name__=='__main__':main()
