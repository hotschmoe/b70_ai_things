#!/usr/bin/env python3
"""CPU source/coverage negatives; no actual model/native GPU evidence."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from collect_prefix_residual30_v1 import collect

class CoverageTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.meta=self.root/'p30-mock.json';self.request={1:[19]};self.stage={0:(0,48)};self.fields=[]
  for layer in range(48):
   for phase in ('input','attention','ffn'):
    path=self.root/('CPU-mock-l%d-%s.f32'%(layer,phase));path.write_bytes(bytes(40960));self.fields.append(dict(layer=layer,phase=phase,bytes=40960,file=str(path),encoding='LE_F32[rows,4,2560]'))
  self.data=dict(schema=1,pid=9,request=1,stage=0,lb=0,le=48,route='verifier',native_hc_source=True,first_position=0,rows=1,row_floats=10240,source_extent_bytes=40960,nonce=(9<<32)|1,binding_sha256='a'*64,gen_ids=[19],fields=self.fields,full_model_math_qualified=False);self.publish()
 def publish(self):self.meta.write_text(json.dumps(self.data))
 def run_collect(self):return collect(self.root,self.request,self.stage,'a'*64)
 def test_synthetic_all48_threephase_positive(self):self.assertTrue(self.run_collect()['passed'])
 def test_no_or_partial_positive(self):
  self.meta.unlink()
  with self.assertRaises(ValueError):self.run_collect()
 def test_missing_phase_negative(self):
  self.data['fields'].pop();self.publish()
  with self.assertRaises(ValueError):self.run_collect()
 def test_incomplete_stage_roster_negative(self):
  self.stage={0:(0,47)}
  with self.assertRaises(ValueError):self.run_collect()
 def test_token_source_stage_nonce_negative(self):
  for field,value in [('gen_ids',[20]),('binding_sha256','b'*64),('le',47),('nonce',0)]:
   before=copy.deepcopy(self.data);self.data[field]=value;self.publish()
   with self.assertRaises(ValueError):self.run_collect()
   self.data=before
 def test_matrix_extent_and_rowboundary_negative(self):
  self.data['source_extent_bytes']=4096;self.publish()
  with self.assertRaises(ValueError):self.run_collect()
 def test_nonfinite_negative(self):
  path=Path(self.fields[0]['file']);raw=np.zeros(10240,dtype='<f4');raw[0]=np.nan;path.write_bytes(raw.tobytes())
  with self.assertRaises(ValueError):self.run_collect()
 def test_unmatched_request_negative(self):
  self.data['request']=2;self.publish()
  with self.assertRaises(ValueError):self.run_collect()
 def test_duplicate_stage_row_negative(self):
  (self.root/'p30-duplicate.json').write_text(json.dumps(self.data))
  with self.assertRaises(ValueError):self.run_collect()
 def test_entire_earlier_prefill_matrix_required(self):
  self.request={1:[18,19]};self.data['gen_ids']=[18,19];self.data['first_position']=1;self.publish()
  with self.assertRaises(ValueError):self.run_collect()

class BindingTests(CoverageTests):
 def producer_log(self):
  log=self.root/'producer.log';log.write_text('SFD request pid=9 request=1 tokens=1 ids=19 shape=4x2560 first_logits248320\nSFD resume pid=9 request=1 reused=0 evaluated_prompt_pending=1\nP30 frame pid=9 request=1 stage=0 route=verifier rows=1 p0=0 metadata='+str(self.meta)+'\n');return log
 def test_producer_source_binding_positive_and_negatives(self):
  log=self.producer_log();self.assertTrue(collect(self.root,self.request,self.stage,'a'*64,log)['producer_log_crossbinding_verified']);original=log.read_text()
  for before,after in [('ids=19','ids=20'),('reused=0','reused=1'),('P30 frame pid=9','P30 frame pid=10'),(' rows=1',' rows=2')]:
   log.write_text(original.replace(before,after))
   with self.assertRaises(ValueError):collect(self.root,self.request,self.stage,'a'*64,log)
 def test_pair_whole_prefill_matrix_positive(self):
  self.meta.unlink();self.request={1:[17,18,19,20]};self.stage={0:(0,32),1:(32,48)}
  for stage,(lb,le) in self.stage.items():
   for route,p0,n in [('prefill',0,3),('verifier',3,1)]:
    fields=[]
    for layer in range(lb,le):
     for phase in ('input','attention','ffn'):
      path=self.root/('%s-%d-%d-%s.f32'%(route,stage,layer,phase));raw=np.full((n,10240),layer+0.25,dtype='<f4');path.write_bytes(raw.tobytes());fields.append(dict(layer=layer,phase=phase,bytes=n*40960,file=str(path),encoding='LE_F32[rows,4,2560]'))
    data={**self.data,'stage':stage,'lb':lb,'le':le,'gen_ids':self.request[1],'route':route,'first_position':p0,'rows':n,'source_extent_bytes':n*40960,'fields':fields};(self.root/('p30-%s-%d.json'%(route,stage))).write_text(json.dumps(data))
  result=self.run_collect();self.assertEqual(len(result['frames']),4)
  data=json.loads((self.root/'p30-prefill-0.json').read_text());data['rows']=1;data['source_extent_bytes']=40960;(self.root/'p30-prefill-0.json').write_text(json.dumps(data))
  with self.assertRaises(ValueError):self.run_collect()
 def test_consumed_shadow_and_off_source_contract(self):
  patch=(Path(__file__).parent/'patches/0030-sycl-default-off-full-prefix-residual-capture.patch').read_text()
  self.assertIn('sycl/include/strata/core/verify.hpp',patch);self.assertIn('bool prefix30_chunk=false',patch);self.assertIn('prefix30_snapshot_.reset(); // All serial graphs retired',patch)

if __name__=='__main__':unittest.main()
