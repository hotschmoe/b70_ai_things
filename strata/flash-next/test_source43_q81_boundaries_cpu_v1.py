import copy,hashlib,json,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import audit_source43_q81_boundaries_v1 as q
from independent_q8_1_activation_v1 import encode,NATIVE
class Controls(unittest.TestCase):
 def raw(self,n):return struct.pack('<%df'%n,*[(i%31-15)*.125 for i in range(n)])
 def test_exact_both_shapes(self):
  for n in(2560,6144):
   raw=self.raw(n);v=q.packet_comparison(raw,encode(raw,1,n,NATIVE),n);self.assertTrue(v['packet_exact']);self.assertIsNone(v['first_differing_block']);self.assertFalse(v['original_reference_replaced'])
 def test_scale_sum_code_corruption(self):
  raw=self.raw(2560);packet=encode(raw,1,2560,NATIVE)
  for offset,field in((0,'scale_half'),(2,'sum_half'),(4,'code')):
   bad=bytearray(packet);bad[offset]^=1;v=q.packet_comparison(raw,bytes(bad),2560);self.assertFalse(v['packet_exact']);self.assertEqual(v['first_differing_field'],field);self.assertEqual(v['different_bytes'],1)
 def test_bounds_and_nonfinite_refused(self):
  raw=self.raw(2560);packet=encode(raw,1,2560,NATIVE)
  for a,b,n in((raw,packet,True),(raw[:-1],packet,2560),(raw,packet[:-1],2560),(raw,packet,32),(struct.pack('<f',float('nan'))+raw[4:],packet,2560)):
   with self.assertRaises(ValueError):q.packet_comparison(a,b,n)
 def fixture(self,parent):
  root=parent/'actor';directory=root/'source43-num10-ON/layer0';directory.mkdir(parents=True);contract=[(n,e,int(b))for n,e,b in q.re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',(q.HERE/'layer0_numerical_contract_v1.hpp').read_text())];artifacts={};requests=[];pid=47
  for index in range(1,5):
   fields=[];payload={}
   for raw_name,packet_name,n in q.SEAMS:payload[raw_name]=self.raw(n);payload[packet_name]=encode(payload[raw_name],1,n,NATIVE)
   for name,encoding,size in contract:
    path=directory/('%d-%s.bin'%(index,name));digest='a'*64
    if name in payload:path.write_bytes(payload[name]);digest=hashlib.sha256(payload[name]).hexdigest();artifacts[str(path.relative_to(root))]=digest
    fields.append({'name':name,'encoding':encoding,'bytes':size,'path':str(path),'sha256':digest,'provenance':'actual_buffer'})
   frame={'request':index,'pid':pid,'gen_ids':[248045],'token':248045,'position':0,'rows':1,'layer':0,'stage':0,'reused':0,'binding_sha256':q.prior.ACTOR_SOURCE,'completed_nonce':(pid<<32)|index,'fields':fields};meta=directory/('%d.json'%index);meta.write_text(json.dumps(frame));digest=hashlib.sha256(meta.read_bytes()).hexdigest();artifacts[str(meta.relative_to(root))]=digest;requests.append({'ordinal':index,'input_ids':[248045],'metadata':str(meta),'metadata_sha256':digest,'raw33':fields})
  report={'passed':True,'errors':[],'source_plan_sha256':q.prior.ACTOR_SOURCE,'plan_sha256':'b'*64,'artifacts':artifacts};rp=root/'report.json';rp.write_text(json.dumps(report));rh=hashlib.sha256(rp.read_bytes()).hexdigest();compact={'capture_root':str(directory.parent),'source_generation':43,'materialized_mode':1,'engine_pid':pid,'public_report_sha256':rh,'public_source_plan_sha256':q.prior.ACTOR_SOURCE,'binding_sha256':q.prior.ACTOR_SOURCE,'plan_sha256':'b'*64,'full_descriptor_sha256':'c'*64,'requests':requests};reference={'passed':True,'errors':[],'source_plan_sha256':q.prior.CLI2_SOURCE,'actor_root':str(root),'expected_actor_report_sha256':rh,'actual_reference_computation_attempted':True,'current43_capture_before':copy.deepcopy(compact),'current43_capture_after':compact};ref=parent/'reference.json';ref.write_text(json.dumps(reference));return root,rh,ref,hashlib.sha256(ref.read_bytes()).hexdigest()
 def test_actual_four_request_run_and_current_evidence(self):
  with tempfile.TemporaryDirectory()as t:
   parent=Path(t);root,rh,ref,fh=self.fixture(parent);v=q.run(root,rh,ref,fh,parent/'audit');self.assertTrue(v['audit']['all_eight_packets_exact']);self.assertEqual(len(v['consumed_evidence']),22);self.assertFalse(v['weights_read'])
 def test_mutated_raw_with_locally_refreshed_metadata_refused(self):
  with tempfile.TemporaryDirectory()as t:
   parent=Path(t);root,rh,ref,fh=self.fixture(parent);r=json.loads(ref.read_text());field=r['current43_capture_after']['requests'][0]['raw33'][5];p=Path(field['path']);p.write_bytes(bytes(p.stat().st_size));field['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['current43_capture_before']=copy.deepcopy(r['current43_capture_after']);ref.write_text(json.dumps(r));fh=hashlib.sha256(ref.read_bytes()).hexdigest()
   with self.assertRaises(ValueError):q.run(root,rh,ref,fh,parent/'audit')
 def test_owner_nonce_and_boolean_position_refused(self):
  for key,bad in(('pid',999),('completed_nonce',1),('position',False)):
   with tempfile.TemporaryDirectory()as t:
    parent=Path(t);root,rh,ref,fh=self.fixture(parent);r=json.loads(ref.read_text());row=r['current43_capture_after']['requests'][0];meta=Path(row['metadata']);frame=json.loads(meta.read_text());frame[key]=bad;meta.write_text(json.dumps(frame));row['metadata_sha256']=hashlib.sha256(meta.read_bytes()).hexdigest();r['current43_capture_before']=copy.deepcopy(r['current43_capture_after']);report=json.loads((root/'report.json').read_text());report['artifacts'][str(meta.relative_to(root))]=row['metadata_sha256'];(root/'report.json').write_text(json.dumps(report));rh=hashlib.sha256((root/'report.json').read_bytes()).hexdigest();r['expected_actor_report_sha256']=rh
    for c in(r['current43_capture_before'],r['current43_capture_after']):c['public_report_sha256']=rh
    ref.write_text(json.dumps(r));fh=hashlib.sha256(ref.read_bytes()).hexdigest()
    with self.assertRaises(ValueError):q.run(root,rh,ref,fh,parent/'audit')
 def test_failure_still_rechecks_consumed_prefix(self):
  with tempfile.TemporaryDirectory()as t:
   parent=Path(t);root,rh,ref,fh=self.fixture(parent)
   def fail(root,report,rh,reference,consume):consume(ref,fh,1<<20);ref.write_bytes(b'changed');raise ValueError('Original comparison failure')
   with patch.object(q,'audit',side_effect=fail):
    with self.assertRaises(ValueError):q.run(root,rh,ref,fh,parent/'audit')
   v=json.loads((parent/'audit/report.json').read_text());self.assertFalse(v['passed']);self.assertTrue(any('Closing:'in s for s in v['errors']));self.assertTrue(any('Original comparison failure'in s for s in v['errors']))
 def test_output_alias_refused_before_write(self):
  with tempfile.TemporaryDirectory()as t:
   parent=Path(t);root,rh,ref,fh=self.fixture(parent);alias=parent/'alias';alias.symlink_to(root,target_is_directory=True)
   with self.assertRaises(ValueError):q.run(root,rh,ref,fh,alias/'audit')
   self.assertFalse((root/'audit').exists())
if __name__=='__main__':unittest.main()
