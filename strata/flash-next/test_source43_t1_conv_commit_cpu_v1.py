import hashlib,json,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import audit_source43_t1_conv_commit_v1 as q
class Controls(unittest.TestCase):
 def test_exact_channel_history_commit_including_negative_zero(self):
  x=struct.pack('<4f',1.,-2.,-0.,4.);after=b''.join(bytes(8)+x[i*4:i*4+4]for i in range(4));r=q.exact_commit(x,bytes(48),after,4);self.assertTrue(r['exact_channel_zero_zero_qkv']);self.assertFalse(r['arithmetic_or_tolerance_used'])
 def test_nonzero_before_refused(self):
  with self.assertRaises(ValueError):q.exact_commit(bytes(16),b'\x01'+bytes(47),bytes(48),4)
 def test_wrong_history_column(self):
  x=struct.pack('<4f',1.,2.,3.,4.);after=b''.join(x[i*4:i*4+4]+bytes(8)for i in range(4));self.assertFalse(q.exact_commit(x,bytes(48),after,4)['exact_channel_zero_zero_qkv'])
 def test_transposed_layout_not_accepted(self):
  x=struct.pack('<4f',1.,2.,3.,4.);self.assertFalse(q.exact_commit(x,bytes(48),bytes(32)+x,4)['exact_channel_zero_zero_qkv'])
 def test_swapped_channel_corruption(self):
  x=struct.pack('<4f',1.,2.,3.,4.);reverse=b''.join(bytes(8)+x[i*4:i*4+4]for i in reversed(range(4)));self.assertEqual(q.exact_commit(x,bytes(48),reverse,4)['different_bytes'],4)
 def test_consumed_raw_mutation_and_alias(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);p=root/'raw';p.write_bytes(b'abcdef');h=hashlib.sha256(p.read_bytes()).hexdigest();q.consumed(p,h,6,6);p.write_bytes(b'abcdeg')
   with self.assertRaises(ValueError):q.consumed(p,h,6,6)
   a=root/'alias';a.symlink_to(p)
   with self.assertRaises(ValueError):q.consumed(a,h,6,6)
 def fixture(self,root):
  capture=root/'source43-num10-ON';directory=capture/'layer0';directory.mkdir(parents=True);artifacts={};rows=[];pid=47
  for i in range(1,5):
   x=struct.pack('<%df'%q.CHANNELS,*range(q.CHANNELS));raws={'gdn_qkv':x,'gdn_conv_before':bytes(q.CHANNELS*12),'gdn_conv_after':b''.join(bytes(8)+x[j*4:j*4+4]for j in range(q.CHANNELS))};fields=[]
   for name,raw in raws.items():
    p=directory/('%d-%s.bin'%(i,name));p.write_bytes(raw);h=hashlib.sha256(raw).hexdigest();artifacts[str(p.relative_to(root))]=h;fields.append({'name':name,'path':str(p),'bytes':len(raw),'sha256':h,'provenance':'actual_buffer','encoding':'LE_F32'})
   fields+=[{'name':'unused%d'%j}for j in range(30)];frame={'request':i,'pid':pid,'gen_ids':[248045],'token':248045,'position':0,'rows':1,'layer':0,'stage':0,'reused':0,'binding_sha256':q.ACTOR_SOURCE,'completed_nonce':(pid<<32)|i};meta=directory/('%d.json'%i);meta.write_text(json.dumps(frame));mh=hashlib.sha256(meta.read_bytes()).hexdigest();artifacts[str(meta.relative_to(root))]=mh;rows.append({'ordinal':i,'raw':{'ids':[248045],'fresh':1,'pin':None},'layer0':{'frame':frame,'metadata':str(meta),'metadata_sha256':mh,'observed':fields}})
  report={'passed':True,'errors':[],'source_plan_sha256':q.ACTOR_SOURCE,'plan_sha256':'b'*64,'artifacts':artifacts};p=root/'report.json';p.write_text(json.dumps(report));h=hashlib.sha256(p.read_bytes()).hexdigest();d={'public_actor_admission':{'passed':True,'root':str(root),'report_sha256':h,'readonly_reader_fix_source_plan_sha256':q.READER_FIX,'source_plan_sha256':q.ACTOR_SOURCE},'capture_root':str(capture),'source_generation':43,'materialized_mode':1,'NUM10_enabled':True,'QSA3_enabled':False,'binding_sha256':q.ACTOR_SOURCE,'plan_sha256':'b'*64,'engine_pid':pid,'requests':rows};return report,h,d
 def test_actual_shaped_four_request_saved_descriptor_audit(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);report,h,d=self.fixture(root);r=q.audit(root,{'sha256':h,'value':report},d);self.assertTrue(r['all_four_exact']);self.assertEqual(len(r['consumed_evidence']),16);self.assertFalse(r['current_model_or_semantic_admission_performed'])
 def test_wrong_corrected_reader_and_artifact_hash(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);report,h,d=self.fixture(root);d['public_actor_admission']['readonly_reader_fix_source_plan_sha256']='old'
   with self.assertRaises(ValueError):q.audit(root,{'sha256':h,'value':report},d)
   d['public_actor_admission']['readonly_reader_fix_source_plan_sha256']=q.READER_FIX;report['artifacts'][next(iter(report['artifacts']))]='changed'
   with self.assertRaises(ValueError):q.audit(root,{'sha256':h,'value':report},d)
if __name__=='__main__':unittest.main()
