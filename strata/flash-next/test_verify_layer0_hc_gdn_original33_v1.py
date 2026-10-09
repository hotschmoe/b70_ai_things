#!/usr/bin/env python3
"""Bounded synthetic33 admission controls; no actual model/GPU reads."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import verify_layer0_hc_gdn_original33_v1 as c
import test_verify_layer0_ffn_original_v1 as ff_fixture
import test_verify_layer0_gdn_original_v1 as gdn_fixture

class SchemaTests(unittest.TestCase):
 def setUp(self):
  self.fixture=ff_fixture.FrameTests('test_complete33_bound_frame_positive');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.row=self.fixture.row
 def test_genuine33_synthetic_positive(self):
  values,packets,binding=c.load33(self.row);self.assertEqual(len(binding['complete33_source_bindings']['fields']),33);self.assertEqual(set(packets),set(c.gdn.PACKETS));self.assertEqual(values['gdn_state_before'].shape,(786432,))
 def test_no_old26_loader(self):
  self.assertNotIn('load_fields(',Path(c.__file__).read_text());self.assertNotIn('gdn.verify(',Path(c.__file__).read_text())
 def test_frozen26_admission_still_rejects33(self):
  self.row['layer0']['full_model_math_qualified']=False
  with self.assertRaises(ValueError):c.gdn.load_fields(self.row)
 def test_partial26_rejected(self):
  self.row['layer0']['observed']=self.row['layer0']['observed'][:24]+self.row['layer0']['observed'][31:]
  with self.assertRaises(ValueError):c.load33(self.row)
 def test_derived_claim_source_rejected(self):
  self.fixture.frame['fields'][25]['provenance']='actual_buffer';self.fixture.publish()
  with self.assertRaises(ValueError):c.load33(self.row)
 def test_stale_nonce_rejected(self):
  self.fixture.frame['completed_nonce']+=1;self.fixture.publish()
  with self.assertRaises(ValueError):c.load33(self.row)
 def test_wrong_native_packet_rejected(self):
  field=next(f for f in self.row['layer0']['observed'] if f['name']=='attn_input_q81');raw=bytearray(Path(field['path']).read_bytes());raw[4]=1;Path(field['path']).write_bytes(raw);field['sha256']=c.sha(field['path'])
  with self.assertRaises(ValueError):c.load33(self.row)
 def test_nonfinite_hc_source_rejected(self):
  field=next(f for f in self.row['layer0']['observed'] if f['name']=='attn_hc_low');raw=np.zeros(320,dtype='<f4');raw[0]=np.nan;Path(field['path']).write_bytes(raw.tobytes());field['sha256']=c.sha(field['path'])
  with self.assertRaises(ValueError):c.load33(self.row)
 def test_missing_run_roster_rejected(self):
  with self.assertRaises(ValueError):c.validate_roster([self.row])
 def test_crossrequest_reuse_rejected(self):
  rows=[]
  for i,n in enumerate((1,2,4,8)):
   row=copy.deepcopy(self.row);row['prefix']=n;row['layer0']['frame']['request']=i+1;rows.append(row)
  with self.assertRaises(ValueError):c.validate_roster(rows)
 def test_source_dependencies_and_tolerances(self):
  c.source_pins();self.assertIs(c.gdn.conditional_gdn,__import__('verify_layer0_gdn_original_v1').conditional_gdn);self.assertIs(c.hc.check,__import__('verify_layer0_hc_original_v1').check)

class MathTests(unittest.TestCase):
 def test_original_gdn_fixture_and_incoming_state_negative(self):
  fixture=gdn_fixture.ConditionalTests('test_gdn_positive');fixture.setUp()
  self.assertTrue(all(v['passed'] for v in c.gdn.conditional_gdn(fixture.v,fixture.gamma).values()))
  fixture.v['gdn_state_before']+=.1
  self.assertFalse(all(v['passed'] for v in c.gdn.conditional_gdn(fixture.v,fixture.gamma).values()))
 def test_hc_scalar_seams_positive_and_negative(self):
  v={k:np.zeros(n) for k,n in {**c.gdn.F32,**c.HC_F32}.items()}
  def hc_read(residual,half):return {'normalized':np.zeros(10240),'low_silu':np.zeros(320),'gate':np.zeros(10240),'inject':np.ones(4),'mixed':np.zeros(2560)}
  layer=SimpleNamespace(hc_read=hc_read,hc_write=lambda residual,block,inject:residual+np.asarray(inject)[:,None]*block)
  v['attn_hc_inject']=np.ones(4)
  self.assertTrue(all(x['passed'] for x in c.conditional_hc(layer,v).values()))
  v['gdn_block_output']+=1
  self.assertFalse(c.conditional_hc(layer,v)['attn_hc_write']['passed'])
 def test_exact_original_roles_and_type_negative(self):
  tensors={name:({'path':'CPU fixture'},dict(type=kind,shape_ggml_order=list(shape),absolute_offset=0,packed_bytes=0),[1,2,3,4,5]) for name,(kind,shape) in {**c.gdn.ROLES,**c.HC_ROLES}.items()}
  provider=SimpleNamespace(reader=SimpleNamespace(tensors=tensors));self.assertEqual(len(c.role_bindings(provider)),13)
  tensors['blk.0.hc_attn_down.weight'][1]['type']='BF16'
  with self.assertRaises(ValueError):c.role_bindings(provider)

class RunReceiptTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);(self.root/'child/candidatecombined_on').mkdir(parents=True);self.engine=self.root/'CPU_mock_engine';self.engine.mkdir();(self.engine/'receipt.json').write_text('CPU fake SDK metadata ONLY')
  self.plan={'engine_root':str(self.engine),'engine_receipt_sha256':c.sha(self.engine/'receipt.json'),'observed_fields':33,'source_value_fields':31,'derived_value_fields':2,'layout_fields':33};self.planpath=self.root/'plan.json';self.write(self.planpath,self.plan);digest=c.sha(self.planpath)
  self.parent={'passed':True,'child_return_code':0,'owned_containers_terminal':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'errors':[],'interrupted':False,'forced_cleanup':False,'plan':str(self.planpath),'plan_sha256':digest,'child_terminal_epoch':10.,'finished_epoch':30.}
  self.identity={'passed':True,'rows':[{}]*4,'started':20.,'finished':25.,'after_child_terminal_epoch':10.};self.ip=self.root/'post-model-identity.json';self.write(self.ip,self.identity)
  self.report={'passed':True,'numerical_and_teardown_passed':True,'post_health_passed':True,'layer0_logical_lifecycle_qualified':True,'packet_contract_qualified':True,'full_model_math_qualified':False,'plan_sha256':digest,'engine_receipt_sha256':self.plan['engine_receipt_sha256'],'post_model_identity':{'path':str(self.ip),'sha256':c.sha(self.ip)}}
  self.requests=[{'prefix':n,'layer0':{'frame':{'binding_sha256':digest}}} for n in (1,2,4,8)];self.publish()
 def write(self,p,j):p.write_text(json.dumps(j))
 def publish(self):
  self.write(self.root/'parent-qualification.json',self.parent);self.write(self.root/'child/report.json',self.report);self.write(self.root/'child/candidatecombined_on/requests.json',self.requests)
 def test_synthetic_receipt_admission_positive(self):self.assertEqual(len(c.run_admission(self.root)[0]),4)
 def test_original_postidentity_byte_change_rejected(self):
  self.ip.write_text('{}')
  with self.assertRaises(ValueError):c.run_admission(self.root)
 def test_prior26_schema_rejected(self):
  self.plan['observed_fields']=26;self.write(self.planpath,self.plan)
  with self.assertRaises(ValueError):c.run_admission(self.root)
 def test_failed_terminal_parent_rejected(self):
  self.parent['owned_containers_terminal']=False;self.publish()
  with self.assertRaises(ValueError):c.run_admission(self.root)
 def test_preterminal_identity_rejected(self):
  self.identity['started']=9.;self.write(self.ip,self.identity);self.report['post_model_identity']['sha256']=c.sha(self.ip);self.publish()
  with self.assertRaises(ValueError):c.run_admission(self.root)
 def test_wrong_source_binding_rejected(self):
  self.requests[0]['layer0']['frame']['binding_sha256']='0'*64;self.publish()
  with self.assertRaises(ValueError):c.run_admission(self.root)
if __name__=='__main__':unittest.main()
