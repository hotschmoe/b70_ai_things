#!/usr/bin/env python3
"""Bounded synthetic own-state/config/layout controls; no original payload/GPU."""
import gc,hashlib,inspect,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import owned_layer0_gdn_replay_v1 as q
import explore_owned_layer0_gdn_num10_v1 as cli
from full48_synthetic_original_roles_v1 import SyntheticOriginalRoles
from test_full48_route_reference_cpu_v2 import config
import test_ple_prompt35_cpu_v1 as f35

class OwnLayer0Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):f35.SourceTests.setUpClass();cls.source=f35.SourceTests.source
 @classmethod
 def tearDownClass(cls):f35.SourceTests.tearDownClass()
 def tearDown(self):gc.collect()
 def model(self,provider=None,pair=False):
  args,env=config(pair);return q.OwnedLayer0GdnReplay(provider or SyntheticOriginalRoles(),'a'*64,args,env,self.source)
 def test_zero_state_and_full_owned_field_roster(self):
  model=self.model();result=model.tokens([19]);row=result['rows'][0];self.assertEqual(set(row['fields_owned']),set(q.F32_SHAPES));self.assertEqual(set(row['packets_owned']),set(q.PACKET_SHAPES));self.assertTrue(np.count_nonzero(row['fields_owned']['gdn_state_before'])==0);self.assertTrue(np.count_nonzero(row['fields_owned']['gdn_conv_before'])==0);self.assertFalse(result['actual_original_payload_used']);self.assertFalse(result['captured_inputs_used']);self.assertFalse(result['full_model_math_qualified']);self.assertFalse(result['numerical_tolerance_assigned']);self.assertLessEqual(model.p.max_rows_bytes,64<<20)
 def test_allprefixes_route_windows_and_own_causal_history(self):
  model=self.model()
  for n in (1,2,4,8):
   result=model.tokens([19+i for i in range(n)]);self.assertEqual(len(result['rows']),n);self.assertTrue(all(row['route']=='verifier' for row in result['rows']));self.assertEqual(result['rows'][-1]['normal_dispatch'],'target_verify')
   for previous,row in zip(result['rows'],result['rows'][1:]):
    self.assertTrue(np.array_equal(previous['outgoing_state_owned']['recurrent'],row['incoming_state_owned']['recurrent']));self.assertTrue(np.array_equal(previous['outgoing_state_owned']['conv'],row['incoming_state_owned']['conv']))
   self.assertTrue(np.array_equal(result['rows'][-1]['fields_owned']['gdn_state_after'],q.physical_recurrent(result['state_owned']['recurrent'])))
 def test_repeat_reset_and_changed_prefix_changes_owned_state(self):
  model=self.model();a=model.tokens([19,20]);b=model.tokens([19,20]);c=model.tokens([20,19]);self.assertTrue(np.array_equal(a['state_owned']['recurrent'],b['state_owned']['recurrent']));self.assertFalse(np.array_equal(a['state_owned']['recurrent'],c['state_owned']['recurrent']))
  self.assertTrue(np.count_nonzero(c['rows'][0]['incoming_state_owned']['recurrent'])==0)
 def test_state_route_native_inputs_not_accepted(self):
  model=self.model();self.assertEqual(list(inspect.signature(model.tokens).parameters),['token_ids'])
  for key in ('state','native_inputs','native_route','captured_state'):
   with self.assertRaises(TypeError):model.tokens([19],**{key:{}})
 def test_recurrent_and_history_physical_permutations(self):
  owned=np.arange(48*128*128,dtype=np.float32).reshape(48,128,128);physical=q.physical_recurrent(owned);self.assertEqual(physical.shape,(128,48,128));self.assertEqual(physical[7,3,11],owned[3,7,11]);self.assertTrue(np.array_equal(physical.transpose(1,0,2),owned));self.assertFalse(np.array_equal(physical.reshape(48,128,128),owned))
  hist=np.arange(3*10240,dtype=np.float32).reshape(3,10240);native=q.physical_conv(hist);self.assertEqual(native[27,1],hist[1,27]);self.assertTrue(np.array_equal(native.T,hist))
  with self.assertRaises(ValueError):q.physical_recurrent(np.zeros((128,48,128)))
  with self.assertRaises(ValueError):q.physical_conv(np.zeros((10240,3)))
 def test_pair_roster_matches_one_owned_component_values(self):
  one=self.model().tokens([19,20]);pair=self.model(pair=True).tokens([19,20]);self.assertTrue(np.array_equal(one['state_owned']['recurrent'],pair['state_owned']['recurrent']));self.assertFalse(pair['actual_T_group_rounding_qualified'])
 def test_mathflag_or_incomingcache_fails_before_original_rows(self):
  args,env=config();p=SyntheticOriginalRoles()
  for key in ('STRATA_VERIFY_EAGER','STRATA_CKPT_REREAD','STRATA_GDN_REC_HEADS'):
   with self.assertRaises(ValueError):q.OwnedLayer0GdnReplay(p,'a'*64,args,{**env,key:'0'},self.source)
  with self.assertRaises(ValueError):q.OwnedLayer0GdnReplay(p,'a'*64,args+['--conversation-cache-mib','1'],env,self.source)
  self.assertEqual(p.row_reads,0)
 def test_invalid_ids_and_role_fail_closed(self):
  model=self.model()
  for ids in ([],[1,2,3],[-1],[248320],[True],[19,248045]):
   with self.assertRaises(ValueError):model.tokens(ids)
  provider=SyntheticOriginalRoles();provider.reader.tensors['blk.0.attn_qkv.weight'][1]['type']='Q4_K'
  with self.assertRaises(ValueError):self.model(provider)
 def test_final_target_comparison_cannot_change_own_state(self):
  result=self.model().tokens([19]);row=result['rows'][-1];state=result['state_owned']['recurrent'].copy();values={k:np.asarray(v).copy() for k,v in row['fields_owned'].items()};packets=dict(row['packets_owned']);proof=cli.compare_final(result,values,packets);self.assertIsNone(proof['first_field_bitwise_difference']);self.assertFalse(proof['numeric_pass_claim']);self.assertIsNone(proof['tolerance_gate'])
  values['gdn_state_before'][0,0,0]=1;proof=cli.compare_final(result,values,packets);self.assertEqual(proof['first_field_bitwise_difference'],'gdn_state_before');self.assertTrue(np.array_equal(state,result['state_owned']['recurrent']))
 def test_saved_owned_fields_bytes_and_packets(self):
  result=self.model().tokens([19])
  with tempfile.TemporaryDirectory(prefix='ownedL0-CPU-') as name:
   root=Path(name);arrays,packets=cli.save_owned(root,result);self.assertEqual(len(arrays),22);self.assertEqual(len(packets),2)
   for row in arrays.values():self.assertEqual(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(),row['sha256'])
   with self.assertRaises(ValueError):cli.save_owned(root,result)
 def test_native_target_adapter_binds_finalized_requests_and_raw_shapes(self):
  import copy,json
  from layer0_numerical_qualification_v10 import field_contract
  owned=self.model().tokens([19]);fields=owned['rows'][0]['fields_owned'];packets=owned['rows'][0]['packets_owned']
  with tempfile.TemporaryDirectory(prefix='L0-NUM10-target-CPU-') as name:
   root=Path(name);directory=root/'child/candidatecombined_on';blobs=directory/'layer0';blobs.mkdir(parents=True);cli.write(root/'child/plan.snapshot.json',{'CPU_SYNTHETIC':True});binding=cli.sha(root/'child/plan.snapshot.json');observed=[]
   for contract in field_contract():
    key=contract['name'];path=blobs/(key+'.raw')
    if key in fields:raw=np.asarray(fields[key],dtype='<f4').tobytes()
    elif key in packets:raw=packets[key]
    else:raw=bytes(contract['bytes'])
    path.write_bytes(raw);observed.append({'name':key,'path':str(path),'sha256':cli.sha(path),'bytes':len(raw),'encoding':contract['encoding'],'provenance':contract['provenance']})
   frame={'position':0,'token':19,'rows':1,'layer':0,'binding_sha256':binding,'gen_ids':[19],'pid':123,'request':1}
   row={'prefix':1,'raw':{'ids':[19],'fresh':1},'layer0':{'frame':frame,'observed':observed,'metadata_sha256':'a'*64}}
   rows=[row]+[dict(row,prefix=n) for n in [2,4,8]];cli.write(directory/'requests.json',rows);plan={'prefixes':{'1':[19]}}
   ids,values,packetvalues,proof=cli.load_target(root,plan,1,rows);self.assertEqual(ids,[19]);self.assertEqual(set(values),set(fields));self.assertEqual(packetvalues,packets);self.assertTrue(proof['captured_values_used_only_as_output_comparison_targets'])
   with self.assertRaises(ValueError):cli.load_target(root,plan,1,rows[:-1])
   target=blobs/'gdn_state_before.raw';before=target.read_bytes();target.write_bytes(before[:-4])
   with self.assertRaises(ValueError):cli.load_target(root,plan,1,rows)
   target.write_bytes(before);target.unlink();target.symlink_to(blobs/'gdn_state_after.raw')
   with self.assertRaises(ValueError):cli.load_target(root,plan,1,rows)
 def test_completed_original_replay_requires_allrows_and_both_state_bytes(self):
  import copy
  owned=self.model().tokens([19,20]);args,env=config();plan={'args':args,'engine_receipt_sha256':'a'*64,'prepared':'CPU_FAKE_PREPARED'}
  with tempfile.TemporaryDirectory(prefix='L0-original-replay-CPU-') as name:
   root=Path(name);arrays={}
   for row in owned['rows']:
    for phase,key in [('input','residual_input'),('attention','residual_after_attn')]:
     label='p%d_l0_%s'%(row['position'],phase);arrays[label]=cli.save_array(root,label+'.f32',row['fields_owned'][key])
   for key in ('recurrent','conv'):arrays['gdn0_'+key]=cli.save_array(root,'state-'+key+'.f32',owned['state_owned'][key])
   binding={key:'a'*64 for key in ('parent_sha256','child_sha256','plan_sha256','source_plan_sha256')};post={'CPU_SYNTHETIC':True,'finished':2};report={'status':'exploratory complete; NO numerical qualification','errors':[],'numeric_pass_claim':False,'full_model_math_qualified':False,'tolerance_gate':None,'dependency_sha256':{'CPU_SYNTHETIC':True},'native_finalized_binding':binding,'declared_schedule':owned['declared_schedule'],'source_dispatch_binding':owned['source_dispatch_binding'],'post_dispatch_binding':owned['source_dispatch_binding'],'computation_terminal_epoch':1,'finished_epoch':3,'post_original_identity':post,'exploration':{'arrays':arrays}}
   cli.write(root/'report.json',report);realread=cli.read
   def mockedread(path):return {'env':env} if str(path)=='CPU_FAKE_PREPARED/server-config.json' else realread(path)
   with patch.object(cli,'read',mockedread),patch.object(cli.original,'dependency_binding',lambda:{'CPU_SYNTHETIC':True}),patch.object(cli.original,'finalized_binding',lambda path:({}, {}, plan,binding)),patch.object(cli,'identity_admission',lambda *a:post),patch.object(cli,'guard',lambda *a:{}):
    proof=cli.compare_completed_original(owned,root,cli.sha(root/'report.json'),root,root,[root]*4,plan,env);self.assertTrue(proof['allrows_and_finalstates_replay_bitwise_equal']);self.assertEqual(len(proof['checks']),6)
    target=Path(arrays['gdn0_recurrent']['path']);data=bytearray(target.read_bytes());data[:4]=np.asarray([1.],dtype='<f4').tobytes();target.write_bytes(data);arrays['gdn0_recurrent']['sha256']=cli.sha(target);cli.write(root/'report.json',report)
    with self.assertRaises(ValueError):cli.compare_completed_original(owned,root,cli.sha(root/'report.json'),root,root,[root]*4,plan,env)
 def test_missing_NUM10_never_falls_back(self):
  with patch.dict('sys.modules',{'validate_layer0_numerical_final_v10':None}):
   with self.assertRaises(ModuleNotFoundError):cli.num10_admission('/CPU_NOT_ACTUAL_RUNTIME')
if __name__=='__main__':unittest.main()
