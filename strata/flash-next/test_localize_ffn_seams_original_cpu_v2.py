"""Synthetic original-role seam controls; no actual model/native/GPU receipts."""
import ast,copy,json,tempfile,unittest,sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import localize_ffn_seams_original_v2 as q
from full48_synthetic_original_roles_v1 import SyntheticOriginalRoles
from full48_owned_composition_storage_v2 import RouteAwareFull48OwnedComposition,derive_schedule
from test_full48_route_reference_cpu_v2 import config
import test_ple_prompt35_cpu_v1 as source35
import os

class ConditionalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):source35.SourceTests.setUpClass();args,env=config();cls.model=RouteAwareFull48OwnedComposition(SyntheticOriginalRoles(),'a'*64,args,env,source35.SourceTests.source)
 @classmethod
 def tearDownClass(cls):source35.SourceTests.tearDownClass()
 def test_existing_original_FFN_chain_replay_and_packet_provenance(self):
  x=np.broadcast_to(np.linspace(-.2,.3,2560,dtype='<f4'),(4,2560)).copy()
  for layer in q.LAYERS:
   a,first=q.conditional_case(self.model,[19,22],layer,x,np.zeros_like(x));b,second=q.conditional_case(self.model,[19,22],layer,x,a)
   self.assertTrue(np.array_equal(a,b));self.assertTrue(second['output_metrics']['bitwise_equal']);self.assertEqual(first['independently_derived_router_ids'],second['independently_derived_router_ids']);self.assertEqual(len(first['independently_derived_router_ids']),self.model.g.topk)
   self.assertEqual(first['independently_derived_mixed_q8_packet_sha256'],second['independently_derived_mixed_q8_packet_sha256']);self.assertTrue(first['packet_events_independently_derived']);self.assertTrue(all(not e['stored_s_used'] for e in first['packet_events_independently_derived']));self.assertFalse(first['native_mixed_or_packet_observed']);self.assertFalse(first['full_model_math_qualified']);self.assertIsNone(first['tolerance_gate'])
 def test_exact_8_cases_at_absolute_row1(self):
  self.assertEqual(q.LAYERS,(1,2,4,10));self.assertEqual(len(q.case_roster()),8);self.assertEqual({position for position,layer,lane in q.case_roster()},{1})
 def test_route_from_config_changes_windowshape_not_native_metadata(self):
  x=np.zeros((4,2560),dtype='<f4');_,last=q.conditional_case(self.model,[19,22],1,x,x);_,earlier=q.conditional_case(self.model,[19,22,23,24],1,x,x)
  self.assertEqual(last['config_derived_window']['normal_dispatch'],'target_verify');self.assertEqual(earlier['config_derived_window']['normal_dispatch'],'prompt_verify');self.assertEqual(earlier['config_derived_window']['rows'],2);self.assertFalse(earlier['native_route_used_as_math_input']);self.assertFalse(earlier['actual_window_group_shape_rounding_qualified'])
 def test_perturbing_conditional_input_changes_packet_and_output(self):
  x=np.zeros((4,2560),dtype='<f4');y=x.copy();y[0,101]=.75
  a,left=q.conditional_case(self.model,[19,22],4,x,x);b,right=q.conditional_case(self.model,[19,22],4,y,x)
  self.assertNotEqual(left['attention_sha256'],right['attention_sha256']);self.assertNotEqual(left['independently_derived_mixed_q8_packet_sha256'],right['independently_derived_mixed_q8_packet_sha256']);self.assertFalse(np.array_equal(a,b));self.assertGreater(right['output_metrics']['nmse'],0)
 def test_bad_layer_shape_and_nonfinite_refused(self):
  x=np.zeros((4,2560),dtype='<f4')
  for layer,attention,target in [(3,x,x),(4,x.reshape(-1),x),(4,x,np.full_like(x,np.nan))]:
   with self.assertRaises(ValueError):q.conditional_case(self.model,[19,22],layer,attention,target)
 def test_never_calls_original_fullmodel_tokens_or_uses_native_router(self):
  tree=ast.parse(Path(q.__file__).read_text());calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
  self.assertFalse(any(name.endswith('.tokens') for name in calls));self.assertIn('model.hc_read',calls);self.assertIn('model.hc_write',calls);self.assertIn("model.ffn[layer].run",calls)

class PreflightTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.native={'head':np.zeros(248320,dtype='<f4')};self.observed={'nativePID':9};self.ident={'path':str(self.root/'post-original-model-identity.json'),'sha256':'mock','started':3.,'finished':4.}
  row=q.original.save_array(self.root,'synthetic-matrix.f32',np.zeros((4,2560),dtype='<f4'));head=q.original.save_array(self.root,'synthetic-head.f32',self.native['head']);arrays={'head':head,'native_head':head,'CPU_extra_state':row,**{'p%d_l%d_%s'%(pos,l,p):row for pos in range(2) for l in range(48) for p in ('input','attention','ffn')}}
  self.binding={key:'CPU-only' for key in ('parent_sha256','child_sha256','plan_sha256','source_plan_sha256')};self.dependencies={'original_dependencies':{'CPU':'synthetic-only'}}
  self.report={'status':'exploratory complete; NO numerical qualification','errors':[],'numeric_pass_claim':False,'full_model_math_qualified':False,'tolerance_gate':None,'dependency_sha256':self.dependencies['original_dependencies'],'native_finalized_binding':self.binding,'native_observation_binding':self.observed,'computation_terminal_epoch':2.,'finished_epoch':5.,'post_original_identity':self.ident,'exploration':{'arrays':arrays},'runtime':{'native_config_env':{'CPU':'only'}}};q.write(self.root/'report.json',self.report);self.prepared=self.root/'prepared';self.prepared.mkdir();q.write(self.prepared/'server-config.json',{'env':{'CPU':'only'}});args,env=config();self.plan={'prepared':str(self.prepared),'args':args,'engine_root':str(self.root)};self.report['declared_schedule']=derive_schedule([19,22],args,env);self.report['runtime']={'native_config_env':env,'numpy_version':np.__version__,'numpy_module_sha256':q.sha(np.__file__),'numpy_binary_sha256':{str(p):q.sha(p) for p in sorted(Path(np.__file__).parent.rglob('*.so'))},'env':{k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}};self.report['source_dispatch_binding']=self.report['post_dispatch_binding']={'CPU_source3':'mock'};q.write(self.prepared/'server-config.json',{'env':env});self.save()
 def preflight(self):
  with patch.object(q,'source_binding',return_value=self.dependencies),patch.object(q.original,'finalized_binding',return_value=({}, {}, self.plan,self.binding)),patch.object(q.original,'native_prefix',return_value=([19,22],self.native,self.observed)),patch.object(q,'bind_dispatch_source',return_value={'CPU_source3':'mock'}),patch.object(q.original,'identity_admission',return_value=self.ident):return q.preflight(self.root,self.root,q.sha(self.root/'report.json'),self.root/'lock',[])
 def save(self):q.write(self.root/'report.json',self.report)
 def test_tiny_synthetic_positive_preflight(self):self.assertEqual(self.preflight()[3]['p1_l4_ffn'].shape,(4,2560))
 def test_pending_or_failed_original_scan_refused(self):
  for key,value in [('status','computed; final full4 source proof pending'),('errors',['source4 failed']),('full_model_math_qualified',True)]:
   old=self.report[key];self.report[key]=value;self.save()
   with self.assertRaises(ValueError):self.preflight()
   self.report[key]=old
 def test_native_run_or_raw_binding_drift_refused(self):
  self.report['native_finalized_binding']=dict(self.binding,parent_sha256='foreign');self.save()
  with self.assertRaises(ValueError):self.preflight()
 def test_dependency_drift_refused(self):
  self.report['dependency_sha256']={};self.save()
  with self.assertRaises(ValueError):self.preflight()
 def test_saved_source3_or_schedule_drift_refused(self):
  self.report['post_dispatch_binding']={'CPU_source3':'changed'};self.save()
  with self.assertRaises(ValueError):self.preflight()
 def test_matched_numpy_and_thread_profile_required(self):
  self.report['runtime']['env']['OMP_NUM_THREADS']='2';self.save()
  with self.assertRaises(ValueError):self.preflight()
 def test_ownarray_hash_read_race_rejected(self):
  row=self.report['exploration']['arrays']['p1_l4_ffn'];target=Path(row['path']);reader=Path.read_bytes;count=[0]
  def changed(path):
   raw=reader(path)
   if path==target:
    count[0]+=1
    if count[0]==2:raw=bytes([raw[0]^1])+raw[1:]
   return raw
  with patch.object(Path,'read_bytes',changed),self.assertRaisesRegex(ValueError,'changed/truncated'):q.own_array(self.root,row)
 def test_own_raw_mutation_or_outside_root_refused(self):
  row=self.report['exploration']['arrays']['p1_l4_ffn'];Path(row['path']).write_bytes(b'\1'*40960)
  with self.assertRaises(ValueError):self.preflight()
 def test_own_array_symlink_refused(self):
  row=dict(self.report['exploration']['arrays']['p1_l4_ffn']);link=self.root/'link';link.symlink_to(row['path']);row['path']=str(link)
  with self.assertRaises(ValueError):q.own_array(self.root,row)
 def test_report_SHA_rejected_before_native_or_payload(self):
  with patch.object(q,'source_binding',return_value=self.dependencies),patch.object(q.original,'finalized_binding',side_effect=AssertionError('No native/payload reads')):
   with self.assertRaises(ValueError):q.preflight(self.root,self.root,'0'*64,self.root/'lock',[])
 def test_post_CPU_identity_association_refused(self):
  self.report['post_original_identity']=dict(self.ident,sha256='oldscan');self.save()
  with self.assertRaises(ValueError):self.preflight()
 def test_failed_preflight_never_constructs_original_provider(self):
  output=self.root/'failed-admission';args=['seam','--native-run-root',str(self.root),'--original-run-root',str(self.root),'--original-report-sha256','a'*64,'--output',str(output)]
  with patch.object(sys,'argv',args),patch.object(q,'preflight',side_effect=ValueError('CPU negative incomplete original proof')),patch.object(q,'BoundOriginalFull48Rows',side_effect=AssertionError('No payload')) as provider,patch.object(q.original,'full_buffered_identity') as full4:
   self.assertEqual(q.main(),1);provider.assert_not_called();full4.assert_not_called()
  self.assertTrue(q.read(output/'report.json')['errors'])
 def test_failed_conditional_computation_still_scans_newfull4_and_finalpages(self):
  output=self.root/'failed-computation';args=['seam','--native-run-root',str(self.root),'--original-run-root',str(self.root),'--original-report-sha256','a'*64,'--output',str(output)]
  (self.root/'mock-identity').write_text('CPU mock identity; not model proof')
  inputs={'p1_l%d_%s'%(layer,phase):np.zeros((4,2560),dtype='<f4') for layer in q.LAYERS for phase in ('attention','ffn')};prior=dict(self.report,original_role_binding={});provider=SimpleNamespace(reader=SimpleNamespace(tensors={}))
  with patch.object(sys,'argv',args),patch.object(q,'preflight',return_value=(prior,self.plan,inputs,inputs,self.root/'mock-identity',{})),patch.object(q.original,'guard',return_value={'CPU_mock':True}) as guard,patch.object(q,'BoundOriginalFull48Rows',return_value=provider),patch.object(q,'RouteAwareFull48OwnedComposition',return_value=object()),patch.object(q,'conditional_case',side_effect=ValueError('CPU simulated compute failure')),patch.object(q.original,'full_buffered_identity',return_value={'passed':True}) as full4,patch.object(q.original,'identity_admission',return_value=self.ident):
   self.assertEqual(q.main(),1);self.assertEqual(full4.call_count,1);self.assertEqual(guard.call_count,2)
   report=q.read(output/'report.json');self.assertEqual(full4.call_args.args[4],report['computation_terminal_epoch']);self.assertEqual(Path(full4.call_args.args[3]).name,'post-seam-model-identity.json');self.assertIn('final_known_pages',report);self.assertIn('CPU simulated compute failure',report['errors']);self.assertFalse(report['full_model_math_qualified'])
 def test_frozen_dependency_source_plan(self):self.assertIn('original_dependencies',q.source_binding())
 def test_new_full4_and_pages_follow_payload_even_on_failure(self):
  source=Path(q.__file__).read_text();self.assertLess(source.index('=preflight('),source.index('provider=BoundOriginalFull48Rows'));self.assertIn('finally:',source);self.assertIn("out/'post-seam-model-identity.json',terminal",source);self.assertIn("watch('final')",source)

if __name__=='__main__':unittest.main()
