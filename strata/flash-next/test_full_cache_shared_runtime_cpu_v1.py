"""CPU-only tiny phase/protocol/receipt controls; no model, GPU or Docker."""
import ast,copy,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_shared_phase_contract_v1 as p
import full_cache_shared_raw49_v1 as raw
import full_cache_shared_http_client_v1 as client
import run_full_cache_shared_runtime_v1 as run
class Contracts(unittest.TestCase):
 def case(self):
  # Saved authentic case metadata, no source3 admission/model payload read.
  return p.read(Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/full-cache-shared-authentic-case-v2.json'))
 def test_entire_authenticated_phase_roster_and_preregistered_prime(self):
  case=self.case();s=p.schedules(case);self.assertEqual([r['name'] for r in s],list(p.PHASES));self.assertEqual(s[1]['expected_reused'],[0]);self.assertEqual(s[2]['expected_reused'],[272]);self.assertEqual(s[1]['rows'],s[2]['rows']);self.assertEqual(s[1]['max_new'],[1]);self.assertEqual(s[-1]['policy'],{'strata_fresh':True})
 def test_all_bounded_scenarios_under_actual_static64MiB(self):
  for name in ('shared','independent','cancellation','eviction'):
   s=p.scenario_schedule(self.case(),name);self.assertLessEqual(s['maximum_source_observer_bytes'],64<<20);self.assertFalse(s['shared_state_transferred_between_actors']);self.assertEqual(s['all_other_fullscope_requirements_still_mandatory'],list(p.MANDATORY))
  ev=p.scenario_schedule(self.case(),'eviction');self.assertEqual([r['max_new'] for r in ev['phases'] if r['name'].startswith('eviction')],[[1]]*6)
 def test_whole_armed_roster_static_bound_not_disk_quota(self):
  source=(p.fixture.SDK/'source/sycl/include/strata/core/batch_fidelity_contract.hpp').read_text();self.assertIn('BYTE_LIMIT=64*1024*1024',source);self.assertGreater(28*p.RAW49_BYTES,p.ACTOR_BYTE_LIMIT)
 def test_bad_shared_boundary_or_phase_input_refused(self):
  c=self.case();c['shared_boundary']['tokens']=271
  with self.assertRaises(ValueError):p.schedules(c)
  with self.assertRaises(ValueError):p.scenario_schedule(self.case(),'unregistered_subset')
 def test_fresh_and_pin_types_presence_contract(self):
  self.assertEqual(client.validate_policy({'strata_fresh':False,'strata_shared_prefix':{'tokens':272}})['strata_shared_prefix']['tokens'],272)
  for body in ({'strata_fresh':1},{'strata_fresh':False,'strata_shared_prefix':{'tokens':True}},{'strata_fresh':True,'strata_shared_prefix':{'tokens':0}},{'strata_fresh':False,'strata_shared_prefix':{'tokens':272,'extra':0}}):
   with self.assertRaises(ValueError):client.validate_policy(body)
 def test_all_full_requirements_cannot_pass_by_boolean_declaration(self):
  state=p.full_scope_status({name:True for name in p.MANDATORY});self.assertFalse(state['full_cache_runtime_qualified']);self.assertIn('exact_checkpoint_victim',state['mandatory_requirements']);self.assertIn('diskcache_wrong_fingerprint_batch0',state['mandatory_requirements']);self.assertIn('physical_host_device_expert_residency',state['mandatory_requirements'])
 def test_source_client_precontent_close_and_unique_phase_requestids(self):
  source=Path(client.__file__).read_text();ast.parse(source);self.assertIn('sock.shutdown(socket.SHUT_RDWR)',source);self.assertIn('out.parent.name',source);self.assertIn('not 1<=v<=64',source);self.assertIn("intentional_cancel_transport_exception",source)
 def test_real_phase_marker_original_line_bounds_and_uniqueness(self):
  trace='ORIGINAL_LINE0\nHARNESS FULLCACHE_PHASE index=1 name=prime0\nCPU_LINE2\nHARNESS FULLCACHE_PHASE index=2 name=prime1\nCPU_LINE4\n';selected,bounds=run.phase_trace(trace,'prime0',1);self.assertEqual(selected,'CPU_LINE2');self.assertEqual(bounds['original_first_line'],3);self.assertEqual(bounds['original_last_line'],3)
  with self.assertRaises(ValueError):run.phase_trace(trace+'HARNESS FULLCACHE_PHASE index=1 name=prime0\n','prime0',1)
 def test_producer_uses_actual_sink_sequence_and_active_call_guard(self):
  source=Path(__file__).with_name('full_cache_shared_api_trace_v1.py').read_text();ast.parse(source);self.assertIn("semantic=True)['sequence']",source);self.assertIn('if active_calls:raise',source);self.assertIn('fullcache_phase_errors',source);self.assertIn('active_calls.discard(call)',source)
class RawControls(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='CPU-fullcache-raw49-');self.root=Path(self.temp.name);self.job={'rid':1,'role':'admission','call':1,'ids':[10,11],'position':1,'token':11};self.work=[{'rid':1,'call':1,'pid':77,'slotgen':1,'input_ids':[10,11],'actual_reused':0}];self.lines=['SBF request pid=77 rid=1 enginegen=1 requestgen=1 slot=0 slotgen=1 prompt=2','SBF resume pid=77 rid=1 enginegen=1 requestgen=1 slot=0 slotgen=1 reused=0 read_from=0 reread_to=-1','SBF span pid=77 rid=1 enginegen=1 requestgen=1 slot=0 slotgen=1 stage=0 lb=0 le=48 begin=0 end=2 phase=admission_target_verify completed=1','SBF replay pid=77 stage=0 epoch=1 lb=0 le=48 graph=0 enginegen=1 event=0 rows=1 admission=1 active_mask=1 selected_mask=1 full_roster=1 input_output_verified=1 stage_context_verified=1 geometry=48x4x2560x248320','SBF row pid=77 rid=1 stage=0 epoch=1 lb=0 le=48 graph=0 enginegen=1 requestgen=1 slot=0 slotgen=1 pos=1 token=11 row=0 batchrows=1 event=0 admission=1 selected=1']
  for layer in range(-1,48):
   count=248320 if layer==-1 else 10240;path=self.root/('CPU_FIELD_'+str(layer)+'.f32');path.write_bytes(b'\0'*(count*4));phase='admission_logits_before_sampler' if layer==-1 else 'admission_residual';self.lines.append('SBF vector pid=77 rid=1 stage=0 epoch=1 lb=0 le=48 graph=0 enginegen=1 requestgen=1 slot=0 slotgen=1 pos=1 token=11 row=0 batchrows=1 event=0 layer='+str(layer)+' phase='+phase+' floats='+str(count)+' bytes='+str(count*4)+' canonical=le_f32 file=/results/captures/'+path.name)
 def tearDown(self):self.temp.cleanup()
 def collect(self,lines=None):
  with patch.object(raw,'prefix_jobs',return_value={'jobs':[self.job]}):return raw.collect('\n'.join(lines or self.lines),[],self.work,[(0,0,48)],self.root,{1:['admission']})
 def test_real_complete49_raw_extent_and_replay_positive(self):self.assertEqual(self.collect()['actual_raw_fields'],49)
 def test_wrong_request_replay_mask_stage_or_missing_layer_refused(self):
  for key in ('request','mask','layer','stage','span'):
   lines=list(self.lines)
   if key=='request':lines[-1]=lines[-1].replace('requestgen=1','requestgen=2')
   if key=='mask':lines[3]=lines[3].replace('active_mask=1','active_mask=3')
   if key=='layer':lines.pop()
   if key=='stage':lines[-1]=lines[-1].replace('stage=0','stage=1')
   if key=='span':lines[2]=lines[2].replace('end=2','end=1')
   with self.subTest(key=key),self.assertRaises(ValueError):self.collect(lines)
 def test_truncated_currentraw_or_foreign_alias_refused(self):
  path=self.root/'CPU_FIELD_0.f32';path.write_bytes(b'\0'*4)
  with self.assertRaises(ValueError):self.collect()
 def test_actual_comparison_requires_complete_ids_and_current_sha(self):
  group=self.collect()['groups'][0];self.assertTrue(raw.compare49(group,group)['all49_bitwise_equal']);changed=copy.deepcopy(group);changed['input_ids']=[10,12]
  with self.assertRaises(ValueError):raw.compare49(group,changed)
  Path(group['vectors']['0']['path']).write_bytes(b'\0'*4)
  with self.assertRaises(ValueError):raw.compare49(group,group)
if __name__=='__main__':unittest.main()
