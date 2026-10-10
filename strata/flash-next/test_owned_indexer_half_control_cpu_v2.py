"""Source-only producer-shaped fixture/raw/lifecycle refusal controls."""
import tempfile,unittest,copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import owned_indexer_half_control_v2 as c
import qualify_owned_indexer_half_control_v2 as q
class Controls(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.base=Path(self.t.name);self.root=self.base/'build/native-output';self.root.mkdir(parents=True);self.fixture=self.base/'fixture';self.fixture.mkdir();self.raw=np.linspace(-2,2,512,dtype='<f4').tobytes();(self.fixture/'raw.f32').write_bytes(self.raw);(self.root/'consumed-raw.f32').write_bytes(self.raw);self.prior={'CPU':'original'};self.binding={'root':str(self.fixture),'record':{'own_producer_binding':self.prior}}
  expected=c.proposal.preregistered(self.raw)
  for r in range(3):
   out=self.root/('route'+str(r));out.mkdir()
   for name,value in [('expression.f32',self.raw),('materialized.f32',expected['FP16_RNE_widened_F32']),('materialized.f16',expected['FP16_RNE']),('input.f32',self.raw)]: (out/name).write_bytes(value)
  self.log='HALF37_DEVICE backend=level_zero affinity=0 fp16=1 vendor=CPU driver=synthetic name=fixture\nHALF37_FP_CONFIG flags=1,2, observed_device_flags_only=1 compiler_lowering_unobserved=1\n'+''.join('HALF37_FRAME route=%d graph_replay=%d fields=4 own_input_restored=1 values=512\n'%(r,r) for r in range(3))+'HALF37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 compiler_lowering_qualified=0 full_model_math_qualified=0\n';self.logpath=self.base/'runtime.log';self.logpath.write_text(self.log)
 def compare(self):return c.compare_routes(self.root,self.binding,self.prior)
 def test_positive_full_raw_and_materialized_actual_store_load_join(self):
  r=self.compare();self.assertTrue(r['actual_materialized_F16_to_F32_join_verified']);self.assertTrue(r['fields']['expression.f32']['identity_F32_match']);self.assertFalse(r['candidate_reference_modified']);self.assertIsNone(r['tolerance_gate'])
 def test_materialized_half_vs_F32_mismatch_refused_even_when_all_replays_agree(self):
  for r in range(3):(self.root/('route'+str(r))/'materialized.f32').write_bytes(self.raw)
  with self.assertRaises(ValueError):self.compare()
 def test_extra_route_file_refused(self):
  (self.root/'route0/foreign.f32').write_bytes(b'')
  with self.assertRaises(ValueError):self.compare()
 def test_extra_root_file_refused(self):
  (self.root/'foreign').write_bytes(b'')
  with self.assertRaises(ValueError):self.compare()
 def test_symlink_even_same_raw_refused(self):
  p=self.root/'route0/input.f32';p.unlink();p.symlink_to(self.fixture/'raw.f32')
  with self.assertRaises(ValueError):self.compare()
 def test_missing_duplicate_truncated_or_extra_marker_refused(self):
  for log in (self.log.rstrip('\n'),self.log+self.log.splitlines()[2]+'\n',self.log+'HALF37_EXTRA\n',self.log.replace('HALF37_FP_CONFIG','OTHER')):
   self.logpath.write_text(log)
   with self.assertRaises(ValueError):self.compare()
 def test_raw_read_mutation_rejected(self):
  p=self.fixture/'raw.f32';original=Path.read_bytes;calls=[0]
  def mutate(path):
   b=original(path)
   if path==p:
    calls[0]+=1
    if calls[0]==2:return bytes(len(b))
   return b
  with patch.object(Path,'read_bytes',mutate):
   with self.assertRaises(ValueError):c.consume(p,2048)
 def test_foreign_producer_and_raw_echo_rejected(self):
  with self.assertRaises(ValueError):c.compare_routes(self.root,self.binding,{'CPU':'foreign'})
  (self.root/'consumed-raw.f32').write_bytes(bytes(2048))
  with self.assertRaises(ValueError):self.compare()
 def test_actual_typed_original_command_receipt_epoch_rc_error_mutations_refused(self):
  p=self.base/'x.log';p.write_text('ok\n');cmd=self.base/'x.command.json';q.write(cmd,['CPU']);receipt=self.base/'x.receipt.json';row={'phase_cleanup_error':None,'passed':True,'reader_retired':True,'eof':True,'error':None,'command_error':None,'return_code':0,'path':str(p),'sha256':q.sha(p),'command':['CPU'],'command_sha256':q.sha(cmd),'finished_epoch':11};q.write(receipt,row);q.command_binding(row,p,cmd,receipt)
  for key,val in [('finished_epoch',12),('return_code',1),('error','bad'),('phase_cleanup_error','bad')]:
   changed=dict(row);changed[key]=val
   with self.assertRaises(ValueError):q.command_binding(changed,p,cmd,receipt)
 def test_wrong_fixture_source_refused_before_original_source_admission(self):
  q.write(self.fixture/'input-binding.json',{'schema':1})
  with patch.object(c.own,'fixture_binding') as forbidden:
   with self.assertRaises(ValueError):c.fixture_binding(self.fixture)
   forbidden.assert_not_called()
 def test_producer_shaped_new_fixture_roundtrip_and_mutation(self):
  original=self.base/'original';original.mkdir();(original/'raw.f32').write_bytes(self.raw);origin={'root':str(original),'fixture_sha256':'CPU','record':{'own_producer_root':'CPU_PRODUCER','own_producer_binding':self.prior}};plan=self.base/'plan.json';plan.write_text('{"CPU":true}');out=self.base/'newfixture'
  with patch.object(c.own,'fixture_binding',return_value=origin),patch.object(q,'source_binding',return_value={'CPU':'source'}),patch.object(q,'PLAN',plan):
   c.prepare(original,out);r=c.fixture_binding(out);self.assertEqual(r['record']['raw_sha256'],c.sha(self.raw));(out/'raw.f32').write_bytes(bytes(2048))
   with self.assertRaises(ValueError):c.fixture_binding(out)
 def test_new_source_closure_and_separate_leaf_compile_flags(self):
  r=q.source_binding();self.assertIn('/leaf/owned_indexer_half_discrimination_gpu_v2.cpp',r['actual_half_leaf_argv']);self.assertIn('-fp-model=precise',r['actual_half_leaf_argv']);self.assertEqual(r['actual_half_leaf_argv'][-1],'/out/owned-indexer-half37')
 def test_post_full4_cannot_precede_actual_runtimeEOF(self):
  r={'run_command':{'finished_epoch':100},'GPU_terminal_epoch':90,'post_health':{'finished_epoch':110},'post_journal':{'started_command_epoch':111,'finished_epoch':112},'post_full4':{'started':113,'finished':114},'before_post_full4_pages':{'epoch':112.5},'post_pages':{'epoch':115},'finished_epoch':116}
  with self.assertRaises(ValueError):q.chronology(r)
if __name__=='__main__':unittest.main()
