"""Matched settings/source ownership/startup shape, tiny synthetic only."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source43_pair_topology_v1 as q
import run_source43_pair_model_cli_v1 as a
import validate_source43_pair_model_cli_v1 as r
from test_source43_normal_model_cpu_v1 import Controls as Shapes
class Controls(unittest.TestCase):
 def fixture(self):
  m,cfg,inputs=Shapes().fixture();m['port']=18082;return m,cfg,inputs
 def test_exact_topology_only_delta_with_matched_model_settings(self):
  m,cfg,inputs=self.fixture();one,x=q.recipe(q.PHASES[0],m,cfg,inputs,'a'*64);pair,y=q.recipe(q.PHASES[1],m,cfg,inputs,'a'*64);self.assertEqual(x['requests'],y['requests']);self.assertEqual(pair['cards'],[0,1]);self.assertEqual(one['cards'],[0]);self.assertEqual(y['native_argv'][:-5],x['native_argv']);self.assertEqual(y['native_argv'][-5:],['--layer-split','32','--split-device','1','--trim-stage-weights']);xe=dict(x['native_env']);ye=dict(y['native_env']);self.assertEqual(xe.pop('ZE_AFFINITY_MASK'),'0');self.assertEqual(ye.pop('ZE_AFFINITY_MASK'),'0,1');self.assertEqual(xe,ye);self.assertEqual(y['stages'],[[0,0,32],[1,32,48]]);self.assertEqual(x['native_argv'][x['native_argv'].index('--max-context')+1],'2048')
 def test_inherited_parser_only_two_named_startup_call_changes(self):
  self.assertIs(a._PORTS['PHASES'],q.PHASES);self.assertIs(r.ports()['a'],a);self.assertIs(a._PORTS['intake'],a.base.intake);self.assertIn('memory_admission',a._PORTS['run_phase'].__code__.co_names)
 def startup(self,kind):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);text='strata generate: expert cache 8000 slots,\nstrata generate: layer split: CUDA1 runs layers 32-47, expert cache 8192 slots (25 GiB), 8192 of its 8192 profiled pairs; slot 0 verified\nstrata generate: layer split: CUDA0 runs layers 0-31\nstage mirror: layers0-31 missing8384 mirrored8384 bytes1000 global_used1000 remaining5000\nstage mirror: layers32-47 missing0 mirrored0 bytes0 global_used1000 remaining5000\nstrata serve: 1800 MiB of VRAM free with everything loaded\nthe prompt path allocates its own buffers\n'
   if kind=='missingGPU':text=text.replace('CUDA1','CUDA0')
   elif kind=='mirror':text=text.replace('layers32-47','layers31-47')
   elif kind=='fallback':text+='trying a smaller expert cache\n'
   (root/'server-engine.log').write_text(text);plan={'cards':[0,1],'stages':[[0,0,32],[1,32,48]]}
   if kind:
    with self.assertRaises(ValueError):q.memory_admission(root,{},plan)
   else:
    value=q.memory_admission(root,{},plan);self.assertTrue(value['both_GPU_constructor_ranges_observed']);self.assertFalse(value['both_device_headroom_qualified']);self.assertEqual(value['stages'][1]['mirror_bytes'],0)
 def test_actualshaped_pair_zero_missing_secondstage_valid(self):self.startup(None)
 def test_missing_real_secondGPU_constructor_refused(self):self.startup('missingGPU')
 def test_overlapping_stage_mirror_refused(self):self.startup('mirror')
 def test_capacityfallback_refused(self):self.startup('fallback')
 def test_numeric_requires_real_both_stage_returns(self):
  import serial_prefix_qualification_v6 as old
  events=[{'event':'begin','pid':1,'request':1,'tokens':2,'input_sha256_le32':old.le32_digest([7,8])}]
  for device,lo,hi in[(0,0,32),(1,32,48)]:
   for complete in(False,True):events.append({'event':'stage_span','pid':1,'request':1,'phase':'prefill','device':device,'lb':lo,'le':hi,'lo':0,'hi':2,'complete':complete})
  raw={'ids':[7,8],'fresh':1,'pin':None,'cancel_requested':None,'stop_sent':False,'stderr':['PCL '+json.dumps(e)for e in events]}
  with patch.object(old,'extract_numeric',return_value={'actual':'CPU_SYNTHETIC'})as extract:
   self.assertEqual(q.numeric(raw,Path('/CPU_SYNTHETIC'),[[0,0,32],[1,32,48]])['actual'],'CPU_SYNTHETIC');self.assertEqual(extract.call_args.args[-1],[(0,32),(32,48)])
   bad=copy.deepcopy(raw);bad['stderr']=bad['stderr'][:-1]
   with self.assertRaises(ValueError):q.numeric(bad,Path('/CPU_SYNTHETIC'),[[0,0,32],[1,32,48]])
if __name__=='__main__':unittest.main()
