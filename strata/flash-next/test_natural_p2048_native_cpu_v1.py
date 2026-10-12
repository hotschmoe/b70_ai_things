"""Two targeted construction/early-EOS checks, no model/GPU execution."""
import unittest
from unittest.mock import patch
import run_natural_p2048_native_v1 as a
import validate_natural_p2048_native_v1 as reader
class Tests(unittest.TestCase):
 def test_six_requests_one_process_recipe(self):
  cases=[{'id':str(i),'ids':[10]*2048}for i in range(3)]
  scope={'generation':51,'prefill':512,'cards':2,'esimd':0,'mirror':1,'corpus':'CPU_FIXTURE','corpus_sha256':'x'}
  a.SETTINGS=scope
  original=a.BASE_PHASE
  def render(*args):return {'runtime':{'image':'CPU'},'cards':[0]}, {'native_argv':['/build/strata','--serve','--prefill','64','--max-context','2048','--spec','2'],'native_env':{},'PTI_enabled':False}
  from pathlib import Path
  from types import SimpleNamespace
  with patch.object(a,'render_template',side_effect=render),patch.object(a.c,'read',return_value={'cases':cases}),patch.object(a,'fresh',SimpleNamespace(SDK=Path('/CPU'))):
   m,r=a.phase_recipe(a.PHASES[0],{}, {}, {},'CPU')
  self.assertEqual(m['cards'],[0,1]);self.assertEqual(len(r['requests']),6);self.assertEqual([x['process_temperature']for x in r['requests']],['cold','warm','warm','warm','warm','warm']);self.assertTrue(all(x['fresh']==1 and x['max_new']==512 for x in r['requests']));self.assertIn('--stop-eos',r['native_argv']);self.assertIn('--layer-split',r['native_argv']);self.assertEqual(r['requested_prefill'],512)
 def test_early_EOS_is_retained_not_padded(self):
  raw={'done':'DONE 2 2048 10 3 stop','stdout':['T 42','T 248046','DONE 2 2048 10 3 stop'],'stdout_seconds':[.02,.03,.04],'output_ids':[42,248046]}
  with patch.object(a,'requests',return_value=[]):v=reader.observations([{'ordinal':1,'raw':raw}],{'requests':[{'role':'case','process_temperature':'cold','input_temperature':'first'}]})[0]
  self.assertEqual(v['useful_non_EOS_tokens'],1);self.assertFalse(v['natural_output_256_target_met']);self.assertEqual(v['finish'],'stop');self.assertEqual(v['root_client_TTFT_s'],.02)
if __name__=='__main__':unittest.main()
