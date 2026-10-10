"""ABBA schedule, live repeated gate/equality and byte-closure order with tiny mocks."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import benchmark_upload_logical_gate_v1 as b
class Controls(unittest.TestCase):
 def test_interleaved_ABBA_and_strict_bounds(self):
  self.assertEqual(b.schedule(2),list('ABBAABBA'))
  for value in (True,0,5):self.assertRaises(ValueError,b.schedule,value)
 def test_every_warm_call_runs_outer_gate_and_all_byte_boundaries(self):
  events=[];prepared={'upload_lifecycle':{'path':'receipt','oracle':'oracle'},'engine_receipt':'/synthetic/engine.json','pack_receipt':'pack'}
  def epoch(kind):return SimpleNamespace(seal_predevice=lambda **k:events.append(kind+'-seal'),finalize=lambda **k:(events.append(kind+'-post')or {'passed':True}))
  gate=lambda *a,**k:(events.append('live-gate')or {'exact':'original','ordinary_payload_readback_qualified':False})
  for arm in ('A','B'):
   events.clear()
   with patch.object(b,'sources'),patch.object(b,'read_unique',return_value=prepared),patch.object(b,'create_epoch',return_value=epoch('sdk')),patch.object(b,'pack_for_prepared',return_value=epoch('pack')),patch.object(b,'for_prepared',return_value=(epoch('logical'),[],{})),patch.object(b.old,'combined_generation_gate',return_value={'source37':True}),patch.object(b.old,'upload_gate',side_effect=gate),patch.object(b.new,'upload_gate',side_effect=gate):row=b.trial(Path('/synthetic'),arm,3)
   self.assertEqual(events.count('live-gate'),4);self.assertEqual(events[-6:],['logical-seal','pack-seal','sdk-seal','logical-post','pack-post','sdk-post']);self.assertEqual(len(row['timings']),4);self.assertTrue(row['timings'][0]['operation_first']);self.assertFalse(row['OS_page_cache_cold_claimed'])
 def test_changed_gate_return_fails_without_byte_or_result_reuse(self):
  prepared={'upload_lifecycle':{'path':'receipt','oracle':'oracle'},'engine_receipt':'/synthetic/engine.json','pack_receipt':'pack'};e=SimpleNamespace(seal_predevice=lambda **k:None,finalize=lambda **k:{'passed':True})
  with patch.object(b,'sources'),patch.object(b,'read_unique',return_value=prepared),patch.object(b,'create_epoch',return_value=e),patch.object(b,'pack_for_prepared',return_value=e),patch.object(b,'for_prepared',return_value=(e,[],{})),patch.object(b.old,'combined_generation_gate',return_value={}),patch.object(b.old,'upload_gate',side_effect=[{'value':True},{'value':1}]):self.assertRaises(ValueError,b.trial,Path('/synthetic'),'A',1)
if __name__=='__main__':unittest.main()
