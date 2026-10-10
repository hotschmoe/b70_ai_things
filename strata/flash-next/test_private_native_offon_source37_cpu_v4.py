"""Actual immutable OFF/ON metadata epoch regression; mocked fresh model reads."""
import copy,json,math,unittest
from pathlib import Path
from unittest.mock import patch
from test_private_native_offon_source37_cpu_v1 import Protocol,Serial
from test_private_native_offon_source37_cpu_v2 import MixedSerial
import validate_private_native_offon_source37_v4 as reader
BASE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')

class Epoch(unittest.TestCase):
 def fixture(self):
  roots=[BASE/'batch40-pair-native2-off-run-v1',BASE/'batch40-pair-native2-on-run-v1'];plans=[json.loads((r/'input-plan.snapshot.json').read_bytes()) for r in roots];parents=[json.loads((r/'parent-qualification.json').read_bytes()) for r in roots];return plans,parents
 def match(self,plans,parents):
  def current(path,lock,shards):
   x=copy.deepcopy(plans[0]['model_identity']);x['current_known_pages']['epoch']=parents[1]['started_epoch']+100;return x
  with patch.object(reader.ctrl,'verify_model_identity',side_effect=current),patch.object(reader.ctrl.api,'experimental_alias',side_effect=lambda cfg,n,d,lane:plans[int(d)]['research_alias']),patch.object(reader.ctrl.api,'registry_gate',side_effect=lambda alias:next(p['registry_binding'] for p in plans if p['research_alias']==alias)):
   return reader.matched_plans(plans[0],plans[1],parents[0],parents[1])
 def test_actual_exact_four_diff_snapshot_accepts_only_proven_epoch(self):
  p,parents=self.fixture();self.assertNotEqual(p[0]['model_identity']['current_known_pages']['epoch'],p[1]['model_identity']['current_known_pages']['epoch']);x=self.match(p,parents);self.assertFalse(x['OFF']['original_epoch_recomputed_from_current_clock']);self.assertEqual(x['ON']['original_observed_page_read_epoch'],p[1]['model_identity']['current_known_pages']['epoch'])
 def test_future_or_nonfinite_or_boolean_original_epoch_rejects(self):
  for value in (float('nan'),float('inf'),True,0):
   p,parents=self.fixture();p[1]['model_identity']['current_known_pages']['epoch']=value
   with self.assertRaises(ValueError):self.match(p,parents)
  p,parents=self.fixture();p[1]['model_identity']['current_known_pages']['epoch']=parents[1]['started_epoch']+1
  with self.assertRaises(ValueError):self.match(p,parents)
 def test_second_epoch_or_other_model_field_cannot_be_ignored(self):
  for kind in ('digest','stat','path','scope','revision'):
   p,parents=self.fixture()
   if kind=='digest':p[1]['model_identity']['current_known_pages']['rows'][0]['sha256']='0'*64
   elif kind=='stat':p[1]['model_identity']['current_known_pages']['stat_after'][2]+=1
   elif kind=='path':p[1]['model_identity']['current_known_pages']['path']='/wrong'
   elif kind=='scope':p[1]['model_identity']['scope']='changed'
   else:p[1]['model_identity']['model_revision']='changed'
   with self.assertRaises(ValueError):self.match(p,parents)
 def test_eager_corpus_sampler_or_sdk_drift_rejects(self):
  for kind in ('env','tokens','SDK'):
   p,parents=self.fixture()
   if kind=='env':p[1]['env']['STRATA_VERIFY_EAGER']='0'
   elif kind=='tokens':p[1]['tokens']['target'][0][0]+=1
   else:p[1]['engine_receipt_sha256']='0'*64
   with self.assertRaises(ValueError):self.match(p,parents)
 def test_actual_identity_finished_before_original_page_epoch(self):
  p,parents=self.fixture();record=reader.read(p[1]['model_identity']['path']);p[1]['model_identity']['current_known_pages']['epoch']=record['finished']-1
  with self.assertRaises(ValueError):self.match(p,parents)
 def test_canonical_v4_roster_new_and_actual_parent_argument_required(self):
  self.assertTrue(str(reader.serial_roster_path(Path('/tmp/ON'))).endswith('.serial49-roster-v4.json'))
  with self.assertRaises(TypeError):reader.matched_plans({}, {})
if __name__=='__main__':unittest.main()
