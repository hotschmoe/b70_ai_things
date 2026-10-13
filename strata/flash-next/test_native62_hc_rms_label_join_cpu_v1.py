import copy,json,unittest
from pathlib import Path
import native62_hc_rms_label_join_v1 as j
ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
class Controls(unittest.TestCase):
 def fixture(self):
  root=ROOT/'native62-hc-rms-ON-v2-run-v1';report=json.loads((root/'report.json').read_text());off=json.loads((root/'native62-hc-rms-ON/OFF-component.snapshot.json').read_text());right=json.loads((root/'native62-hc-rms-ON/recipe.json').read_text());proposal=json.loads(Path('strata/flash-next/native62-HC-perf-explicit-sqrt-engine-proposal-v1.json').read_text());return [off['actual_phase_recipe'],right,off['original_actor_source_plan_sha256'],report['source_plan_sha256'],Path(off['source_root'])/'native62-hc-rms-OFF/native.stdin.jsonl',root/'native62-hc-rms-ON/native.stdin.jsonl',Path(report['SDK_root']),proposal]
 def test_authentic_five_labels(self):
  proof=j.matched(*self.fixture());self.assertEqual(set(proof['named_producer_labels']),set(j.LABELS));self.assertTrue(proof['artifact_identity_disk_session_consumer_inactive']);self.assertEqual(proof['math_delta_only'],'STRATA_HC_EXPLICIT_SQRT_DIVIDE0/1')
 def test_each_wrong_actual_label_refused(self):
  for k in j.LABELS:
   args=self.fixture();args[1]=copy.deepcopy(args[1]);args[1]['native_env'][k]='0'*64
   with self.assertRaises(ValueError):j.matched(*args)
 def test_math_or_cache_change_not_ignored(self):
  for key in('STRATA_QFUSE','STRATA_HC_PREFILL_TILE4'):
   args=self.fixture();args[1]=copy.deepcopy(args[1]);args[1]['native_env'][key]='foreign'
   with self.assertRaises(ValueError):j.matched(*args)
  args=self.fixture();args[1]=copy.deepcopy(args[1]);a=args[1]['native_argv'];a[a.index('--prompt-cache')+1]='1'
  with self.assertRaises(ValueError):j.matched(*args)
if __name__=='__main__':unittest.main()
