"""Two actual source controls, not numerical/native execution claims."""
import hashlib,json,subprocess,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'prefill-gdn-decoder-projections-source-plan-v1.json'
class Controls(unittest.TestCase):
 def test_apply_and_complete_reverse(self):
  p=json.loads(PLAN.read_bytes());sdk=Path(p['actual62_sdk'])/'source'
  with tempfile.TemporaryDirectory(prefix='b70-pf-gdn-proj-source-')as d:
   d=Path(d)
   for name,want in p['constructed_source_sha256'].items():
    raw=(sdk/name).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),p['actual_source_sha256'][name]);target=d/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
   patch=HERE/'patches/prefill-gdn-decoder-projections-v1.patch';subprocess.run(['git','apply',str(patch)],cwd=d,check=True,capture_output=True)
   for name,want in p['constructed_source_sha256'].items():self.assertEqual(hashlib.sha256((d/name).read_bytes()).hexdigest(),want)
   subprocess.run(['git','apply','--reverse',str(patch)],cwd=d,check=True,capture_output=True)
   for name in p['constructed_source_sha256']:self.assertEqual((d/name).read_bytes(),(sdk/name).read_bytes())
 def test_same_resolved_flag_sizes_and_carves(self):
  p=json.loads(PLAN.read_bytes());source=(Path(p['actual62_sdk'])/'source/sycl/src/prefill/prefill.cpp').read_text();patch=(HERE/'patches/prefill-gdn-decoder-projections-v1.patch').read_text();added=[line[1:]for line in patch.splitlines()if line.startswith('+')and not line.startswith('+++')]
  # Capacity is backed by the actual two modified arena boundaries, not a plan-only size.
  self.assertEqual(sum('m.gdn_decoder_q81 ='in x for x in added),1);self.assertEqual(sum('if(strata::kernels::pf_gdn_decoder_proj::enabled())o.take<uint8_t>'in x for x in added),1)
  self.assertEqual(p['scratch_logical_bytes'],8*(6144//32)*36)
  for statement in ('gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);','gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs,'):
   self.assertEqual(source.count(statement),1);self.assertFalse(any(statement in x for x in added))
if __name__=='__main__':unittest.main()
