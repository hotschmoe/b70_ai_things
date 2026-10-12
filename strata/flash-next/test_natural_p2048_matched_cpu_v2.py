"""Small matching and actual stdlib profile-format controls; no model/GPU."""
import hashlib,struct,tempfile,unittest
from pathlib import Path
import run_natural_p2048_native_v2 as a
class Tests(unittest.TestCase):
 def test_complete_profile_and_duplicate_refused(self):
  pairs=[(l,e)for l in range(48)for e in range(512)];raw=b'STRP'+struct.pack('<5I',1,48,512,24576,24576)+b''.join(struct.pack('<HH',*p)for p in pairs)+struct.pack('<24576i',*range(24576))
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'p.bin';p.write_bytes(raw);self.assertEqual(a.profile_binding(p,hashlib.sha256(raw).hexdigest())['ranked_pairs'],24576);bad=raw[:28]+raw[24:28]+raw[32:];p.write_bytes(bad)
   with self.assertRaises(ValueError):a.profile_binding(p,hashlib.sha256(bad).hexdigest())
 def test_exact_matched_copy_delta_and_output_failure(self):
  l={'native_argv':['--spec','2'],'requests':[],'natural_settings':{},'matched_experiment':'copy','native_env':{'STRATA_COPY_ENGINE':'0','same':'yes'},'expert_profile_binding':{'sha256':'same'}};r={**l,'native_env':{'STRATA_COPY_ENGINE':'1','same':'yes'}};self.assertTrue(a.matched_recipe_gate(l,r));r['native_env']['same']='different'
  with self.assertRaises(ValueError):a.matched_recipe_gate(l,r)
  with self.assertRaises(ValueError):a.equivalence([0]*6,[1]*6)
if __name__=='__main__':unittest.main()
