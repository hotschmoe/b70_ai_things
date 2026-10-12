import unittest,struct,tempfile
from pathlib import Path
import run_original_mtp53_expert_pair_model_v1 as r
class MatchedPlacement(unittest.TestCase):
 def test_actual_counts_profile_prefix_and_segments(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'profile.bin';pairs=[(l,e)for l in range(48)for e in range(512)];p.write_bytes(b'STRP'+struct.pack('<5I',1,48,512,6500,24576)+b''.join(struct.pack('<2I',*v)for v in pairs))
   text='token graph hit path: 7270 resident experts\nstage mirror segment: layers0-47 index0 bytes100 pointer0x12\nstage mirror segment: layers0-47 index1 bytes200 pointer0x34\nstage mirror: layers0-47 missing17306 mirrored17306 bytes300'
   a=r.residency(text,p);b=r.residency(text.replace('0x12','0x55'),p);self.assertEqual(a,b)
   with self.assertRaises(ValueError):r.residency(text.replace('missing17306','missing17305'),p)
   with self.assertRaises(ValueError):r.residency(text.replace('index1','index2'),p)
 def test_both_mtp_one_kernel_delta_no_cache_policy_change(self):
  s=Path(r.__file__).read_text();self.assertIn("'STRATA_ORIGINAL_MTP53_GREEDY':'1','STRATA_EXPERT_PAIR_REUSE':str(mode)",s)
  self.assertIn("off['resolved_residency']!=on['resolved_residency']",s)
  self.assertNotIn('sized_slots',s);self.assertIn("fresh=recipe['fresh_operator_compile']",s)
