import unittest,struct,tempfile
from pathlib import Path
import run_original_mtp53_expert_pair_model_v2 as r
class OriginalProfile(unittest.TestCase):
 def test_actual_HH_and_inverse_rank_schema(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'profile.bin';pairs=[(l,e)for l in range(48)for e in range(512)][::-1];inverse=[0]*24576
   for i,(l,e)in enumerate(pairs):inverse[l*512+e]=i
   prefix=b''.join(struct.pack('<2H',*v)for v in pairs);p.write_bytes(b'STRP'+struct.pack('<5I',1,48,512,24576,24576)+prefix+struct.pack('<24576i',*inverse))
   text='token graph hit path: 7270 resident experts\nstage mirror segment: layers0-47 index0 bytes300 pointer0x12\nstage mirror: layers0-47 missing17306 mirrored17306 bytes300'
   actual=r.residency(text,p);self.assertEqual(actual['profile_prefix_sha256'],r.hashlib.sha256(prefix[:7270*4]).hexdigest())
   bad=bytearray(p.read_bytes());bad[-4:]=struct.pack('<i',1);p.write_bytes(bad)
   with self.assertRaises(ValueError):r.residency(text,p)
 def test_explicit_original_compiled_plan_association(self):
  s=Path(r.__file__).read_text();self.assertIn("recipe.get('pair_model_source_plan_sha256')!=plan['compiled_model_source_plan_sha256']",s)
  self.assertIn("struct.iter_unpack('<2H',raw[24:24+ranked*4])",s)
  self.assertIn("raw[24:24+count*4]",s)
