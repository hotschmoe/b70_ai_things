import unittest,json
from pathlib import Path
import original_mtp53_reference_pp_policy_v1 as p
class Controls(unittest.TestCase):
 def test_actual_reference_and_other_prompt_boundaries(self):
  policy=p.policy();recipe=json.loads(p.REFERENCE.read_bytes());prompt=recipe['requests'][0]['ids'];self.assertEqual(p.cuts(prompt,policy),[24,1536,2041,2047])
  shapes=[];begin=0
  for end in p.cuts(prompt,policy):
   n=end-begin
   while n:take=min(n,512);shapes.append(take);n-=take
   begin=end
  self.assertEqual(shapes,[24,512,512,488,505,6])
  other=[7]*2048;other[0]=248045;other[9]=248045;other[2000]=248045
  self.assertEqual(p.cuts(other,policy),[9,1536,2000,2047])
 def test_only_prefix_call_changed(self):
  h=Path(__file__).resolve().parent/'original-mtp53';old=(h/'paired_greedy_v1.hpp').read_text();new=(h/'paired_greedy_v2.hpp').read_text();start=new.index('std::string error;const auto cuts=');end=new.index('\n int64_t position=',start)
  start0=old.index('std::string error;const bool prompt_ok=');end0=old.index('\n int64_t position=',start0)
  restored=new[:start]+old[start0:end0]+new[end:];restored=restored.replace('\n#include "strata/kernels/original_mtp53_reference_pp_policy.hpp"','')
  self.assertEqual(restored,old)
if __name__=='__main__':unittest.main()
