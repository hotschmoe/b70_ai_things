import ast,unittest
from pathlib import Path
import full_cache_shared_batch0_identity_v2 as purpose

class Batch0(unittest.TestCase):
 def test_actual_source_body_roundtrip_and_route_preserved(self):
  path=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T122458Z-_0gr_xk8/source/serve/server.py');original=purpose.method_source(path);new=purpose.adapt(original);self.assertEqual(purpose.undo(new),original);ast.parse('if True:\n'+new)
  self.assertIn('yield from self.generate_batched',new);self.assertEqual(new.count('self._begin_request_identity()'),2);self.assertIn('self._request_keys(sampling or {})',new)
 def test_real_serial_branch_emits_actual_protocol_rid_without_token_mutation(self):
  original="""    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.progress, self.progress_ms, self.reused = None, 0, 0
        head = f"GENI {int(max_new)}{self.sampling_keys(sampling or {})} {embeddings}" if embeddings else f"GEN {int(max_new)}{self.sampling_keys(sampling or {})}"
        yield head+' '+','.join(str(t) for t in ids)
"""
  def validate(sampling,embeddings):
   if embeddings is not None:raise ValueError('embedding forbidden')
  namespace={'strict_batch_request':validate};exec('if True:\n'+purpose.adapt(original),namespace)
  class Engine:
   strict_batch_identity=True
   def _begin_request_identity(self):self.rid=7
   def _request_keys(self,sampling):return ' temp=0 fresh=1 rid='+str(self.rid)
  engine=Engine();rows=list(namespace['generate'](engine,[11,12],1,{},None));self.assertEqual(rows,['GEN 1 temp=0 fresh=1 rid=7 11,12']);engine.strict_batch_identity=False
  with self.assertRaises(ValueError):list(namespace['generate'](engine,[11,12],1,{},None))
 def test_changed_body_header_rejected(self):
  with self.assertRaises(ValueError):purpose.adapt('unrelated source')
if __name__=='__main__':unittest.main()
