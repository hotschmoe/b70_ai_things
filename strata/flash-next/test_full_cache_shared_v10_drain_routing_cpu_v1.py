"""Actual producer/reader routing, fixed deadline and full persisted scope."""
import ast,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class Routing(unittest.TestCase):
 def function(self,name,func):return next(n for n in ast.parse((HERE/name).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==func)
 def test_actual_HTTP_worker_scope_proof_ACK_join_and_freeze(self):
  n=self.function('run_full_cache_shared_runtime_v10.py','run_phase');source=ast.unparse(n);self.assertIn('with ContinuousDrain(stream, directory, write) as drain:',source);self.assertLess(source.index('with ContinuousDrain'),source.index('client = cohort'));self.assertLess(source.index("drain.require_sequence(proof['actual_event']['sequence'])"),source.index('api_adjudication = adjudicate'));self.assertLess(source.index('drain_receipt = drain.finish()'),source.index('stream.freeze()'));self.assertIn('time.monotonic() + 60',source)
 def test_persisted_all_six_phases_keep_continuous_source_and_original_actor(self):
  for func in ('request','session'):
   source=ast.unparse(self.function('run_full_cache_shared_batch0_persisted_v10.py',func));self.assertIn('with ContinuousDrain(stream, directory, ctrl.write) as drain:',source);self.assertIn("'continuous_drain': drain_receipt",source);self.assertLess(source.index('drain.finish()'),source.index('stream.freeze()'))
  source=(HERE/'validate_full_cache_shared_batch0_persisted_v10.py').read_text();self.assertIn('ack_binding(ack,markers[0])',source);self.assertNotIn("all(markers[0][k]==v for k,v in ack.items())",source);self.assertIn("['prime','after_wrong','after_valid']",source);self.assertIn("['save','wrong_restore','valid_restore']",source)
 def test_public_readers_bind_original_drain_receipt_source_ownership(self):
  for name in ('validate_full_cache_shared_runtime_v10.py','validate_full_cache_shared_batch0_persisted_v10.py'):
   source=(HERE/name).read_text();self.assertIn("'continuous-drain-receipt.json'",source);self.assertIn("actor-owner.json",source);self.assertIn('drain_binding(',source)
  source=(HERE/'validate_full_cache_shared_runtime_v10.py').read_text();self.assertIn("drain['last_progress']['sequence']>=proof['actual_event']['sequence']",source);self.assertIn('actual_terminals',source)
if __name__=='__main__':unittest.main()
