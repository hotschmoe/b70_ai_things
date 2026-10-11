import ast,sys,signal,unittest
from pathlib import Path
from unittest.mock import patch
import compare_original_first_hc_num38_v1 as a
class Controls(unittest.TestCase):
 def test_actual_parser_dispatch_without_provider(self):
  args=['compare','--actor-root','/tmp/closed44','--model-identity','/tmp/fresh-full4.json','--expected-identity-sha256','a'*64,'--output','/tmp/new-first-HC']
  signals={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGHUP,signal.SIGINT)}
  try:
   with patch.object(sys,'argv',args),patch.object(a,'execute')as call:a.main()
   value=call.call_args.args[0];self.assertEqual(value.actor_root,Path('/tmp/closed44'));self.assertEqual(value.expected_identity_sha256,'a'*64)
  finally:
   for sig,handler in signals.items():signal.signal(sig,handler)
 def test_no_historical_current_admission_or_full48_call(self):
  tree=ast.parse(Path(a.__file__).read_bytes());calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
  self.assertFalse(any('captured_binding'in n or 'finalized_binding'in n for n in calls));self.assertFalse(any(n.endswith('.tokens')for n in calls));self.assertIn('OriginalTensorRows',calls);self.assertIn('hc.hc_read_original',calls)
if __name__=='__main__':unittest.main()
