import signal,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import compare_original_first_hc_bulk_v1 as a
class Controls(unittest.TestCase):
 def test_actual_argument_dispatch_no_provider(self):
  argv=['compare','--actor-root','/tmp/old44','--model-identity','/tmp/fresh.json','--expected-identity-sha256','a'*64,'--bulk-elf','/tmp/cpu.so','--expected-bulk-sha256',a.ELF_SHA,'--scalar-root','/tmp/scalar','--output','/tmp/new']
  old={s:signal.getsignal(s)for s in(signal.SIGTERM,signal.SIGHUP,signal.SIGINT)}
  try:
   with patch.object(sys,'argv',argv),patch.object(a,'execute')as execute:a.main()
   self.assertEqual(execute.call_args.args[0].bulk_elf,Path('/tmp/cpu.so'))
  finally:
   for s,handler in old.items():signal.signal(s,handler)
 def test_actual_byte_match_and_mismatch(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp);baseline={'owned.f32':b'\0'*4,'packet.q81':b'\1'*36}
   for name,raw in baseline.items():(p/name).write_bytes(raw)
   self.assertTrue(a.compare_scalar(p,baseline)['all_bitwise_equal']);(p/'packet.q81').write_bytes(b'\0'*36)
   with self.assertRaises(ValueError):a.compare_scalar(p,baseline)
if __name__=='__main__':unittest.main()
