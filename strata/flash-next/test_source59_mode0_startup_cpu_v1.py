import tempfile,unittest
from pathlib import Path
import source59_trace_pair_startup_v1 as new
import run_source43_pair24_latency_v2 as original
import run_source59_mode0_fidelity_v3 as actor
from serial37_canonical_json_v3 import read_unique
F17=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
class Startup(unittest.TestCase):
 def test_actual_trace_and_original_negative(self):
  root=F17/'source59-mode0-fidelity-v2-run-v1';source=root/'source59-mode0-fidelity/startup/server-engine.log';before=source.stat();data=source.read_bytes();self.assertGreater(len(data),16<<20);self.assertLess(len(data),128<<20);self.assertIn(b'READY 4096 stop\n',data)
  metadata=read_unique(root/'prepared.snapshot.json')
  with tempfile.TemporaryDirectory()as tmp:
   directory=Path(tmp);(directory/'server-engine.log').write_bytes(data)
   with self.assertRaisesRegex(ValueError,'Original bounded pair startup'):original.memory_admission(directory,metadata)
   result=new.memory_admission(directory,metadata)
   self.assertEqual([r['resident_experts']for r in result['stages']],[9101,8866]);self.assertEqual([r['mirror_experts']for r in result['stages']],[3187,3422]);self.assertTrue(result['both_GPU_constructor_ranges_observed']);self.assertFalse(result['both_device_headroom_qualified'])
   malformed=data.replace(b'CUDA1 runs layers 24-47, expert cache 8866 slots',b'CUDA1 bad layers 24-47, expert cache 8866 slots',1);self.assertNotEqual(malformed,data);(directory/'server-engine.log').write_bytes(malformed)
   with self.assertRaisesRegex(ValueError,'secondaryGPU16-layer cache missing'):new.memory_admission(directory,metadata)
  self.assertEqual(before,source.stat())
 def test_current_producer_public_callback_and_bound(self):
  self.assertIs(actor.NS['life'].prepared_ctrl.memory_admission,new.memory_admission)
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp)/'server-engine.log'
   with p.open('wb')as f:f.truncate((128<<20)+1)
   with self.assertRaisesRegex(ValueError,'Original bounded pair startup'):new.memory_admission(Path(tmp),{})
