import copy,json,tempfile,unittest
from pathlib import Path
import native62_hc_rms_comparison_v2 as c
import recover_native62_hc_rms_OFF_v1 as r
class Controls(unittest.TestCase):
 def test_actual_bounded_consume_API_and_SHA(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'tiny';p.write_bytes(b'own original bytes');want=c.sha(p);self.assertEqual(c.consume(p,32,want),p.read_bytes())
   with self.assertRaises(ValueError):c.consume(p,32,'0'*64)
   with self.assertRaises(ValueError):c.consume(p,1,want)
 def test_exact_failed_status_remains_failed(self):
  p=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/native62-hc-rms-fidelity-v1-run-v1/report.json');report=json.loads(p.read_text());r.recovery_gate(report,c.sha(p),report['artifacts']);self.assertFalse(report['passed'])
  changed=copy.deepcopy(report);changed['passed']=True
  with self.assertRaises(ValueError):r.recovery_gate(changed,c.sha(p),report['artifacts'])
  changed=copy.deepcopy(report);changed['errors']=[]
  with self.assertRaises(ValueError):r.recovery_gate(changed,c.sha(p),report['artifacts'])
if __name__=='__main__':unittest.main()
