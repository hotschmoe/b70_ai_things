import json,unittest
from pathlib import Path
import source44_num38_numeric_v2 as old
import source59_literal1_numeric_lpdisabled_v1 as new
import recover_source59_mode0_fidelity_v3_v1 as r
ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source59-mode0-fidelity-v3-run-v1')
class Recovery(unittest.TestCase):
 def test_authentic_numeric_formatter_and_all_requests(self):
  d=ROOT/'source59-mode0-fidelity';rows=json.loads((d/'requests.json').read_bytes())
  for row in rows:
   raw=row['raw'];self.assertEqual(raw['ids'],[248045])
   with self.assertRaisesRegex(ValueError,'Actual exact fresh4'):old.port()[0](raw,d,[(0,24),(24,48)])
   current=new.numeric(raw,d,[(0,24),(24,48)]);self.assertEqual(len(current['residuals']),48);self.assertEqual(len(current['logits']),1);self.assertEqual(current['ledger']['generated'],1)
  first=dict(rows[0]['raw']);first['command']=first['command'].replace('temperature=0 ','temperature=0 logprobs=20 ')
  with self.assertRaisesRegex(ValueError,'Actual exact fresh4'):new.numeric(first,d,[(0,24),(24,48)])
 def test_authentic_failure_scope_and_foreign_status(self):
  value=r.c.read(ROOT/'report.json');r.recovery_gate(value,r.REPORT_SHA)
  for field,newvalue in [('passed',True),('errors',['different failure']),('results',{'invented':'pass'})]:
   current=dict(value);current[field]=newvalue
   with self.assertRaises(ValueError):r.recovery_gate(current,r.REPORT_SHA)
  with self.assertRaises(ValueError):r.recovery_gate(value,'0'*64)
