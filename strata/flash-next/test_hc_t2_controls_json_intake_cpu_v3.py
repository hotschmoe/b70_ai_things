import copy,unittest
from pathlib import Path
import run_hc_t2_marked_paired_mtp_v3 as r
REPORT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/hc-t2-marked-paired-controls-v2/report.json')
SHA='66ef31547eea2532bbfa08895e667441260e70a8ccb6adf4e94e604a275acb01'
class Controls(unittest.TestCase):
 def test_actual_successful_controls_JSON_and_negative_count(self):
  self.assertEqual(r.sha(REPORT),SHA);saved=r.read(REPORT);self.assertTrue(saved['passed']);self.assertEqual(saved['source_plan_sha256'],r.OLD_CONTROL_PLAN_SHA);self.assertEqual(r.sha(r.OLD_CONTROL_PLAN),r.OLD_CONTROL_PLAN_SHA)
  directory=REPORT.parent/'controls';parsed=r.control_rows((directory/'native.stdout').read_text());parsed['residency']=r.residency((directory/'native.stderr').read_text());r.fresh_reset_marker((directory/'native.stderr').read_text());parsed['HC_T2_route']=r.route.counts((directory/'native.stderr').read_text(),parsed,'controls',1)
  original=saved['arm_results'][0]['actual'];self.assertNotEqual(parsed,original);self.assertEqual(r.controls_equal(parsed,original),original)
  bad=copy.deepcopy(original);bad['HC_T2_route']['actual_T2_windows_per_stage']['0']=2
  with self.assertRaises(ValueError):r.controls_equal(parsed,bad)
if __name__=='__main__':unittest.main()
