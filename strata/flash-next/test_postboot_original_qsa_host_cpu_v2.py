"""Consumer successor controls; no original artifacts or host helpers invoked."""
import ast,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent

class Controls(unittest.TestCase):
    def test_only_host_binding_and_recursive_reader_routes_change(self):
        names=('postboot_original_rms_dependencies','postboot_owned_hc_work',
               'postboot_owned_qsa_work','postboot_rms37_v7_historical_reader',
               'postboot_rms37_v8_historical_reader','postboot_owned_hc_v3_historical_reader',
               'postboot_owned_qsa_v4_historical_reader')
        for name in names:
            expected=(HERE/(name+'_v1.py')).read_text()
            for ref in names:expected=expected.replace(ref+'_v1',ref+'_v2')
            if name=='postboot_owned_hc_work':
                expected=expected.replace('import qualify_hc35_host_bulk_runtime_v1 as bulk',
                                          'import postboot_hc35_bulk_host_v1 as bulk')
                expected=expected.replace('bulk.finalized_binding(self.bulk_root)',
                                          "bulk.finalized_binding(self.bulk_root,self.model_association)['original_binding']")
            self.assertEqual((HERE/(name+'_v2.py')).read_text(),expected)
    def test_complete_host_scope_retained_and_old_current_gate_refused(self):
        source=(HERE/'admit_postboot_original_qsa_reference_v2.py').read_text()
        self.assertIn("host_proof['old_current_runtime_gate_passed'] is False",source)
        self.assertIn("'host_identity_association':host_proof",source)
        self.assertIn('host_admit(host_root,proof)',source)
    def test_model_computation_remains_refused(self):
        for name in ('postboot_owned_hc_work_v2','postboot_owned_qsa_work_v2'):
            tree=ast.parse((HERE/(name+'.py')).read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
            fun=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
            self.assertEqual(len(fun.body),1);self.assertIsInstance(fun.body[0],ast.Raise)

if __name__=='__main__':unittest.main()
