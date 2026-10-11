"""Currentboot consumer wiring; fixtures never authorize model/device work."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import native_qsa40_runtime_v9 as runtime

class Controls(unittest.TestCase):
    def test_own_admission_binds_exact_original_identity_and_current_native(self):
        with TemporaryDirectory() as d:
            root=Path(d);runtime.write(root/'report.json',{'original_work_config':{'model_identity':'/exact/original.json'}})
            with patch('admit_postboot_original_qsa_reference_v3.finalized_binding',return_value={'scope':'fixture'}) as admit:
                result=runtime.own_binding(root,'/exact/current_c140','/exact/current.json','a'*64)
            admit.assert_called_once_with(root,'/exact/original.json','/exact/current.json','a'*64,'/exact/current_c140')
            self.assertEqual(result,{'scope':'fixture'})
    def test_preparation_current_identity_options_mandatory(self):
        source=Path(runtime.__file__).read_text()
        self.assertIn("add_argument('--current-identity',type=Path,required=True)",source)
        self.assertIn("add_argument('--current-identity-sha256',required=True)",source)
        self.assertIn("own_binding(plan['own_producer_root'],plan['prepared'],plan['current_identity'],plan['current_identity_sha256'])",source)
    def test_localization_uses_explicit_association_only(self):
        import native_qsa40_localization_v9 as localization
        source=Path(localization.__file__).read_text()
        self.assertIn('producer_binding(root,model_association)',source)
        self.assertNotIn('producer_binding(root)',source)
        self.assertIn('native_captured_operands_used',source)
    def test_new_source33_decode_uses_fresh_associated_identity(self):
        for name in ('qualify_native_qsa40_v9.py','validate_native_qsa40_v9.py'):
            source=Path(runtime.__file__).with_name(name).read_text()
            self.assertIn("identity=plan['current_identity']",source)
            self.assertNotIn("identity=read(Path(plan['own_producer_root'])",source)

if __name__=='__main__':unittest.main()
