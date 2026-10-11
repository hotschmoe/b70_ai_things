import tempfile,unittest,json,hashlib,copy
from pathlib import Path
from unittest.mock import patch
import source43_normal_historical_association_v1 as a
class Tests(unittest.TestCase):
    def test_tiny_current_roster_and_changed_content(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);file=root/'report.json';file.write_bytes(b'{}');expected={'report.json':{'sha256':hashlib.sha256(b'{}').hexdigest(),'bytes':2}};self.assertEqual(a.tree_binding(root,expected)['report.json']['bytes'],2);file.write_bytes(b'[]')
            with self.assertRaises(ValueError):a.tree_binding(root,expected)
    def test_hidden_file_and_alias_refusal(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'report.json').write_bytes(b'{}');expected={'report.json':{'sha256':hashlib.sha256(b'{}').hexdigest(),'bytes':2}};(root/'hidden').write_bytes(b'x')
            with self.assertRaises(ValueError):a.tree_binding(root,expected)
            (root/'hidden').unlink();(root/'alias').symlink_to(root/'report.json')
            with self.assertRaises(ValueError):a.tree_binding(root,expected)
    def test_root_alias_refusal(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)/'real';root.mkdir();alias=Path(d)/'alias';alias.symlink_to(root)
            with self.assertRaises(ValueError):a.tree_binding(alias,{'report.json':{'bytes':1,'sha256':'x'}})
    def fixture(self):
        public=dict(passed=True,report_sha256=a.REPORT_SHA,root=str(a.NORMAL),source_plan_sha256=a.PRODUCER_SOURCE_SHA,source_generation=43,materialized_mode=0,OFF_equivalence_passed=True,normal43_fresh_constructor_and_bounded_outputs_observed=True,current_public_full4_rehashed=True,source43_sdk={'SDK':43})
        record=dict(role='normal43_mode0',root=str(a.NORMAL),report_sha256=a.REPORT_SHA,original_public_admission=public,byte_roster={'report.json':{'sha256':a.REPORT_SHA}},old_current_registry_gate_passed=False,historical_completed43_source_and_lifecycle_only=True,fresh_API_or_original_math_authority_transferred=False)
        report=dict(passed=True,errors=[],phase='off-equivalence',source_plan_sha256=a.PRODUCER_SOURCE_SHA);return record,public,report
    def test_historical_true_payload_unchanged_outer_false(self):
        r,p,q=self.fixture();old=copy.deepcopy(p);result=a.envelope(r,p,q);self.assertEqual(p,old);self.assertIs(result['original_public_admission'],p);self.assertTrue(p['current_public_full4_rehashed']);self.assertFalse(result['current_public_full4_rehashed']);self.assertTrue(result['fresh_per_arm_full4_pages_health_and_retirement_required'])
    def test_failed_or_foreign_evidence_refused(self):
        r,p,q=self.fixture()
        for changed in ({**q,'passed':False},{**q,'source_plan_sha256':'foreign'}):
            with self.assertRaises(ValueError):a.envelope(r,p,changed)
        with self.assertRaises(ValueError):a.envelope({**r,'role':'generic'},p,q)
    def test_matching_current_fields_never_qualifies_current_bytes(self):
        r,p,q=self.fixture();record=a.envelope(r,p,q)
        with patch.object(a,'bound',return_value=(b'{"args":[]}',{})):
            result=a.match_current_contract(record,{'SDK':43},{'args':[]});self.assertFalse(result['current_SDK_bytes_verified_by_this_function'])
            with self.assertRaises(ValueError):a.match_current_contract(record,{'SDK':44},{'args':[]})
    def test_duplicate_and_nonfinite_metadata_refused(self):
        for raw in(b'{"x":1,"x":2}',b'{"x":NaN}'):
            with self.assertRaises(ValueError):a.parse(raw)
if __name__=='__main__':unittest.main()
