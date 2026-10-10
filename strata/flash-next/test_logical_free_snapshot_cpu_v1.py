"""Independent tiny real-parser trace/negative controls, no runtime evidence reads."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import collect_logical_free_snapshot_v1 as pure
import parse_usm_logical_free_trace as parser
from operation_evidence_snapshot_v1 import EvidenceEpoch
TRACE='''<--- urContextGetNativeHandle(.hContext = 0x1, .phNativeContext = 0x2 (0x3)) -> UR_RESULT_SUCCESS;
<--- urUSMDeviceAlloc(.hContext = 0x1, .size = 16, .ppMem = 0x4 (0x5)) -> UR_RESULT_SUCCESS;
UPLOAD_USM {"event":"owner_register","stage":0,"ze_context":"0x3","pointer":"0x5","bytes":16}
UPLOAD_USM {"event":"destroy_begin","stage":0}
<--- urUSMFree(.hContext = 0x1, .pMem = 0x5) -> UR_RESULT_SUCCESS;
UPLOAD_USM {"event":"destroy_end","stage":0,"owning_destructor_returned":true}
'''
class Controls(unittest.TestCase):
 def fixture(self,r):
  log=r/'log';log.write_text(TRACE);logical=r/'logical';checked=parser.parse_trace(TRACE);self.assertTrue(checked['passed']);checked['negative_controls']=parser.negative_controls(TRACE,True);logical.write_text(json.dumps(checked));code=Path(pure.__file__).resolve();parser_code=Path(parser.__file__).resolve();roster=[(p,hashlib.sha256(p.read_bytes()).hexdigest()) for p in (log,logical,code,parser_code)];return log,logical,code,parser_code,roster
 def call(self,e,log,logical,code,parser_code,roster,count=1):return e.pure_once('original-upload-logical-free-subset',code,hashlib.sha256(code.read_bytes()).hexdigest(),{'log_path':str(log),'logical_path':str(logical),'expected_owner_count':count},[log,logical,parser_code],pure.validate,current_roster=roster)
 def test_original_five_parser_passes_execute_once_for_eighty_callers(self):
  with tempfile.TemporaryDirectory() as d:
   log,logical,code,parser_code,roster=self.fixture(Path(d));e=EvidenceEpoch(roster)
   with patch.object(pure,'parse_trace',wraps=pure.parse_trace) as parsed,patch.object(pure,'negative_controls',wraps=pure.negative_controls) as negatives:
    for i in range(80):self.assertTrue(self.call(e,log,logical,code,parser_code,roster)['original_upload_logical_subset_passed'])
    self.assertEqual(parsed.call_count,1);self.assertEqual(negatives.call_count,1)
   e.seal_predevice();e.finalize()
 def test_changed_expected_owner_count_keeps_original_predicate_failure(self):
  with tempfile.TemporaryDirectory() as d:
   log,logical,code,parser_code,roster=self.fixture(Path(d));e=EvidenceEpoch(roster);self.call(e,log,logical,code,parser_code,roster);self.assertRaises(ValueError,self.call,e,log,logical,code,parser_code,roster,2)
 def test_corrupted_raw_trace_never_borrows_old_success(self):
  with tempfile.TemporaryDirectory() as d:
   log,logical,code,parser_code,roster=self.fixture(Path(d));log.write_text(TRACE.replace('UR_RESULT_SUCCESS;\nUPLOAD_USM {"event":"destroy_end"','UR_RESULT_ERROR_UNKNOWN;\nUPLOAD_USM {"event":"destroy_end"'));roster=[(p,hashlib.sha256(p.read_bytes()).hexdigest()) for p,_ in roster];e=EvidenceEpoch(roster);self.assertRaises(ValueError,self.call,e,log,logical,code,parser_code,roster)
if __name__=='__main__':unittest.main()
