"""Exact known UR log view and untouched-tree controls; no model/device reads."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import adjudicate_hc_composition35_v3_log_v1 as a
def log():
 caps=' | '.join('UR_DEVICE_FP_CAPABILITY_FLAG_'+v for v in a.CAPABILITIES)
 return '\n'.join(['   <--- urDeviceGetInfo(.hDevice = 0x123, .propName = UR_DEVICE_INFO_DRIVER_VERSION, .propSize = 13, .pPropValue = 0x456 (CPU_SYNTHETIC), .pPropSizeRet = nullptr) -> UR_RESULT_SUCCESS;','HC35_COMPOSITION_DEVICE backend=level_zero vendor=CPU_SYNTHETIC driver=CPU_SYNTHETIC affinity=0 selector=level_zero:gpu',a.PREFIX,'   <--- urDeviceGetInfo(.hDevice = 0x123, .propName = UR_DEVICE_INFO_SINGLE_FP_CONFIG, .propSize = 4, .pPropValue = 0x456 ('+caps+'), .pPropSizeRet = nullptr) -> UR_RESULT_SUCCESS;','32,16,2,4,8,64,65,'+a.SCOPE,'HC35_COMPOSITION_CONFIG synthetic=1 compiled_composition=1 shadows_separate=1 subgroup=32 eps=1e-6 graph_and_queue=1 normal_model_graph_qualified=0 device_intrinsics_qualified=0 model_math_qualified=0'])+'\n'
class Controls(unittest.TestCase):
 def test_exact_three_line_source_query_join_retains_original(self):
  text=log();view=a.known_interleaved_observation(text);self.assertEqual(view['observed_enum_values'],[32,16,2,4,8,64,65]);self.assertEqual(view['original_lines'],text.splitlines()[2:5]);self.assertFalse(view['device_general_FP_mode_qualified']);self.assertEqual(view['joined_FP_CONFIG'],'HC35_COMPOSITION_FP_CONFIG flags=32,16,2,4,8,64,65,'+a.SCOPE)
 def test_extra_foreign_failed_truncated_query_refused(self):
  text=log();cases=[text.replace('UR_DEVICE_INFO_SINGLE_FP_CONFIG','UR_DEVICE_INFO_HALF_FP_CONFIG'),text.replace('UR_RESULT_SUCCESS;\n32,','UR_RESULT_ERROR_UNKNOWN;\n32,'),text.replace('SINGLE_FP_CONFIG, .propSize = 4','SINGLE_FP_CONFIG, .propSize = 8'),text.replace('0x123, .propName = UR_DEVICE_INFO_SINGLE','0x999, .propName = UR_DEVICE_INFO_SINGLE'),text.replace('32,16,2,4,8,64,65,','32,16,'),text.replace(a.PREFIX,a.PREFIX+'\n   ---> urDeviceGetInfo'),text.replace(a.PREFIX,a.PREFIX+'\n'+a.PREFIX),text.replace('32,16,2,4,8,64,65,'+a.SCOPE,'')]
  for bad in cases:
   with self.subTest(bad=bad[:70]),self.assertRaises(ValueError):a.known_interleaved_observation(bad)
 def test_original_saved_logSHA_is_mandatory(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);path=root/'leaf.log';path.write_text(log());saved={'log_sha256':a.sha(path)};r=a.joined_leaf_binding(root,saved);self.assertFalse(r['device_general_FP_mode_qualified']);path.write_text(log().replace('vendor=CPU_SYNTHETIC','vendor=CHANGED'))
   with self.assertRaises(ValueError):a.joined_leaf_binding(root,saved)
 def test_tree_signature_detects_mutation_and_symlinks(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/'CPU.txt';p.write_text('before');before=a.tree_binding(root);p.write_text('after');self.assertNotEqual(before,a.tree_binding(root));(root/'alias').symlink_to(p)
   with self.assertRaises(ValueError):a.tree_binding(root)
 def test_only_original_known_failure_allowed_hook_restored_and_tree_unchanged(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'CPU.json').write_text('{}');pins={'CPU.json':a.sha(root/'CPU.json')};original=a.frozen.leaf_log_binding;calls=[]
   def validation(actual):
    calls.append(a.frozen.leaf_log_binding)
    if len(calls)==1:raise ValueError('FP capability observation scope changed')
    self.assertIs(a.frozen.leaf_log_binding,a.joined_leaf_binding);return {'passed':True,'CPU_MOCK_ONLY':True},{'CPU_MOCK_ONLY':True},{'all_other_frozen_gates_run':True,'CPU_MOCK_ONLY':True}
   with patch.object(a,'RUN_ROOT',root),patch.object(a,'PINS',pins),patch.object(a,'source_binding',return_value={'CPU_ONLY':True}),patch.object(a.frozen,'finalized_binding',validation):p,r,b=a.finalized_binding(root)
   self.assertIs(a.frozen.leaf_log_binding,original);self.assertTrue(b['original_reader_failed']);self.assertFalse(b['original_reader_success_claimed']);self.assertFalse(b['original_evidence_rewritten']);self.assertTrue(b['original_tree_before_after_unchanged'])
 def test_unrelated_original_failure_cannot_be_adjudicated(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   with patch.object(a,'RUN_ROOT',root),patch.object(a,'PINS',{}),patch.object(a,'source_binding',return_value={}),patch.object(a.frozen,'finalized_binding',side_effect=ValueError('CPU_FAULT_OR_SOURCE_CHANGED')):
    with self.assertRaises(ValueError):a.finalized_binding(root)
if __name__=='__main__':unittest.main()
