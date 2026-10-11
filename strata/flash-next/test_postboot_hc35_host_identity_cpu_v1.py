import ast,copy,tempfile,unittest
from pathlib import Path
import postboot_hc35_host_identity_v1 as h
import qualify_hc35_host_runtime_v1 as scalar

class Controls(unittest.TestCase):
 def fixture(self,root):
  path=root/'tiny-helper';path.write_bytes(b'CPU tiny bytes');current=scalar.file_binding(path);old=copy.deepcopy(current);old['stat'][0]+=100;mapping={'mapping':[{'historical_stat5':old['stat'],'current_stat5':current['stat']}]};return old,current,mapping
 def test_actual_shape_helper_and_build_history_device_only(self):
  with tempfile.TemporaryDirectory()as tmp:
   old,current,mapping=self.fixture(Path(tmp));record={'helper':old,'receipts':{'compile.log':old}};ctx=h.Association(record,mapping);self.assertTrue(ctx.compare(record,{'helper':current,'receipts':{'compile.log':current}}));self.assertFalse(ctx.evidence()['old_current_runtime_gate_passed'])
 def test_actual_file_left_or_right_source_dispatch(self):
  with tempfile.TemporaryDirectory()as tmp:
   old,current,mapping=self.fixture(Path(tmp));ctx=h.Association(old,mapping);self.assertTrue(ctx.compare(current,old));self.assertTrue(ctx.compare(old,current))
 def test_inode_size_timestamps_hash_path_and_dev_undeclared_refused(self):
  with tempfile.TemporaryDirectory()as tmp:
   old,current,mapping=self.fixture(Path(tmp));ctx=h.Association(old,mapping)
   for index in range(5):
    bad=copy.deepcopy(current);bad['stat'][index]+=1;self.assertRaises(ValueError,ctx.compare,old,bad)
   for key,value in [('sha256','0'*64),('path',str(Path(tmp)/'foreign'))]:
    bad={**current,key:value};self.assertRaises(ValueError,ctx.compare,old,bad)
 def test_arbitrary_stat_array_or_bool_not_normalized(self):
  ctx=h.Association({}, {'mapping':[]});self.assertRaises(ValueError,ctx.compare,{'stat':[51,2,3,4,5]},{'stat':[45,2,3,4,5]});self.assertRaises(ValueError,ctx.compare,True,1)
 def test_byte_change_after_observation_refused(self):
  with tempfile.TemporaryDirectory()as tmp:
   old,current,mapping=self.fixture(Path(tmp));ctx=h.Association(old,mapping);ctx.compare(old,current);Path(current['path']).write_bytes(b'altered');self.assertRaises(ValueError,ctx.evidence)
 def test_PATH_is_separate_and_all_nonPATH_exact(self):
  with tempfile.TemporaryDirectory()as tmp:
   old,current,mapping=self.fixture(Path(tmp));env={'PATH':'historical','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'};a={'libraries':old,'execution_environment':env,'rounding':'FE_TONEAREST'};b={'libraries':current,'execution_environment':{**env,'PATH':'current'},'rounding':'FE_TONEAREST'};ctx=h.Association(a,mapping);self.assertTrue(ctx.runtime_compare(a,b));self.assertEqual(ctx.paths[0]['historical_PATH'],'historical');self.assertEqual(ctx.paths[0]['current_PATH'],'current')
   for key,value in [('OMP_NUM_THREADS','2'),('LD_PRELOAD','FOREIGN'),('NUMEXPR_NUM_THREADS','1')]:
    wrong=copy.deepcopy(b);wrong['execution_environment'][key]=value;self.assertRaises(ValueError,ctx.runtime_compare,a,wrong)
 def test_unchanged_original_dispatch_and_mandatory_byte_guards(self):
  before_file=scalar.file_binding;before_runtime=scalar.runtime_binding
  with tempfile.TemporaryDirectory()as tmp:
   old,current,mapping=self.fixture(Path(tmp));ctx=h.Association(old,mapping);ns=h.derived(scalar,('validate_fixture','finalized_binding'),ctx,{'_historical_runtime':{}});self.assertIs(scalar.file_binding,before_file);self.assertIs(scalar.runtime_binding,before_runtime);self.assertNotEqual(ns['finalized_binding'],scalar.finalized_binding)
 def test_host_runtime_shape_preserves_old_prepost_equality(self):
  node=ast.parse("assert r['runtime_pre']==r['runtime_post']==current");text=ast.unparse(ast.fix_missing_locations(h.Port().visit(node)));self.assertIn("r['runtime_pre'] == r['runtime_post']",text);self.assertIn('_runtime_associate',text)

if __name__=='__main__':unittest.main()
