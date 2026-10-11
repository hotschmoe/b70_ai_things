import ast,copy,importlib,inspect,json,tempfile,unittest
from pathlib import Path
import postboot_original_native_manifest_v1 as p
import postboot_original_model_association_v1 as m

class Controls(unittest.TestCase):
 def association(self,path):
  current=m.current_stat(path);old=[current[0]+100,*current[1:]];return {'mapping':[{'path':str(path),'historical_stat5':old,'current_stat5':current,'publisher_sha256':'synthetic-byte-fixture'}]},old
 def test_actual_tiny_stat_association_no_stat_or_read_patch(self):
  with tempfile.TemporaryDirectory()as temp:
   path=Path(temp)/'tiny';path.write_bytes(b'fixture');association,old=self.association(path);self.assertTrue(p.associated(association,path,old));path.write_bytes(b'changed');self.assertRaises(ValueError,p.associated,association,path,old)
 def test_non_device_stat_change_and_bool_rejected(self):
  with tempfile.TemporaryDirectory()as temp:
   path=Path(temp)/'tiny';path.write_bytes(b'fixture');association,old=self.association(path)
   for index in range(1,5):
    bad=list(old);bad[index]+=1;self.assertRaises(ValueError,p.associated,association,path,bad)
   bad=list(old);bad[0]=True;self.assertRaises(ValueError,p.associated,association,path,bad)
 def test_explicit_parent_missing_field_view_exact_only(self):
  with tempfile.TemporaryDirectory()as temp:
   path=Path(temp)/'original.json';path.write_text('{"parent_generation":1373,"value":"original"}');view={'parent_generation':1373,'value':'original','started_epoch':1.25};self.assertEqual(p.before_view(path,view),view)
   for bad in [{**view,'value':'modified'},{**view,'extra':1},{**view,'started_epoch':True}]:self.assertRaises(ValueError,p.before_view,path,bad)
   self.assertEqual(json.loads(path.read_text()),{'parent_generation':1373,'value':'original'})
 def test_exact_original_controller_edges_and_mandatory_argument(self):
  for name in sorted(p.CONTROLLERS):
   original=importlib.import_module(name);stat_method=original.stat_signature;read_method=original.read;ns,binding=p.controller(original,{})
   self.assertEqual(binding['stat_edges'],3);self.assertIs(original.stat_signature,stat_method);self.assertIs(original.read,read_method);self.assertNotEqual(ns['validate_prepared'],original.validate_prepared);self.assertEqual(inspect.signature(ns['validate_prepared']).parameters['_model_association'].default,inspect.Parameter.empty)
 def test_parent_final_only_one_stat_edge_preserves_original_source(self):
  for name in sorted(p.PARENTS):
   original=importlib.import_module(name);before=Path(original.__file__).read_bytes();ns,binding=p.derive(original,('validate_final_source_proof',),{});self.assertEqual(binding['stat_edges'],1);self.assertEqual(Path(original.__file__).read_bytes(),before)
 def test_exact_comparison_retains_publisher_hash_and_stability(self):
  source="assert row['sha256']==row['expected_sha256']==want['sha256'] and row['stat_before']==row['stat_after']==c1.stat_signature(Path(row['path']))";tree=ast.parse(source);port=p.Port([]);updated=ast.fix_missing_locations(port.visit(copy.deepcopy(tree)));text=ast.unparse(updated);self.assertIn("row['sha256'] == row['expected_sha256'] == want['sha256']",text);self.assertIn("row['stat_before'] == row['stat_after']",text);self.assertIn('_associated(_model_association',text);self.assertEqual(port.stat_edges,1)
 def test_metadata_candidate_upload_dispatch_is_explicitly_ported(self):
  original=importlib.import_module('layer0_numerical_qualification_v10');tree=ast.parse(Path(original.__file__).read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='candidate_binding');updated=p.Port({'candidate_binding','manifest_binding'}).visit(copy.deepcopy(node));text=ast.unparse(updated);self.assertNotIn('c1.metadata_admission_gate(',text);self.assertIn('_controller_metadata(raw, model_association=_model_association)',text);self.assertIn('_parent_validate(prepared, raw, model_association=_model_association)',text)
 def test_manifest_candidates_mandatory_controller_and_parent_ports(self):
  for name in sorted(p.MANIFESTS):
   original=importlib.import_module(name);ns,binding=p.derive(original,('candidate_binding','manifest_binding'),{}, {'_controller_validate':p.validate_prepared,'_parent_validate':p.validate_final_source_proof});self.assertEqual(binding['stat_edges'],0);self.assertEqual(inspect.signature(ns['manifest_binding']).parameters['_model_association'].default,inspect.Parameter.empty)
 def test_bad_association_current_receipt_refused_before_modelstat(self):
  self.assertRaises(ValueError,p.recheck,{'current_identity_sha256':'not-actual-root-receipt'})
 def test_scope_never_current_runtime_or_old_stat_pass(self):
  scoped=p.scope({'mapping':[]});self.assertFalse(scoped['old_current_stat_gate_passed']);self.assertFalse(scoped['historical_GPU_health_transferred']);self.assertFalse(scoped['current_runtime_qualified']);self.assertFalse(scoped['new_known_page_probe_performed'])
 def test_historical_sentinel_metadata_no_new_page_read(self):
  with tempfile.TemporaryDirectory()as temp:
   path=Path(temp)/'tiny-00003-of-00004';path.write_bytes(b'fixture');association,old=self.association(path);record={'schema':3,'path':str(path),'pages':[{'offset':5,'bytes':4096,'sha256':'x','expected_sha256':'x'},{'offset':9,'bytes':4096,'sha256':'y','expected_sha256':'y'}],'read_mode':'buffered read only; no invalidation/write/repair'}
   self.assertEqual(p.historical_sentinel(association,[{'path':str(path),'stat':old}],record,[(5,'x'),(9,'y')]),record);wrong=copy.deepcopy(record);wrong['pages'][0]['sha256']='other';self.assertRaises(ValueError,p.historical_sentinel,association,[{'path':str(path),'stat':old}],wrong,[(5,'x'),(9,'y')]);self.assertEqual(path.read_bytes(),b'fixture')

if __name__=='__main__':unittest.main()
