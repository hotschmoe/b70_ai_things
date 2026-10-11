"""Raw parent/child/container generation and actual endpoint identity joins."""
import copy,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from source43_api_baseline_profile_v1 import ALIAS
import validate_source43_api_baseline_v3 as reader
import source43_api_live_owner_decode_v2 as joins
from source43_api_evidence_v3 import models_gate
import test_source43_api_provenance_cpu_v1 as old
class Controls(unittest.TestCase):
 def fixture(self):
  p,m,l=old.Controls().owner();p['schema']=2;p['container_stat_before']=p['container_stat_after']='400 (python) S 7 '+' '.join(['0']*17+['100','0']);p['children'][0]['stat_before']=p['children'][0]['stat_after']='401 (strata) S 400 '+' '.join(['0']*17+['123','0']);return p,m,l
 def test_same_actual_native_container_generation(self):
  p,m,l=self.fixture();self.assertEqual(joins.owner_binding(p,m,l)['container_id'],'container')
 def test_native_PPid_zero_and_foreign_rejected(self):
  for pid in(0,999):
   p,m,l=self.fixture();p['children'][0]['stat_before']=p['children'][0]['stat_before'].replace(' S 400 ',' S '+str(pid)+' ');self.assertRaises(ValueError,joins.owner_binding,p,m,l)
 def test_parent_start_or_PID_changed(self):
  for raw in('999 (python) S 7 '+' '.join(['0']*17+['100','0']),'400 (python) S 7 '+' '.join(['0']*17+['101','0'])):
   p,m,l=self.fixture();p['container_stat_after']=raw;self.assertRaises(ValueError,joins.owner_binding,p,m,l)
 def models(self):return {'data':[{'id':n,'meta':{'artifact_identity':{'artifact_identity_sha256':'sha'}}}for n in('hotschmoe-dd',ALIAS)]}
 def test_original_live_models_signed_identity(self):self.assertTrue(models_gate(self.models(),{'artifact_manifest_sha256':'sha'}))
 def test_wrong_manifest_missing_row_or_alias(self):
  for change in(lambda v:v['data'][0]['meta']['artifact_identity'].update(artifact_identity_sha256='foreign'),lambda v:v['data'].pop(),lambda v:v['data'][1].update(id='foreign')):
   v=self.models();change(v);self.assertRaises(ValueError,models_gate,v,{'artifact_manifest_sha256':'sha'})

 def test_actual_live_terminal_containerId_join(self):
  self.assertTrue(reader.terminal_owner_binding({'Id':'actual'},{'actual_native_host_owner':{'container_id':'actual'}}));self.assertRaises(ValueError,reader.terminal_owner_binding,{'Id':'replacement'},{'actual_native_host_owner':{'container_id':'actual'}})
 def test_complete_post_read_guard_on_success_and_failure(self):
  for fails in(False,True):
   with tempfile.TemporaryDirectory()as td:
    root=Path(td);(root/'report.json').write_text('{}');(root/'raw').write_text('original')
    def mutate():
     (root/'raw').write_text('mutated')
     if fails:raise ValueError('original body failure')
     return {'passed':True}
    with patch.object(reader.c,'source_binding',return_value='source'):
     with self.assertRaises(ValueError)as caught:reader.guarded_read(root,mutate)
    self.assertIn('tree changed',str(caught.exception))
    if fails:self.assertIn('original body failure',str(caught.exception))
 def test_failed_read_preserved_when_tree_stable(self):
  with tempfile.TemporaryDirectory()as td:
   root=Path(td);(root/'report.json').write_text('{}')
   with patch.object(reader.c,'source_binding',return_value='source'):
    with self.assertRaisesRegex(ValueError,'original failed read'):reader.guarded_read(root,lambda:(_ for _ in()).throw(ValueError('original failed read')))
if __name__=='__main__':unittest.main()
