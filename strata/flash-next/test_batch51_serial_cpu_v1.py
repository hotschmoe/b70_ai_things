"""Serial H51 job/role/shape/handshake/source controls; no actual data execution."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import batch51_serial_jobs_v1 as j
import batch51_prelease_v1 as pre
import batch_numerical_execution_v51 as c
from batch_numerical_prefixes_v2 import token_sha
HERE=Path(__file__).resolve().parent
class Controls(unittest.TestCase):
 def value(self):
  rows=[]
  for role in ('admission','later'):
   for rid in range(2001,2005):
    ids=[rid,7]+([8]if role=='later'else []);rows.append({'rid':rid,'role':role,'ids':ids,'ids_sha256_le32':token_sha(ids),'position':len(ids)-1,'token':ids[-1],'max_new':1,'fresh_serial':True,'original_own_state_math_reference':False})
  return {'jobs':rows,'supplied_consumed_prefix_only':True,'full_model_math_qualified':False}
 def test_eight_actual_shaped_jobs_two_groups392_pairs_no_solo(self):
  v=self.value();a=j.group(v,4,0);b=j.group(v,4,1);self.assertEqual((a['selected_job_count'],b['selected_job_count']), (6,2));self.assertEqual((a['selected_full49_pairs'],b['selected_full49_pairs']),(294,98));self.assertEqual(a['all_full49_pairs'],392);self.assertEqual(a['groups_required'],2);self.assertFalse(a['solo_migration_observed'])
 def fixture(self,root):
  v=self.value();(root/'serial-jobs.json').write_text(json.dumps(v));requests={str(rid):{'ids':[rid,7],'generated':[8,9]}for rid in range(2001,2005)};(root/'requests.json').write_text(json.dumps(requests));lines=[]
  for row in v['jobs']:
   phase='admission_last'if row['role']=='admission'else 'batch_step_first'
   for layer in range(-1,48):lines.append('SBF vector rid='+str(row['rid'])+' layer='+str(layer)+' phase='+phase+' pos='+str(row['position'])+' token='+str(row['token'])+' file=tiny')
  (root/'engine.combined.log').write_text('\n'.join(lines));return v
 def test_actual_shaped_recollection_matches_all48_head_and_source_history(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);v=self.fixture(root);self.assertEqual(j.recollect(root,4),v)
 def test_missing_duplicate_role_or_head_and_changed_prefix_refused(self):
  for change in ('missing_job','duplicate','missing_head','wrong_input','solo'):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d);v=self.fixture(root)
    if change=='missing_job':v['jobs'].pop()
    elif change=='duplicate':v['jobs'].append(copy.deepcopy(v['jobs'][0]))
    elif change=='wrong_input':v['jobs'][0]['ids'][0]+=1;v['jobs'][0]['ids_sha256_le32']=token_sha(v['jobs'][0]['ids'])
    elif change=='solo':v['jobs'][0]['role']='solo_migration'
    else:(root/'engine.combined.log').write_text((root/'engine.combined.log').read_text().split('\n',1)[1])
    (root/'serial-jobs.json').write_text(json.dumps(v));self.assertRaises(ValueError,j.recollect,root,4)
 def test_bounds_booleans_wrong_budget_and_extra_group_refused(self):
  v=self.value();self.assertRaises(ValueError,j.group,v,4,2);self.assertRaises(ValueError,j.group,v,4,True)
  for key,value in [('rid',True),('max_new',2),('position',False),('fresh_serial',1)]:
   bad=copy.deepcopy(v);bad['jobs'][0][key]=value;self.assertRaises(ValueError,j.validate,bad,4)
  bad=copy.deepcopy(v);bad['jobs']*=3;self.assertRaises(ValueError,j.validate,bad,4)
 def test_serial_only_prelease_refuses_native_and_API_before_any_digest(self):
  def require(ok,message):
   if not ok:raise ValueError(message)
  ctrl=SimpleNamespace(require=require)
  for kind in ('native','api'):
   self.assertRaises(ValueError,pre.cheap,{'schema':4,'harness_generation':51,'slots':4,'kind':kind},ctrl)
 def test_absentPIN_cacheOFF_extractor_and_exact49_comparison_unchanged(self):
  import run_batch_serial_controls_v51 as serial
  from serial_prefix_qualification_v8 import Protocol
  from extract_serial_cacheoff_numerical_v1 import extract
  self.assertIs(serial.CacheOffProtocol.request,Protocol.request);self.assertIs(serial.extract,extract);source=Path(serial.__file__).read_text();self.assertIn('fresh=1, max_new=1',source);self.assertIn('range(-1, 48)',source);self.assertIn('compare_all',source);self.assertNotIn('allow_legacy_pin_zero=True',source)
 def test_semanticREADY_thenACK_currentjobs_bytes_and_unrelaxed_health(self):
  source=(HERE/'run_batch_serial_controls_v51.py').read_text();self.assertLess(source.index('engine_binding('),source.index('ready_row=ready'));self.assertLess(source.index('ack_row=wait_ack'),source.index('pack_epoch.seal_predevice()'));self.assertLess(source.index('sdk_epoch.seal_predevice()'),source.index('protocol = CacheOffProtocol'));self.assertIn('<=300',source);self.assertIn('freshness(health,plan,time.time())',source)
 def test_exact_H50_collector_and_H51_parent_generation_joins(self):
  auditor=(HERE/'audit_batch_numerical_suite_v51.py').read_text();self.assertIn('from audit_batch_numerical_suite_v50 import parent_arm as collector_arm',auditor);self.assertIn("collector_plan['harness_generation']==50",auditor);self.assertIn('harness_generation\') == 51',auditor);self.assertIn('recollect',auditor)
  self.assertIn("parent_generation')==51",(HERE/'batch51_runtime_evidence_v1.py').read_text())
if __name__=='__main__':unittest.main()
