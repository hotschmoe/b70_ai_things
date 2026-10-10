"""Strict typed terminal and complete Docker recipe mutation controls, CPU only."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import batch53_serial_group_binding_v1 as group
class Controls(unittest.TestCase):
 def result(self):return {'passed':True,'engine_rc':0,'removed':True,'error':None,'failure_traceback':None,'state':{'Running':False,'ExitCode':0,'OOMKilled':False,'Error':'','Status':'exited','Paused':False,'Restarting':False,'Dead':False,'Pid':0}}
 def test_real_shaped_normal_terminal_and_each_contradiction(self):
  self.assertTrue(group.terminal(self.result()))
  for key,value in [('passed',False),('engine_rc',9),('engine_rc',False),('removed',False),('error','bad'),('failure_traceback','failure')]:
   r=self.result();r[key]=value;self.assertRaises(ValueError,group.terminal,r)
  for key,value in [('Running',True),('ExitCode',9),('ExitCode',False),('OOMKilled',True),('Error','contradictory docker error'),('Status','running'),('Paused',True),('Restarting',True),('Dead',True),('Pid',123)]:
   r=self.result();r['state'][key]=value;self.assertRaises(ValueError,group.terminal,r)
 def recipe(self,root):
  child=root/'child';child.mkdir();(child/'plan.snapshot.json').write_text('{}');plan={'args':['--batch','4','--prompt-cache','0','--conversation-cache-mib','0'],'env':{'STRATA_CRITICAL_PATH_TRACE':'0'},'engine_root':'/synthetic-sdk','pack':'/synthetic-pack','image':'synthetic-pinned-image'};cmd,args,env=group.recipe(plan,child,123,0,456);return child,plan,cmd
 def test_exact_PID_group_image_mounts_env_and_batch_override(self):
  with tempfile.TemporaryDirectory()as d:
   child,plan,cmd=self.recipe(Path(d));self.assertEqual(cmd[cmd.index('--name')+1],'b70-prefix-123-serial0');self.assertIn('b70.prefix.plan='+group.sha(child/'plan.snapshot.json'),cmd);self.assertIn('/synthetic-sdk/build:/build:ro',cmd);self.assertIn('/synthetic-sdk/source:/src:ro',cmd);self.assertIn('/synthetic-pack:/pack:ro',cmd);self.assertIn(str(child/'serial-0')+':/results',cmd);self.assertIn('STRATA_BATCH_FIDELITY_DIAG=0',cmd);self.assertIn('STRATA_FIDELITY_DIAG_ACTIVATIONS=1',cmd);self.assertIn('--batch 0',cmd[-1]);self.assertEqual(cmd[-2],plan['image'])
 def test_foreign_command_cannot_equal_reconstruction(self):
  with tempfile.TemporaryDirectory()as d:
   child,plan,cmd=self.recipe(Path(d))
   for change in ('image','pid','env','mount','memory'):
    bad=copy.deepcopy(cmd)
    if change=='image':bad[-2]='foreign'
    elif change=='pid':bad[bad.index('--name')+1]='b70-prefix-999-serial0'
    elif change=='env':bad[bad.index('STRATA_BATCH_FIDELITY_DIAG=0')]='STRATA_BATCH_FIDELITY_DIAG=1'
    elif change=='mount':bad[bad.index('/synthetic-pack:/pack:ro')]='/foreign:/pack:ro'
    else:bad[bad.index('--memory')+1]='999g'
    self.assertNotEqual(group.canonical(bad),group.canonical(cmd))
 def test_public_reader_checks_group_before_extraction_and_actualH50_collector(self):
  here=Path(group.__file__).parent;s=(here/'audit_batch_numerical_suite_v53.py').read_text();start=s.index('def serial_vectors');part=s[start:s.index('def ',start+4)];self.assertLess(part.index('group_binding('),part.index('run_batch_serial_cacheoff_v8'));self.assertIn("child['collection_and_teardown_passed']is True",Path(group.__file__).read_text());self.assertIn("selected['selected_full49_pairs']",Path(group.__file__).read_text())
 def binding_fixture(self,root):
  child,plan,cmd=self.recipe(root);directory=child/'serial-0';directory.mkdir();ids=[7,8];job={'rid':2001,'role':'admission','ids':ids,'ids_sha256_le32':__import__('batch_numerical_prefixes_v2').token_sha(ids),'position':1,'token':8,'max_new':1,'fresh_serial':True,'original_own_state_math_reference':False};value={'jobs':[job],'supplied_consumed_prefix_only':True,'full_model_math_qualified':False};selected=__import__('batch51_serial_jobs_v1').group(value,4,0);plan.update(batch_parent=str(root/'collector'),group_index=0,slots=4,actual_serial_group_binding=selected);digest=group.sha(child/'plan.snapshot.json');parent={'child_pid':123,'plan_sha256':digest};comparisons={str(i):{'bitwise_equal':True}for i in range(49)};result={'passed':True,'collection_and_teardown_passed':True,'full_model_math_qualified':False,'plan_sha256':digest,'actual_serial_job_count':1,'actual_matched_full49_vector_pairs':49,'comparisons':comparisons};(directory/'result.json').write_text(json.dumps(self.result()));(directory/'command.json').write_text(json.dumps(cmd));raw={'ids':ids,'fresh':1};(directory/'serial-2001-admission.json').write_text(json.dumps(raw));(directory/'requests.json').write_text(json.dumps([{'job':job,'raw':raw}]));(child/'serial-comparison.json').write_text(json.dumps({'passed':True,'group':0,'groups_required':1,'comparisons':comparisons}));return plan,parent,result,value
 def test_actual_shaped_binding_positive_then_each_contradiction_rejected(self):
  from types import SimpleNamespace
  original_stat=group.os.stat
  for mutate in ('none','terminal','foreign_command','child_pass','job_count','pair_count','wrong_plan','raw','comparison'):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d);plan,parent,child,value=self.binding_fixture(root);directory=root/'child/serial-0'
    if mutate=='terminal':bad=self.result();bad['state']['Error']='contradictory docker error';(directory/'result.json').write_text(json.dumps(bad))
    elif mutate=='foreign_command':(directory/'command.json').write_text(json.dumps(['CPU_FOREIGN_IMAGE_RECIPE']))
    elif mutate=='child_pass':child['collection_and_teardown_passed']=False
    elif mutate=='job_count':child['actual_serial_job_count']=2
    elif mutate=='pair_count':child['actual_matched_full49_vector_pairs']=48
    elif mutate=='wrong_plan':child['plan_sha256']='0'*64
    elif mutate=='raw':(directory/'serial-2001-admission.json').write_text('{}')
    elif mutate=='comparison':child['comparisons'].pop('0')
    def stat(path,*a,**k):return SimpleNamespace(st_gid=456)if str(path)=='/dev/dri/renderD128'else original_stat(path,*a,**k)
    with patch.object(group,'recollect',return_value=value),patch.object(group.os,'stat',side_effect=stat):
     if mutate=='none':self.assertTrue(group.binding(root,parent,plan,child)['actual_normal_group_terminal'])
     else:self.assertRaises(ValueError,group.binding,root,parent,plan,child)
if __name__=='__main__':unittest.main()
