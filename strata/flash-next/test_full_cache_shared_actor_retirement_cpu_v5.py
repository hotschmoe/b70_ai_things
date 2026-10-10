import copy,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_shared_actor_retirement_v5 as a

class Retirement(unittest.TestCase):
 def obj(self,running=False,rc=0):return {'Name':'/owned','State':{'Running':running,'ExitCode':rc,'OOMKilled':False,'Error':''}}
 def test_actual_running_then_terminal_removed(self):
  seen=[];rows=[self.obj(True),self.obj(False)]
  def owned():return rows.pop(0) if rows else self.obj(False)
  def run(cmd,**kwargs):seen.append(cmd);return types.SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
  result=a.retire('owned',owned,run,lambda n:True,lambda t:None,lambda p,r:None,Path('/tiny'));self.assertTrue(result['normal_terminal']);self.assertEqual(seen,[['docker','kill','--signal','TERM','owned'],['docker','rm','owned']])
 def test_crashed_terminal_removed_but_never_normal_qualified(self):
  commands=[]
  def run(cmd,**kwargs):commands.append(cmd);return types.SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
  result=a.retire('owned',lambda:self.obj(False,139),run,lambda n:True,lambda t:None,lambda p,r:None,Path('/tiny'));self.assertFalse(result['normal_terminal']);self.assertEqual(commands,[['docker','rm','owned']])
 def test_foreign_unknown_inspection_never_stopped_or_removed(self):
  commands=[]
  def owned():raise ValueError('foreign/unknown ownership')
  with self.assertRaises(ValueError):a.retire('owned',owned,lambda *x,**k:commands.append(x),lambda n:True,lambda t:None,lambda p,r:None,Path('/tiny'))
  self.assertEqual(commands,[])
 def test_running_timeout_or_failed_removal_cannot_be_absence(self):
  def run(cmd,**kwargs):return types.SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
  with patch.object(a.time,'monotonic',side_effect=[0,181]),self.assertRaises(ValueError):a.retire('owned',lambda:self.obj(True),run,lambda n:True,lambda t:None,lambda p,r:None,Path('/tiny'))
  seen=[False,True];result=a.retire('owned',lambda:self.obj(),run,lambda n:seen.pop(0),lambda t:None,lambda p,r:None,Path('/tiny'));self.assertFalse(result['normal_terminal']);self.assertTrue(result['removed'])
 def test_typed_terminal_exit_not_boolean(self):
  with self.assertRaises(ValueError):a.retire('owned',lambda:self.obj(False,False),lambda *x,**k:None,lambda n:True,lambda t:None,lambda p,r:None,Path('/tiny'))
 def test_unknown_marker_protects_and_both_parent_kill_paths_require_no_ACK(self):
  import qualify_full_cache_shared_runtime_v5 as parent
  source=parent.adapted_source();self.assertEqual(source.count("and not parent.get('leaf_ack') and not critical.get('retirement_in_progress')"),2)
  with tempfile.TemporaryDirectory()as tmp:
   path=Path(tmp)/'actor-retirement-state.json';path.write_text('{partial')
   self.assertTrue(a.critical_guard(tmp)['retirement_in_progress'])
   path.write_text('{"retirement_in_progress": 0}')
   self.assertTrue(a.critical_guard(tmp)['retirement_in_progress'])
 def test_unknown_then_actual_owned_retirement_records_recovery_not_normal(self):
  obj=self.obj();attempts=[0];commands=[];saved={}
  def owned():
   attempts[0]+=1
   if attempts[0]==1:raise ValueError('unobserved owner')
   return obj
  def run(cmd,**kwargs):commands.append(cmd);return types.SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
  with patch.object(a.time,'sleep',return_value=None):result,ledger=a.retain_until_settled('owned',owned,run,lambda n:True,lambda t:None,lambda p,r:saved.update({p.name:copy.deepcopy(r)}),Path('/tiny'))
  self.assertEqual(commands,[['docker','rm','owned']]);self.assertEqual(ledger['failure_count'],1);self.assertTrue(ledger['owned_terminal_and_removed']);self.assertTrue(result['normal_terminal']);self.assertFalse(ledger['retirement_in_progress'])
 def test_unknown_absence_after_successful_rm_never_repeats_rm_or_inspection(self):
  inspections=[];commands=[];attempts=[0]
  def owned():inspections.append(True);return self.obj()
  def absent(name):
   attempts[0]+=1
   if attempts[0]==1:raise ValueError('temporary Docker inspection transport error')
   return True
  def run(cmd,**kwargs):commands.append(cmd);return types.SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
  result=a.retire('owned',owned,run,absent,lambda t:None,lambda p,r:None,Path('/tiny'))
  self.assertEqual(len(inspections),2);self.assertEqual(commands,[['docker','rm','owned']]);self.assertTrue(result['removed']);self.assertFalse(result['normal_terminal']);self.assertTrue(result['absence_recovery_failures'])
if __name__=='__main__':unittest.main()
