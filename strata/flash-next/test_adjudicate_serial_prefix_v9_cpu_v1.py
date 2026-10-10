"""Fail-closed admission negatives; no actual adjudication/GPU/model reads."""
import copy,unittest
import adjudicate_serial_prefix_v9_v1 as a
class Tests(unittest.TestCase):
 def setUp(self):
  self.parent={'passed':False,'errors':['V9 finalize rejected qualification'],'child_return_code':0,'owned_containers_terminal':True,'forced_cleanup':False,'interrupted':False,'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True}
  result={'passed':True,'error':None,'removed':True,'engine_rc':0,'terminal':{'Running':False,'ExitCode':0,'OOMKilled':False,'Error':''}}
  self.child={'schema':9,'group':'remaining','passed':False,'post_health_passed':False,'error':None,'numerical_and_teardown_passed':True,'complete_prefix_qualified':False,'results':[dict(copy.deepcopy(result),name=g) for g in a.v.REMAINING_GROUPS]}
  self.log='line 502, in finalize\nValueError: Cold raw request copy changed\n'
 def gate(self):return a.closed_admission(self.parent,self.child,self.log)
 def test_known_synthetic_admission_scope_only(self):self.assertTrue(self.gate())
 def test_other_failure_or_prior_pass_not_accepted(self):
  for key,value in [('errors',['some GPU error']),('errors',['V9 finalize rejected qualification','other']),('passed',True),('child_return_code',1),('interrupted',True),('forced_cleanup',True),('owned_containers_terminal',False),('pre_health_passed',False),('post_health_passed',False),('kernel_fault_gate_passed',False)]:
   before=self.parent[key];self.parent[key]=value
   with self.assertRaises(ValueError):self.gate()
   self.parent[key]=before
 def test_wrong_exception_or_numerical_scope_rejected(self):
  self.log='ValueError: other'
  with self.assertRaises(ValueError):self.gate()
  self.log='line 502, in finalize\nValueError: Cold raw request copy changed'
  for key,value in [('schema',8),('group','parked'),('passed',True),('post_health_passed',True),('error','failed'),('numerical_and_teardown_passed',False),('complete_prefix_qualified',True)]:
   before=self.child[key];self.child[key]=value
   with self.assertRaises(ValueError):self.gate()
   self.child[key]=before
 def test_missing_reordered_or_unhealthy_group_rejected(self):
  full=copy.deepcopy(self.child['results'])
  for changed in [full[:-1],list(reversed(full))]:
   self.child['results']=changed
   with self.assertRaises(ValueError):self.gate()
  self.child['results']=copy.deepcopy(full);self.child['results'][2]['terminal']['OOMKilled']=True
  with self.assertRaises(ValueError):self.gate()
if __name__=='__main__':unittest.main()
