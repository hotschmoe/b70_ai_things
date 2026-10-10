import copy,json,tempfile,unittest
from pathlib import Path
from full_cache_shared_launch_binding_v2 import binding
import test_full_cache_shared_memory_capture_cpu_v2 as memory_controls

class Launch(unittest.TestCase):
 def fixture(self,root):
  sample_root,sample,command=memory_controls.Capture().fixture(root);actor=sample_root.parent.parent;launchroot=actor/'owned-launch-client';launchroot.mkdir();obj=sample['ownership_receipt']['inspection'];obj['State']['Running']=False
  launch={'command':command,'return_code':0,'error':None,'stop_signals':[],'subreaper':{'subreaper':True,'owner_pid':17},'client_pid':91,'client_started_epoch':2.,'client_terminal_epoch':3.,'Docker_daemon_async_creation_time_bound_claimed':False,'launch_descendants':{'producer_pid':91,'producer_session':91,'launch_session_empty':True,'tracking_errors':[],'complete_process_ancestry_from_polling_claimed':False,'subreaper_wait_and_session_census_required':True,'adopted_children':{}}};child={'producer_pid':17,'started_epoch':1.,'finished_epoch':4.,'owned_launch_client':launch}
  (actor/'launch.command.json').write_text(json.dumps(command));(actor/'owned-terminal-inspection.json').write_text(json.dumps(obj));(launchroot/'stdout.log').write_text('actual\n');(launchroot/'stderr.log').write_text('');self.save(actor,child);return actor,child,{'image':'pinned'}
 def save(self,actor,child):(actor/'owned-launch-client/receipt.json').write_text(json.dumps(child['owned_launch_client']))
 def test_original_create_id_and_actual_session_join(self):
  with tempfile.TemporaryDirectory() as td:
   actor,child,plan=self.fixture(Path(td));self.assertTrue(binding(actor,child,plan)['actual_owned_launch_session_empty'])
 def test_timeout_zero_rc_and_live_descendant_rejected(self):
  for kind in ('timeout','descendant','epoch','owner','boolrc'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
    actor,child,plan=self.fixture(Path(td));r=child['owned_launch_client']
    if kind=='timeout':r['error']='deadline'
    elif kind=='descendant':r['launch_descendants']['launch_session_empty']=False
    elif kind=='epoch':r['client_terminal_epoch']=5.
    elif kind=='owner':r['subreaper']['owner_pid']=18
    else:r['return_code']=False
    self.save(actor,child)
    with self.assertRaises(ValueError):binding(actor,child,plan)
 def test_wrong_actual_create_id_cannot_borrow_terminal(self):
  with tempfile.TemporaryDirectory() as td:
   actor,child,plan=self.fixture(Path(td));(actor/'owned-launch-client/stdout.log').write_text('foreign\n')
   with self.assertRaises(ValueError):binding(actor,child,plan)
if __name__=='__main__':unittest.main()
