"""Synthetic lifecycle/pre-publisher and actual-shaped native marker controls."""
import copy,datetime,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import native_rms_lifecycle_binding_v5 as l
import test_native_rms_publisher_cpu_v2 as fixtures
class Tests(unittest.TestCase):
 def markers(self):return ['RMS37_DEVICE pid=23 backend=level_zero affinity=0 name=ACTUAL_GPU vendor=ACTUAL_VENDOR driver=ACTUAL_DRIVER']+[f'RMS37_FRAME route={r} graph_replay={r} actual_compiled_hc=1 fields=4 synthetic_norm=1 synthetic_zero_down_up=1 original_model_math_qualified=0' for r in range(3)]+['RMS37_RESULT routes=3 owned_allocations_freed=1 graph_retired=1 device_intrinsics_qualified=0 full_model_math_qualified=0']
 def test_native_source_marker_roster_and_error_negatives(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'runtime.log';good=self.markers();p.write_text('\n'.join(good)+'\n');self.assertTrue(l.native_trace(p)['actual_unique_frame_free_markers'])
   for bad in [good[:-1],good+[good[-1]],good[:1]+list(reversed(good[1:4]))+good[4:],good+['RMS37_ERROR actual failure'],[s.replace('owned_allocations_freed=1','owned_allocations_freed=0') for s in good]]:
    p.write_text('\n'.join(bad)+'\n')
    with self.assertRaises(ValueError):l.native_trace(p)
 def report(self):
  iso=lambda v:datetime.datetime.fromtimestamp(v,datetime.timezone.utc).isoformat();return {'compile_command':{'started_command_epoch':1.,'finished_epoch':4.},'run_command':{'started_command_epoch':5.,'finished_epoch':8.},'compile_terminal':{'State':{'StartedAt':iso(2),'FinishedAt':iso(3)}},'run_terminal':{'State':{'StartedAt':iso(6),'FinishedAt':iso(7)}},'GPU_terminal_epoch':9.}
 def test_actual_container_EOF_sequence(self):
  l.docker_epochs(self.report())
  for kind in ('compilefinish','runtimebeforecompile','GPUbeforeEOF','missingzone'):
   r=self.report()
   if kind=='compilefinish':r['compile_terminal']['State']['FinishedAt']='1970-01-01T00:00:05Z'
   elif kind=='runtimebeforecompile':r['run_command']['started_command_epoch']=3.
   elif kind=='GPUbeforeEOF':r['GPU_terminal_epoch']=7.5
   else:r['run_terminal']['State']['StartedAt']='1970-01-01T00:00:06'
   with self.assertRaises(ValueError):l.docker_epochs(r)
 def fixture(self,root):
  lock,old=fixtures.Publisher().fixture(root);proof=old['post_full4'];proof['after_terminal_and_post_health_epoch']=15.5;return lock,{'pre_full4':proof,'pre_pages':old['before_post_full4_pages'],'pre_full4_after_pages':old['post_pages'],'pre_health':{'rows':[{'started_command_epoch':22.}]},'started_epoch':14.}
 def bind(self,root,lock,r):
  (root/'pre-full4.json').write_text(json.dumps(r['pre_full4'])+'\n');h=hashlib.sha256(b'X'*4096).hexdigest()
  with patch.object(l.p,'KNOWN_PAGES',[(0,h),(4096,h)]):return l.pre_publisher(root,r,lock,'CPU_LOCK',root)
 def test_original_pre_full4_pages_before_health(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);lock,r=self.fixture(root);self.assertTrue(self.bind(root,lock,r))
 def test_forged_pre_hash_size_order_page_health_negatives(self):
  for kind in ('bytes','hash','order','beforepage','afterpage','health'):
   with tempfile.TemporaryDirectory() as t:
    root=Path(t);lock,r=self.fixture(root)
    if kind=='bytes':r['pre_full4']['rows'][0]['bytes']=1
    elif kind=='hash':r['pre_full4']['rows'][0]['sha256']='CHANGED'
    elif kind=='order':r['pre_full4']['rows'].reverse()
    elif kind=='beforepage':r['pre_pages']['epoch']=17.
    elif kind=='afterpage':r['pre_full4_after_pages']['epoch']=19.
    else:r['pre_health']['rows'][0]['started_command_epoch']=20.5
    with self.assertRaises(ValueError):self.bind(root,lock,r)
if __name__=='__main__':unittest.main()
