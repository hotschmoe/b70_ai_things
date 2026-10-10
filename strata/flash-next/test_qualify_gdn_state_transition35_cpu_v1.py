"""CPU-only synthetic receipt/raw/ownership tests. No Docker/GPU/model payload."""
import array,copy,json,tempfile,unittest,shlex,sys,os,time
from pathlib import Path
from unittest.mock import patch
import collect_gdn_state_transition35_v1 as collector
import qualify_gdn_state_transition35_v1 as q

class CompileTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.path=self.root/'receipt.json';self.plan=q.read(q.HERE/'gdn-state-transition35-v1-compile-run-plan.json');(self.root/'plan.snapshot.json').write_bytes((q.HERE/'gdn-state-transition35-v1-compile-run-plan.json').read_bytes());(self.root/'compile-controller.snapshot.py').write_bytes(Path(q.compiler.__file__).read_bytes());self.binary=self.root/'gdn_state_transition35_v1';self.binary.write_bytes(b'\x7fELFCPU_SYNTHETIC_NOT_EXECUTABLE')
  self.receipt={'passed':True,'errors':[],'compile_return_code':0,'post_source_unchanged':True,'container_removed':True,'actual_GPU_run':False,'plan':self.plan,'container_terminal':{'Running':False,'ExitCode':0,'OOMKilled':False,'Error':''},'controller_sha256':q.sha(q.compiler.__file__),'plan_sha256':q.PLAN_SHA,'binary':str(self.binary),'binary_sha256':q.sha(self.binary),'archive_association_limit':self.plan['archive_association_limit'],'model_math_qualified':False,'container':'CPU_mock_compile','started_epoch':1.,'finished_epoch':2.,'command':['docker','run','--name','CPU_mock_compile','--network','none','--user','1000:1000','--entrypoint','/bin/bash','--label','b70.gdn35.compile='+q.PLAN_SHA,'-v',self.plan['engine_root']+':/sdk:ro','-v',str(Path(self.plan['leaf_source']).parent)+':/leaf:ro','-v',str(self.root)+':/out',self.plan['image'],'-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1\nexec '+shlex.join(self.plan['compile_argv_inside_pinned_image'])]}
 def save(self):q.write(self.path,self.receipt)
 def gate(self):
  self.save()
  with patch.object(q.producer,'prepare',return_value=self.plan):return q.compile_binding(self.path)
 def test_synthetic_contract_positive_not_actual_SDK_or_ELF_claim(self):
  receipt,binding=self.gate();self.assertEqual(binding['archive_association_limit'],self.plan['archive_association_limit']);self.assertFalse(receipt['actual_GPU_run'])
 def test_failed_or_incomplete_compile_rejected(self):
  for key,value in [('passed',False),('compile_return_code',1),('post_source_unchanged',False),('container_removed',False),('actual_GPU_run',True),('errors',['compile failed'])]:
   prior=self.receipt[key];self.receipt[key]=value
   with self.assertRaises(ValueError):self.gate()
   self.receipt[key]=prior
 def test_wrong_terminal_oom_or_error_rejected(self):
  for key,value in [('Running',True),('ExitCode',1),('OOMKilled',True),('Error','failed')]:
   before=self.receipt['container_terminal'][key];self.receipt['container_terminal'][key]=value
   with self.assertRaises(ValueError):self.gate()
   self.receipt['container_terminal'][key]=before
 def test_changed_source_archive_or_8ELF_admission_refused(self):
  self.save()
  with patch.object(q.producer,'prepare',side_effect=ValueError('Current SDK/archive source changed')):
   with self.assertRaises(ValueError):q.compile_binding(self.path)
 def test_mutated_binary_or_compile_snapshot_refused(self):
  self.binary.write_bytes(b'\x7fELFchanged')
  with self.assertRaises(ValueError):self.gate()
 def test_missing_fresh_compile_snapshot_refused(self):
  (self.root/'compile-controller.snapshot.py').unlink()
  with self.assertRaises(FileNotFoundError):self.gate()
 def test_nested_base_entrypoint_compile_command_refused(self):
  self.receipt['command'].remove('--entrypoint')
  with self.assertRaises(ValueError):self.gate()
 def test_launch_only_binary_outputmount_card0_correct_entrypoint(self):
  with patch.object(q.os,'stat',return_value=type('CPU',(),{'st_gid':109})()):cmd=q.launch_command(self.receipt,self.root,'owned','a'*64)
  self.assertEqual(cmd[cmd.index('--entrypoint')+1],'/bin/bash');self.assertIn('ZE_AFFINITY_MASK=0',cmd);self.assertIn('SYCL_UR_TRACE=2',cmd);self.assertIn(str(self.binary.parent)+':/out:ro',cmd);self.assertTrue(any('/results' in x for x in cmd));self.assertFalse(any('/model' in x or '/sdk' in x for x in cmd));self.assertEqual(cmd[-2],'-c');self.assertIn('STRATA_VERIFY_EAGER+x',cmd[-1])
 def test_current_sourceplan_closure(self):self.assertIn('strata/flash-next/compile_gdn_state_transition35_v1.py',q.source_binding())

class OwnershipTests(unittest.TestCase):
 def test_exact_name_image_label_required(self):
  obj={'Name':'/owned','Config':{'Image':'image','Labels':{'b70.gdn35.runtime':'binding'}}};self.assertTrue(q.owned_container(obj,'owned','image','binding'))
  for field in ('name','image','binding'):
   args=['owned','image','binding'];args[('name','image','binding').index(field)]='foreign';self.assertFalse(q.owned_container(obj,*args))
 def test_final_success_requires_syntheticmath_free_health_terminal_and_newfull4(self):
  parent={'child_return_code':0,'interrupted':False,'owned_containers_terminal':True,'forced_cleanup':False,'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'post_source_unchanged':True,'errors':[],'child_terminal_epoch':2.,'post_health_finished_epoch':3.};proof={'numerical_and_teardown_passed':True};identity={'passed':True,'rows':[{}]*4,'started':4.};self.assertTrue(q.finalizable(parent,proof,identity))
  for key in ('child_return_code','owned_containers_terminal','pre_health_passed','post_health_passed','kernel_fault_gate_passed','post_source_unchanged'):
   changed=dict(parent);changed.pop(key);self.assertFalse(q.finalizable(changed,proof,identity),key)
  self.assertFalse(q.finalizable(parent,{'numerical_and_teardown_passed':False},identity));self.assertFalse(q.finalizable(parent,proof,dict(identity,started=2.5)));self.assertFalse(q.finalizable(dict(parent,errors=['diagnostic failed']),proof,identity));self.assertFalse(q.finalizable(parent,proof,dict(identity,rows=[{}]*3)))
 def test_leases_retained_and_no_nested_lock_command(self):
  source=Path(q.__file__).read_text();self.assertIn('c1.leased([0,1])',source);self.assertIn('c1.leased([0])',source);self.assertIn('pass_fds=(8,9)',source);self.assertNotIn("'gpu-run','--card'",source);self.assertIn("require(parent['owned_containers_terminal'],'Posthealth",source)

class ParentFailureLifecycleTests(unittest.TestCase):
 def test_nonzero_leaf_retains_posthealth_full4_and_removes_only_owned_terminal(self):
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);out=root/'run';image='CPU_mock_image';binding={'compile_receipt_sha256':'CPU_mock_hash','archive_association_limit':'CPU fixture; no provenance claim'};receipt={'plan':{'image':image}};container='b70-gdn35-runtime-'+str(os.getpid());live=[False];rm=[];health=[]
   class Process:
    def __init__(self,cmd):
     self.cmd=cmd;self.pid=700;self.returncode=1 if cmd[:2]==['docker','run'] else 0
     if cmd[:2]==['docker','run']:live[0]=True
     if cmd[:2]==['docker','rm']:rm.append(cmd[-1]);live[0]=False
     if cmd[0].endswith(('xpu_health_strict.sh','xpu-collective-health')):health.append(cmd)
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode
    def terminate(self):self.returncode=1
   def check_output(cmd,**kwargs):
    return 'CPU_fixture_container\n' if cmd[:3]==['docker','ps','-aq'] and cmd[-1]=='name=^/'+container+'$' and live[0] else ''
   def identity(lock_path,lock,shards,path,after):
    value={'passed':True,'rows':[{}]*4,'started':time.time(),'finished':time.time(),'CPU_mock':True};q.write(path,value);self.assertGreaterEqual(value['started'],after);return value
   obj={'Name':'/'+container,'Config':{'Image':image,'Labels':{'b70.gdn35.runtime':'CPU_mock_hash'}},'State':{'Running':False,'ExitCode':1,'OOMKilled':False,'Error':''}}
   args=['parent','--compile-receipt',str(root/'CPUmockreceipt'),'--output',str(out),'--leased']
   with patch.object(sys,'argv',args),patch.object(q.c1,'leased'),patch.object(q,'source_binding',return_value={}),patch.object(q,'compile_binding',return_value=(receipt,binding)),patch.object(q,'launch_command',return_value=['docker','run','CPU_mock_leaf']),patch.object(q.subprocess,'Popen',side_effect=lambda cmd,**kwargs:Process(cmd)),patch.object(q.subprocess,'check_output',side_effect=check_output),patch.object(q.c1,'inspected',return_value=obj),patch.object(q,'preserve_source_pages',return_value={'passed':True,'CPU_mock':True}),patch.object(q,'full_buffered_identity',side_effect=identity) as scan,patch.object(q,'raw_proofs',return_value={'numerical_and_teardown_passed':False,'synthetic_component_passed':False,'full_model_math_qualified':False}):
    self.assertEqual(q.main(),1);self.assertEqual(scan.call_count,1)
   result=q.read(out/'parent-qualification.json');self.assertFalse(result['passed']);self.assertTrue(result['owned_containers_terminal']);self.assertEqual(rm,[container]);self.assertEqual(len(health),4);self.assertTrue(result['post_health_passed']);self.assertIn('known_pages_before_hash',result);self.assertIn('known_pages_after_hash',result);self.assertFalse(result['full_model_math_qualified']);self.assertTrue(result['errors'])
 def test_incomplete_compile_never_launches_GPU_health_or_scans_model(self):
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);out=root/'fail';args=['parent','--compile-receipt',str(root/'missing'),'--output',str(out),'--leased']
   with patch.object(sys,'argv',args),patch.object(q.c1,'leased'),patch.object(q,'source_binding',return_value={}),patch.object(q,'compile_binding',side_effect=ValueError('Incomplete compile')),patch.object(q.subprocess,'check_output',return_value=''),patch.object(q.subprocess,'Popen',side_effect=AssertionError('No actual GPU/Docker')) as launch,patch.object(q,'full_buffered_identity',side_effect=AssertionError('No model payload')) as scan:
    self.assertEqual(q.main(),1);launch.assert_not_called();scan.assert_not_called()

class RawTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.raw=self.root/'raw-new';self.raw.mkdir();self.log=self.root/'leaf.log';self.lines=[]
  for key,value in [('C',2),('V',4),('ST',8),('CV',6)]:
   mock=patch.object(collector,key,value);mock.start();self.addCleanup(mock.stop)
  blob=lambda v,n:array.array('f',[v]).tobytes()*n
  def save(name,raw):(self.raw/(name+'.f32')).write_bytes(raw)
  states=[bytes(32)]+[blob(n,8) for n in (1,2,3)];convs=[bytes(24)]+[blob(n,6) for n in (1,2,3)];h=blob(4,6);y=b''.join(blob(n+4,4) for n in range(3));save('t1-normalized-qkv',h);save('t1-output',y)
  for n in (1,2,3):save('t1-state-after-'+str(n),states[n]);save('t1-conv-after-'+str(n),convs[n])
  for keep in range(3):
   label='t2-keep'+str(keep);parts=[('forward-state-unmodified','forward-state',8,states[0]),('forward-conv-unmodified','forward-conv',6,convs[0]),('forward-normalized-qkv','forward-normalized-qkv',4,h[:16]),('forward-output','forward-output',8,y[:32]),('accepted-state','committed-state',8,states[keep]),('accepted-conv','committed-conv',6,convs[keep]),('carry-state','carry-state',8,states[keep+1]),('carry-conv','carry-conv',6,convs[keep+1]),('carry-output','carry-output',4,y[keep*16:(keep+1)*16])]
   for component,suffix,n,raw in parts:save(label+'-'+suffix,raw);self.lines.append('GDN35_COMPARE case=%s component=%s words=%d differing=0 first=%d max_abs=0 bitwise=1'%(label,component,n,n))
  save('negative-modified-state-output',blob(99,4));self.lines+=['GDN35_NEGATIVE modified_state_output_detected=1','GDN35_RESULT comparisons=27 failures=0 synthetic_component_passed=1 model_math_qualified=0 all_owned_allocations_freed=1','<--- urUSMDeviceAlloc(.hContext = 0x1, .size = 32, .ppMem = 0x3 (0x10)) -> UR_RESULT_SUCCESS;','<--- urUSMFree(.hContext = 0x1, .pMem = 0x10) -> UR_RESULT_SUCCESS;'];self.log.write_text('\n'.join(self.lines)+'\n')
 def test_tiny_actual_collector27_negative_and_USM_ledger(self):
  result=q.raw_proofs(self.root);self.assertTrue(result['synthetic_component_passed']);self.assertTrue(result['logical_free_passed']);self.assertFalse(result['full_model_math_qualified']);self.assertFalse(result['real_row3_inputs_observed']);trace=q.read(self.root/'usm-logical-free-proof.json');self.assertTrue(all(trace['negative_controls'].values()));self.assertFalse(trace['physical_backing_release_proven'])
 def test_missing_free_keeps_numerical_positive_but_fails_parent(self):
  self.log.write_text('\n'.join(self.lines[:-1])+'\n');result=q.raw_proofs(self.root);self.assertTrue(result['synthetic_component_passed']);self.assertFalse(result['logical_free_passed']);self.assertFalse(result['numerical_and_teardown_passed'])
 def test_synthetic_numeric_false_remains_diagnostic_failure(self):
  path=self.raw/'t2-keep2-carry-output.f32';path.write_bytes(array.array('f',[123]*4).tobytes());self.lines=[line.replace('component=carry-output words=4 differing=0 first=4 max_abs=0 bitwise=1','component=carry-output words=4 differing=4 first=0 max_abs=100 bitwise=0') if 'case=t2-keep2' in line else line for line in self.lines];self.lines=[line.replace('failures=0 synthetic_component_passed=1','failures=1 synthetic_component_passed=0') for line in self.lines];self.log.write_text('\n'.join(self.lines)+'\n');result=q.raw_proofs(self.root);self.assertFalse(result['synthetic_component_passed']);self.assertFalse(result['numerical_and_teardown_passed']);self.assertFalse(result['full_model_math_qualified'])

if __name__=='__main__':unittest.main()
