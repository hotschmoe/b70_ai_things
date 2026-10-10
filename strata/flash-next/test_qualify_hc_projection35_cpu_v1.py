"""CPU-only synthetic receipt/raw/ownership tests. No Docker/GPU/model payload."""
import array,copy,json,tempfile,unittest,shlex,sys,os,time
from pathlib import Path
from unittest.mock import patch
import qualify_hc_projection35_v1 as q

class CompileTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.path=self.root/'receipt.json';self.plan=q.read(q.HERE/'hc-projection35-leaf-compile-plan-v1.json');(self.root/'plan.snapshot.json').write_bytes((q.HERE/'hc-projection35-leaf-compile-plan-v1.json').read_bytes());(self.root/'compile-controller.snapshot.py').write_bytes(Path(q.compiler.__file__).read_bytes());self.binary=self.root/'hc_projection_arithmetic35_gpu_v1';self.binary.write_bytes(b'\x7fELFCPU_SYNTHETIC_NOT_EXECUTABLE');(self.root/'compile.log').write_text('CPU mock compiler log')
  self.receipt={'passed':True,'errors':[],'compile_return_code':0,'post_source_unchanged':True,'container_removed':True,'actual_GPU_run':False,'plan':self.plan,'container_terminal':{'Running':False,'ExitCode':0,'OOMKilled':False,'Error':''},'compile_log_sha256':q.sha(self.root/'compile.log'),'controller_sha256':q.sha(q.compiler.__file__),'plan_sha256':q.PLAN_SHA,'binary':str(self.binary),'binary_sha256':q.sha(self.binary),'archive_association_limit':self.plan['archive_association_limit'],'model_math_qualified':False,'container':'CPU_mock_compile','started_epoch':1.,'finished_epoch':2.,'command':['docker','run','--name','CPU_mock_compile','--network','none','--user','1000:1000','--entrypoint','/bin/bash','--label','b70.hc35.compile='+q.PLAN_SHA,'-v',self.plan['engine_root']+':/sdk:ro','-v',str(Path(self.plan['leaf_source']).parent)+':/leaf:ro','-v',str(self.root)+':/out',self.plan['image'],'-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1\nexec '+shlex.join(self.plan['compile_argv_inside_pinned_image'])]}
 def save(self):q.write(self.path,self.receipt)
 def gate(self):
  self.save()
  with patch.object(q.producer,'prepare',return_value=self.plan):return q.compile_binding(self.path)
 def test_synthetic_contract_positive_not_actual_SDK_or_ELF_claim(self):
  receipt,binding=self.gate();self.assertEqual(binding['archive_association_limit'],self.plan['archive_association_limit']);self.assertFalse(receipt['actual_GPU_run'])
 def test_actual_current_CPU_metadata_matches_plan(self):
  self.assertEqual(q.producer.prepare(Path(self.plan['engine_root'])),self.plan)
 def test_mutated_compiler_log_refused(self):
  (self.root/'compile.log').write_text('mutated CPU compiler log')
  with self.assertRaises(ValueError):self.gate()
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
  with patch.object(q.os,'stat',return_value=type('CPU',(),{'st_gid':109})()):cmd=q.launch_command(self.receipt,self.root,'owned','a'*64,self.root/'package')
  self.assertEqual(cmd[cmd.index('--entrypoint')+1],'/bin/bash');self.assertIn('ZE_AFFINITY_MASK=0',cmd);self.assertIn('SYCL_UR_TRACE=2',cmd);self.assertIn(str(self.binary.parent)+':/out:ro',cmd);self.assertTrue(any('/results' in x for x in cmd));self.assertFalse(any('/model' in x or '/sdk' in x for x in cmd));self.assertIn(str(self.root/'package/inputs')+':/inputs:ro',cmd);self.assertFalse(any('/expected' in x for x in cmd));self.assertEqual(cmd[-2],'-c');self.assertIn('STRATA_VERIFY_EAGER+x',cmd[-1])
 def test_current_sourceplan_closure(self):self.assertIn('strata/flash-next/compile_hc_projection35_v1.py',q.source_binding())

class OwnershipTests(unittest.TestCase):
 def test_exact_name_image_label_required(self):
  obj={'Name':'/owned','Config':{'Image':'image','Labels':{'b70.hc35.runtime':'binding'}}};self.assertTrue(q.owned_container(obj,'owned','image','binding'))
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
   root=Path(name);(root/'manifest.json').write_text('{}');out=root/'run';image='CPU_mock_image';binding={'compile_receipt_sha256':'CPU_mock_hash','archive_association_limit':'CPU fixture; no provenance claim'};receipt={'plan':{'image':image}};container='b70-hc35-runtime-'+str(os.getpid());live=[False];rm=[];health=[]
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
   obj={'Name':'/'+container,'Config':{'Image':image,'Labels':{'b70.hc35.runtime':'CPU_mock_hash'}},'State':{'Running':False,'ExitCode':1,'OOMKilled':False,'Error':''}}
   args=['parent','--compile-receipt',str(root/'CPUmockreceipt'),'--inputs',str(root),'--output',str(out),'--leased']
   with patch.object(sys,'argv',args),patch.object(q.c1,'leased'),patch.object(q,'source_binding',return_value={}),patch.object(q.fixture,'input_binding',return_value={}),patch.object(q,'compile_binding',return_value=(receipt,binding)),patch.object(q,'launch_command',return_value=['docker','run','CPU_mock_leaf']),patch.object(q.subprocess,'Popen',side_effect=lambda cmd,**kwargs:Process(cmd)),patch.object(q.subprocess,'check_output',side_effect=check_output),patch.object(q.c1,'inspected',return_value=obj),patch.object(q,'preserve_source_pages',return_value={'passed':True,'CPU_mock':True}),patch.object(q,'full_buffered_identity',side_effect=identity) as scan,patch.object(q,'raw_proofs',return_value={'numerical_and_teardown_passed':False,'synthetic_component_passed':False,'full_model_math_qualified':False}):
    self.assertEqual(q.main(),1);self.assertEqual(scan.call_count,1)
   result=q.read(out/'parent-qualification.json');self.assertFalse(result['passed']);self.assertTrue(result['owned_containers_terminal']);self.assertEqual(rm,[container]);self.assertEqual(len(health),4);self.assertTrue(result['post_health_passed']);self.assertIn('known_pages_before_hash',result);self.assertIn('known_pages_after_hash',result);self.assertFalse(result['full_model_math_qualified']);self.assertTrue(result['errors'])
 def test_incomplete_compile_never_launches_GPU_health_or_scans_model(self):
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);(root/'manifest.json').write_text('{}');out=root/'fail';args=['parent','--compile-receipt',str(root/'missing'),'--inputs',str(root),'--output',str(out),'--leased']
   with patch.object(sys,'argv',args),patch.object(q.c1,'leased'),patch.object(q,'source_binding',return_value={}),patch.object(q.fixture,'input_binding',return_value={}),patch.object(q,'compile_binding',side_effect=ValueError('Incomplete compile')),patch.object(q.subprocess,'check_output',return_value=''),patch.object(q.subprocess,'Popen',side_effect=AssertionError('No actual GPU/Docker')) as launch,patch.object(q,'full_buffered_identity',side_effect=AssertionError('No model payload')) as scan:
    self.assertEqual(q.main(),1);launch.assert_not_called();scan.assert_not_called()

class RawCollectorTests(unittest.TestCase):
 def test_public_collector_and_free_trace_required(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'leaf.log').write_text('CPU mock raw log')
   raw={'cases':[{'words':3}]*12+[{'words':2}],'negative_controls':[{},{}],'numeric_bitwise_passed':True}
   trace={'passed':True}
   with patch.object(q.fixture,'collect',return_value=raw) as collect,patch.object(q,'parse_trace',return_value=trace),patch.object(q,'negative_controls',return_value={'CPUnegative':True}):
    result=q.raw_proofs(root,root/'inputs');self.assertTrue(result['numerical_and_teardown_passed']);collect.assert_called_once_with(root/'inputs',root/'raw-new',root/'leaf.log')
   with patch.object(q.fixture,'collect',return_value=dict(raw,numeric_bitwise_passed=False)),patch.object(q,'parse_trace',return_value={'passed':True}),patch.object(q,'negative_controls',return_value={'CPUnegative':True}):self.assertFalse(q.raw_proofs(root,root/'inputs')['numerical_and_teardown_passed'])
 def test_printed_result_without_raw_cannot_qualify(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'leaf.log').write_text('HC35_PROJECTION_RESULT execution_completed=1')
   with patch.object(q.fixture,'collect',side_effect=ValueError('missing actual raw')):
    with self.assertRaises(ValueError):q.raw_proofs(root,root/'inputs')

if __name__=='__main__':unittest.main()
