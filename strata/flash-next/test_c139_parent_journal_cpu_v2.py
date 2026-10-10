"""Original journal/command/error controls; fake processes, no GPU/model IO."""
import ast,copy,json,subprocess,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import c139_journal_binding_v2 as j
import qualify_c1_serving_combined_v139_v2 as p
class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.proof={'started_epoch':10.,'pre_health_finished_epoch':11.,'launch_started_epoch':14.,'terminal_epoch':20.,'post_health_finished_epoch':22.,'identity_started_epoch':25.,'journal_binding_epoch':27.,'kernel_journal_receipts':{'active':[]}}
  for stage,start,finish in [('pre',12.,13.),('active',15.,16.),('post',23.,24.)]:
   label=stage+'-kernel-journal'+('-1' if stage=='active' else '');argv=['journalctl','-k','--since','@10','--no-pager'];row=self.row(label+'.log',argv,start,finish,True)
   if stage=='active':self.proof['kernel_journal_receipts']['active'].append(row)
   else:self.proof['kernel_journal_receipts'][stage]=row;self.save(stage)
 def row(self,label,argv,start,finish,journal=False):
  path=self.root/label;path.write_text('CPU_SYNTHETIC_COMPLETE\n');cmd=self.root/(label[:-4]+'.command.json' if journal else label+'.command.json');j.c.write(cmd,argv)
  return {'path':str(path),'sha256':j.c.sha(path),'stdout_sha256':j.c.sha(path),'command':argv,'command_file_sha256':j.c.sha(cmd),'return_code':0,'error':None,'started_epoch':start,'finished_epoch':finish}
 def save(self,stage):j.c.write(self.root/(stage+'-kernel-receipt.json'),self.proof['kernel_journal_receipts'][stage])
 def test_original_complete_pre_active_post(self):self.assertEqual(j.journal_binding(self.root,self.proof),self.proof['kernel_journal_receipts'])
 def test_rczero_with_timeout_error_refused(self):
  row=self.proof['kernel_journal_receipts']['post'];row['error']='TimeoutExpired CPU';self.save('post')
  with self.assertRaises(ValueError):j.journal_binding(self.root,self.proof)
 def test_changed_log_or_sidecar_refused(self):
  (self.root/'active-kernel-journal-1.log').write_text('CPU_CHANGED\n')
  with self.assertRaises(ValueError):j.journal_binding(self.root,self.proof)
 def test_missing_prelaunch_journal_refused(self):
  self.proof['kernel_journal_receipts'].pop('pre')
  with self.assertRaises(ValueError):j.journal_binding(self.root,self.proof)
 def test_missing_readiness_journal_refused(self):
  self.proof['kernel_journal_receipts']['active']=[]
  with self.assertRaises(ValueError):j.journal_binding(self.root,self.proof)
 def test_beforehealth_afterlaunch_beforeterminal_before4_refused(self):
  original=copy.deepcopy(self.proof)
  for stage,key,value in [('pre','started_epoch',10.),('pre','finished_epoch',15.),('post','started_epoch',19.),('post','finished_epoch',26.)]:
   self.proof=copy.deepcopy(original);self.proof['kernel_journal_receipts'][stage][key]=value;self.save(stage)
   with self.subTest(stage=stage,key=key),self.assertRaises(ValueError):j.journal_binding(self.root,self.proof)
 def test_nonfinite_global_or_command_time_refused(self):
  original=copy.deepcopy(self.proof)
  for key in ('started_epoch','launch_started_epoch','terminal_epoch','pre_health_finished_epoch','post_health_finished_epoch','identity_started_epoch','journal_binding_epoch'):
   self.proof=copy.deepcopy(original);self.proof[key]=float('nan')
   with self.subTest(key=key),self.assertRaises(ValueError):j.journal_binding(self.root,self.proof)
 def test_wrong_actual_argv_even_updatedsha_refused(self):
  row=self.proof['kernel_journal_receipts']['pre'];row['command'][-1]='CPU_CHANGED';cmd=self.root/'pre-kernel-journal.command.json';j.c.write(cmd,row['command']);row['command_file_sha256']=j.c.sha(cmd);self.save('pre')
  with self.assertRaises(ValueError):j.journal_binding(self.root,self.proof)
 def test_original_health_rows_exact_positive_and_rcerror_negative(self):
  commands=[[str(j.c.REPO/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',j.HEALTH],[str(j.c.REPO/'bin/xpu-collective-health'),'--img',j.HEALTH,'--p2p','0','--timeout','180']]
  rows=[self.row('pre-'+label,argv,10.2+i*.2,10.3+i*.2) for i,(argv,label) in enumerate(zip(commands,('health.log','collective.log')))]
  h={'passed':True,'cards':[0,1],'health_image':j.HEALTH,'started_epoch':10.1,'finished_epoch':11.,'files':rows};path=self.root/'pre-health.json';j.c.write(path,h);self.proof['pre_health_sha256']=j.c.sha(path);self.assertEqual(j.health_binding(self.root,self.proof,'pre'),h)
  rows[0]['error']='CPU timeout despite rc0';j.c.write(path,h);self.proof['pre_health_sha256']=j.c.sha(path)
  with self.assertRaises(ValueError):j.health_binding(self.root,self.proof,'pre')
 def nested_run(self):
  tree=ast.parse(Path(p.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');node=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='run');code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'CPU_SOURCE_EXTRACTED_REAL_PARENT','exec');scope={'out':self.root,'time':time,'c1':p.c1,'subprocess':p.subprocess,'stopped':False,'page_guard':lambda _:None};exec(code,scope);return scope['run']
 def test_actual_nested_parent_run_persists_original_return(self):
  run=self.nested_run()
  def fake(argv,stdout,**kw):stdout.write('CPU_ACTUAL_MOCK_STDOUT\n');return subprocess.CompletedProcess(argv,0)
  with patch.object(p.subprocess,'run',side_effect=fake):row=run(['CPU_COMMAND'],'CPU.log')
  self.assertEqual(row,p.c1.read(self.root/'CPU.log.receipt.json'));self.assertIsNone(row['error']);self.assertEqual(row['return_code'],0);self.assertEqual(row['sha256'],p.c1.sha(self.root/'CPU.log'))
 def test_actual_nested_parent_timeout_records_failure_before_raise(self):
  run=self.nested_run()
  with patch.object(p.subprocess,'run',side_effect=subprocess.TimeoutExpired(['CPU_COMMAND'],1)):
   with self.assertRaises(subprocess.TimeoutExpired):run(['CPU_COMMAND'],'CPU-fail.log')
  row=p.c1.read(self.root/'CPU-fail.log.receipt.json');self.assertIsNone(row['return_code']);self.assertIn('TimeoutExpired',row['error']);self.assertEqual(row['sha256'],p.c1.sha(self.root/'CPU-fail.log'))
 def test_new_parent_rejects_old_generation_before_other_proof_reads(self):
  with patch.object(p.c1,'read',return_value={'passed':True,'teardown_passed':True,'post_health_passed':True,'post_full4_source_qualified':True,'c1_parent_generation':137}):
   with self.assertRaises(AssertionError):p.validate_final_source_proof(self.root,{})
 def original_parent(self):
  proof=self.proof;identity={'started':25.,'finished':26.};proof.update(parent_controller_sha256='CPU_PARENT',engine_receipt_sha256='CPU_ENGINE',prepared_sha256='CPU_PREPARED',identity={'sha256':'CPU_FULL4'})
  parent={'parent_generation':1392,'parent_controller_sha256':'CPU_PARENT','started_epoch':proof['started_epoch'],'launch_started_epoch':proof['launch_started_epoch'],'pre_health_finished_epoch':proof['pre_health_finished_epoch'],'post_health_finished_epoch':proof['post_health_finished_epoch'],'owned_terminal_epoch':proof['terminal_epoch'],'proof_binding_epoch':proof['journal_binding_epoch'],'kernel_journal_receipts':copy.deepcopy(proof['kernel_journal_receipts']),'engine_receipt_sha256':'CPU_ENGINE','prepared_sha256':'CPU_PREPARED','post_model_identity':{'started':25.,'finished':26.,'sha256':'CPU_FULL4','passed':True}}
  path=self.root/'parent-before-proof.json';j.c.write(path,parent);proof['parent_before_proof_sha256']=j.c.sha(path);return parent,identity
 def test_actual_parent_rows_and_identity_exact_positive(self):
  parent,identity=self.original_parent();self.assertEqual(j.parent_binding(self.root,self.proof,identity,parent),parent)
 def test_actual_identity_before_postjournal_cannot_borrow_forward_proof_epoch(self):
  parent,identity=self.original_parent();identity['started']=23.;self.assertEqual(self.proof['identity_started_epoch'],25.)
  with self.assertRaises(ValueError):j.parent_binding(self.root,self.proof,identity,parent)
 def test_proof_forward_identity_epoch_refused(self):
  parent,identity=self.original_parent();self.proof['identity_started_epoch']=26.
  with self.assertRaises(ValueError):j.parent_binding(self.root,self.proof,identity,parent)
 def test_each_original_journal_launch_health_epoch_mustmatch_proof(self):
  parent,identity=self.original_parent();original=copy.deepcopy(self.proof)
  for key in ('started_epoch','launch_started_epoch','pre_health_finished_epoch','post_health_finished_epoch','terminal_epoch','journal_binding_epoch'):
   self.proof=copy.deepcopy(original);self.proof[key]+=.1
   with self.subTest(key=key),self.assertRaises(ValueError):j.parent_binding(self.root,self.proof,identity,parent)
  self.proof=copy.deepcopy(original);self.proof['kernel_journal_receipts']['active'][0]['return_code']=1
  with self.assertRaises(ValueError):j.parent_binding(self.root,self.proof,identity,parent)
 def test_final_parent_cannot_differ_from_original_rows_or_identity(self):
  parent,identity=self.original_parent()
  for key in ('launch_started_epoch','pre_health_finished_epoch','proof_binding_epoch'):
   changed=copy.deepcopy(parent);changed[key]+=.1
   with self.subTest(key=key),self.assertRaises(ValueError):j.parent_binding(self.root,self.proof,identity,changed)
  changed=copy.deepcopy(parent);changed['post_model_identity']['started']+=.1
  with self.assertRaises(ValueError):j.parent_binding(self.root,self.proof,identity,changed)
 def test_health_started_not_before_parent_or_terminal(self):
  for stage,start,finish in [('pre',9.,11.),('post',19.,22.)]:
   commands=[[str(j.c.REPO/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',j.HEALTH],[str(j.c.REPO/'bin/xpu-collective-health'),'--img',j.HEALTH,'--p2p','0','--timeout','180']];rows=[self.row(stage+'-'+label,argv,start+.1+i*.1,start+.15+i*.1) for i,(argv,label) in enumerate(zip(commands,('health.log','collective.log')))];path=self.root/(stage+'-health.json');j.c.write(path,{'passed':True,'cards':[0,1],'health_image':j.HEALTH,'started_epoch':start,'finished_epoch':finish,'files':rows});self.proof[stage+'_health_sha256']=j.c.sha(path)
   with self.subTest(stage=stage),self.assertRaises(ValueError):j.health_binding(self.root,self.proof,stage)
 def test_readonly_closure_checks_actual_identity_and_finalparent(self):
  source=Path(p.__file__).read_text();self.assertIn('parent_binding(directory,proof,identity)',source);self.assertIn('parent_binding(directory,proof,identity,parent)',source);self.assertIn("c1.write(out/'parent-before-proof.json',result)",source)
 def test_actual_initial_result_constructor_saves_reader_required_start(self):
  from types import SimpleNamespace
  tree=ast.parse(Path(p.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');assignment=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets));actual=eval(compile(ast.fix_missing_locations(ast.Expression(body=assignment.value)),'CPU_REAL_PRODUCER_RESULT','eval'),{'started':10.5,'prepared':{'engine_receipt_sha256':'CPU_ENGINE'},'c1':SimpleNamespace(sha=lambda _: 'CPU_SOURCE'),'Path':Path,'__file__':p.__file__,'CONTROLLER_SHA':'CPU_CONTROLLER','WATCHDOG_SHA':'CPU_WATCHDOG','out':self.root})
  self.assertEqual(actual['started_epoch'],10.5);self.assertEqual(actual['parent_generation'],1392);self.assertFalse(actual['passed'])
 def test_frozenV3_actual_constructor_reproduces_missingfield(self):
  import qualify_c1_serving_combined_v137_v3 as frozen
  tree=ast.parse(Path(frozen.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');assignment=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets));self.assertNotIn('started_epoch',{keyword.arg for keyword in assignment.value.keywords})
if __name__=='__main__':unittest.main()
