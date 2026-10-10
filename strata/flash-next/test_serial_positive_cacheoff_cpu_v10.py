"""Tiny metadata/control tests; all external model/GPU/health boundaries mocked."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import batch_serial_positive_cacheoff_v10 as q
from audit_batch_numerical_suite_v7 import snapshot_plan_join

class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='positive-serial-CPU-');self.root=Path(self.tmp.name);(self.root/'child').mkdir();self.external=self.root/'actual-plan.json';self.plan={'schema':6,'harness_generation':9,'api_cache_positive_generation':2,'kind':'api','diagnostic':1,'slots':2,'driver_sha256':q.sha(Path(q.positive.__file__)),'API_source_plan_sha256':q.sha(q.positive.SOURCE_PLAN),'research_alias':'CPU_SYNTHETIC_ALIAS','registry_binding':{'CPU_ONLY':True},'prepared':'CPU_ONLY'};self.child={'schema':6,'harness_generation':9,'api_cache_positive_generation':2,'collection_and_teardown_passed':True,'error':None,'actual_cached_state_handoff_qualified':False,'full_model_math_qualified':False,'actual_last_live_handoffs':[{'CPU_METADATA_ONLY':True}]};self.parent={'passed':True,'scoped_collection_or_serial_arm_qualified':True,'child_return_code':0,'owned_containers_terminal':True,'forced_cleanup':False,'interrupted':False,'errors':[],'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'source_guard_generation':3,'wrapper_sha256':q.sha(q.HERE/'qualify_batch_api_cache_positive_v2.py'),'controller_sha256':q.sha(Path(q.positive.__file__)),'plan':str(self.external)};self.save()
 def tearDown(self):self.tmp.cleanup()
 def save(self):
  for path in (self.external,self.root/'input-plan.snapshot.json',self.root/'child/plan.snapshot.json'):q.write(path,self.plan)
  self.parent['plan_sha256']=q.sha(self.external);self.child['plan_sha256']=q.sha(self.external);q.write(self.root/'child/report.json',self.child);self.parent['child_report_sha256']=q.sha(self.root/'child/report.json');q.write(self.root/'parent-qualification.json',self.parent)
 def test_new_metadata_header_admitted_without_external_boundaries(self):
  with patch.object(q.positive.contract,'registry_gate',return_value={'CPU_ONLY':True}),patch.object(q.proof,'genuine_baseline') as model,patch.object(q.c1,'leased') as gpu:p,s,c=q.collector_header(self.root)
  self.assertEqual(s,self.plan);model.assert_not_called();gpu.assert_not_called()
 def test_missing_childSHA_before_fullmodel_or_GPU(self):
  self.child.pop('plan_sha256');q.write(self.root/'child/report.json',self.child);self.parent['child_report_sha256']=q.sha(self.root/'child/report.json');q.write(self.root/'parent-qualification.json',self.parent)
  with patch.object(q.proof,'genuine_baseline') as model,patch.object(q.positive_reader,'finalized_binding') as raw,patch.object(q.c1,'leased') as gpu:
   with self.assertRaises(ValueError):q.prepare(SimpleNamespace(batch_parent=self.root,prepared=Path('CPU_ONLY')))
   model.assert_not_called();raw.assert_not_called();gpu.assert_not_called()
 def test_old_or_wrong_collector_before_expensive_join(self):
  for key,value in [('schema',4),('harness_generation',7),('api_cache_positive_generation',1),('kind','native'),('diagnostic',0),('slots',4)]:
   original=copy.deepcopy(self.plan);self.plan[key]=value;self.save()
   with self.subTest(key=key),patch.object(q.positive_reader,'finalized_binding') as raw,patch.object(q.proof,'genuine_baseline') as model:
    with self.assertRaises(ValueError):q.collector_binding(self.root)
    raw.assert_not_called();model.assert_not_called()
   self.plan=original
 def test_wrong_actual_snapshot_or_producer_source_refused(self):
  q.write(self.root/'child/plan.snapshot.json',dict(self.plan,slots=6))
  with self.assertRaises(ValueError):q.collector_header(self.root)
  self.save();self.parent['controller_sha256']='0'*64;q.write(self.root/'parent-qualification.json',self.parent)
  with self.assertRaises(ValueError):q.collector_header(self.root)
 def test_old_failed_or_premature_math_parent_not_transferred(self):
  for key,value in [('passed',False),('forced_cleanup',True),('source_guard_generation',2)]:
   old=self.parent[key];self.parent[key]=value;self.save()
   with self.subTest(key=key),self.assertRaises(ValueError):q.collector_header(self.root)
   self.parent[key]=old
  self.child['actual_cached_state_handoff_qualified']=True;self.save()
  with self.assertRaises(ValueError):q.collector_header(self.root)
 def test_current_registry_entry_association_required_before_raw(self):
  with patch.object(q.positive.contract,'registry_gate',side_effect=ValueError('CPU changedregistry')),patch.object(q.positive_reader,'finalized_binding') as raw:
   with self.assertRaises(ValueError):q.collector_binding(self.root)
   raw.assert_not_called()
 def test_same_cacheoff_config_and_protocol_preserved(self):
  source={'args':['--batch','2','--prompt-cache','3','--conversation-cache-mib','0','--suffix-draft','0','--lookup-chain','0'],'env':{'STRATA_BATCH_FULL_STATE_CHAIN':'1','STRATA_BATCH_PUBLIC_PREFIX':'1','STRATA_PLE_INPUT33':'0','STRATA_PREFIX30':'0'}};before=copy.deepcopy(source);args,env=q.prepared_config(source);control=dict(args=args,env=env);actual_args,actual_env=q.serial.config(control);self.assertEqual(source,before);self.assertEqual(actual_args[actual_args.index('--batch')+1],'0');self.assertEqual(actual_args[actual_args.index('--prompt-cache')+1],'0');self.assertEqual(actual_env['STRATA_BATCH_FULL_STATE_CHAIN'],'0');self.assertEqual(actual_env['STRATA_FIDELITY_DIAG'],'1')
  control['env']['STRATA_BATCH_PUBLIC_PREFIX']='1'
  with self.assertRaises(ValueError):q.serial.config(control)
 def test_terminal_realSHA_counts_sealed_before_report(self):
  out=self.root/'new-child';out.mkdir();plan={'serial_group_job_count':2,'CPU_SYNTHETIC':True};q.write(out/'plan.snapshot.json',plan);q.write(self.root/'input-plan.snapshot.json',plan);q.write(self.external,plan);parent={'plan':str(self.external),'plan_sha256':q.sha(self.external)};result={'passed':True,'comparisons':{str(i):{'CPU_BITWISE_CONTROL':True} for i in range(98)}}
  with patch.object(q.proof,'artifact_bindings',return_value={'CPU_TINY':True}):report=q.completed_report(plan,out,result)
  # Same strict snapshot join over the completed producer output.
  (self.root/'child/plan.snapshot.json').write_bytes((out/'plan.snapshot.json').read_bytes());snapshot_plan_join(self.root,parent,plan,report);self.assertEqual(report['schema'],7);self.assertEqual(report['actual_matched_full49_vector_pairs'],98);self.assertFalse(report['actual_cached_state_handoff_qualified']);self.assertFalse(report['cache_qualification_granted'])
  with patch.object(q.proof,'artifact_bindings') as seal:
   with self.assertRaises(ValueError):q.completed_report(plan,out,{'passed':True,'comparisons':{'onlyone':{}}})
   seal.assert_not_called()
 def test_run_with_mock_external_boundary_preserves_completed_sourceSHA(self):
  out=self.root/'serial-run';plan={'batch_parent':str(self.root),'group_index':0,'serial_group_job_count':1};q.write(self.external,plan)
  def native(actual,batch,output,health,group):output.mkdir();q.write(output/'plan.snapshot.json',actual);return {'passed':True,'comparisons':{str(i):{'CPU_ONLY':True} for i in range(49)}}
  with patch.object(q,'manifest_binding',return_value={'CPU_CURRENT_PROOF':True}),patch.object(q.c1,'leased') as lease,patch.object(q.serial,'run',native),patch.object(q.proof,'artifact_bindings',return_value={'CPU_ONLY':True}):self.assertEqual(q.run(SimpleNamespace(plan=self.external,output=out,pre_health=Path('/CPU'))),0)
  report=q.read(out/'report.json');self.assertEqual(report['plan_sha256'],q.sha(out/'plan.snapshot.json'));lease.assert_called_once_with([0,1])
 def test_unchanged_runtime_writes_snapshot_before_fullcommand_and_seal(self):
  import run_batch_serial_controls_v7 as runtime
  import time,struct
  out=self.root/'deep-serial';batch=self.root/'source-batch';batch.mkdir();(batch/'captures').mkdir();job={'rid':7,'role':'solo_migration','ids':[11,12],'max_new':1};q.write(batch/'serial-jobs.json',{'jobs':[job]});lines=[]
  for layer in range(-1,48):
   name='CPU-source-'+str(layer)+'.bin';(batch/'captures'/name).write_bytes(struct.pack('<f',1));lines.append('SBF vector rid=7 layer='+str(layer)+' phase=solo_migration_'+('head' if layer==-1 else 'residual')+' file=/results/captures/'+name)
  (batch/'engine.combined.log').write_text('\n'.join(lines)+'\n');health=self.root/'CPU-health.json';q.write(health,{'passed':True,'cards':[0],'finished_epoch':time.time()});plan={'lane':'source35','cards':[0],'prepared':'CPU_ONLY','engine_root':'/CPU_SOURCE_SDK','pack':'/CPU_PACK','image':'CPU_ONLY_IMAGE','args':['--batch','2','--prompt-cache','0','--conversation-cache-mib','0','--suffix-draft','0','--lookup-chain','0'],'env':{'STRATA_BATCH_FULL_STATE_CHAIN':'0','STRATA_BATCH_PUBLIC_PREFIX':'0','STRATA_PLE_INPUT33':'0','STRATA_PREFIX30':'0'},'group_index':0,'serial_group_job_count':1};seen=[]
  fake=SimpleNamespace(leased=lambda cards:None,read=q.read,write=q.write,sha=q.sha,require=q.require,inspected=lambda name:{'State':{'Running':False,'ExitCode':0,'OOMKilled':False}},absent=lambda name:True)
  class Protocol:
   def __init__(this,command,directory):
    self.assertTrue((out/'plan.snapshot.json').is_file());self.assertEqual(command[command.index('--label')+1],'b70.prefix.plan='+q.sha(out/'plan.snapshot.json'));seen.append(command)
   def request(this,name,ids,fresh,max_new):self.assertEqual((ids,fresh,max_new),([11,12],1,1));return {'ids':ids,'CPU_TINY':True}
   def close(this):return 0
  def extract(raw,cap,args,env,stages):
   files=[]
   for layer in range(-1,48):
    path=cap/('CPU-control-'+str(layer)+'.bin');path.write_bytes(struct.pack('<f',1));files.append({'layer':layer,'path':str(path)})
   return {'logits':[files[0]],'residuals':files[1:]}
  actual_stat=q.serial.os.stat
  def device_stat(path,*args,**kwargs):return SimpleNamespace(st_gid=1000) if str(path).startswith('/dev/dri/') else actual_stat(path,*args,**kwargs)
  with patch.object(runtime,'providers',return_value=(fake,None)),patch.object(runtime,'genuine_baseline',return_value=({},{})),patch.object(runtime,'engine_binding',return_value={}),patch.object(q.serial,'CacheOffMergedProtocol',Protocol),patch.object(q.serial,'extract',extract),patch.object(runtime.subprocess,'run',return_value=SimpleNamespace(returncode=0)),patch.object(q.serial.os,'stat',device_stat):result=q.serial.run(plan,batch,out,health,0)
  self.assertTrue(result['passed']);self.assertEqual(len(result['comparisons']),49);self.assertEqual(q.read(out/'plan.snapshot.json'),plan);self.assertEqual(q.read(out/'serial-0/command.json'),seen[0])
  with patch.object(q.proof,'artifact_bindings',return_value={'CPU_TINY':True}):report=q.completed_report(plan,out,result)
  self.assertEqual(report['plan_sha256'],q.sha(out/'plan.snapshot.json'));self.assertFalse(report['actual_cached_state_handoff_qualified'])
 def test_allgroups_including_real_solo_required_before_complete_numeric_flag(self):
  import validate_batch_serial_positive_cacheoff_v10 as reader
  collector=self.root/'collector';(collector/'child').mkdir(parents=True);jobs=[{'rid':i,'role':'admission' if i<6 else 'solo_migration'} for i in range(7)];q.write(collector/'child/serial-jobs.json',{'jobs':jobs});q.write(collector/'input-plan.snapshot.json',{'CPU_SYNTHETIC':True});batch={i:'CPU_TINY' for i in range(343)};bound=({}, {}, {}, {'CPU_SOURCE_ONLY':True},batch)
  def group(directory):
   number=int(Path(directory).name);out=self.root/str(number);(out/'child').mkdir(parents=True,exist_ok=True);q.write(out/'child/report.json',{'CPU_GROUP':number});plan={'group_index':number,'batch_parent':str(collector),'collector_plan_sha256':q.sha(collector/'input-plan.snapshot.json')};return {},{},plan,{'matched_vector_pairs':294 if number==0 else 49}
  with patch.object(reader.ctrl,'collector_binding',return_value=bound),patch.object(reader,'finalized_binding',group):
   with self.assertRaises(ValueError):reader.all_groups_binding(collector,[self.root/'0'])
   r=reader.all_groups_binding(collector,[self.root/'0',self.root/'1']);self.assertEqual(r['matched_vector_pairs'],343);self.assertFalse(r['actual_cached_state_handoff_qualified']);self.assertFalse(r['cache_qualification_granted'])
   with self.assertRaises(ValueError):reader.all_groups_binding(collector,[self.root/'0',self.root/'0'])
if __name__=='__main__':unittest.main()
