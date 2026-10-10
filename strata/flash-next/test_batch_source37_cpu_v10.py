"""H49 CPU controls; tiny recipes only, no GPU/model/pack payload."""
import ast,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import batch_numerical_execution_v49 as ctrl
import batch49_profile_v1 as profile
import batch49_plan_snapshot_v1 as snapshots
import batch49_prelease_v1 as prelease
import prepare_batch_case37_v10 as case
class Controls(unittest.TestCase):
 def recipe(self):
  cfg={'args':['--split','32,16','--kv-type','f16','--ple','65536','--prompt-cache','1'],'env':{'ONEAPI_DEVICE_SELECTOR':'level_zero:0,1','KEEP_MATH':'source-baseline'}}
  prepared={'runtime':{'image':'sha256:CPU-synthetic-image'},'pack':'/CPU-synthetic-pack','cards':[0,1]}
  args,env=profile.configuration(cfg,4)
  return cfg,prepared,dict(kind='native',slots=4,args=args,env=env,image=prepared['runtime']['image'],pack=prepared['pack'],cards=[0,1])
 def test_reconstruction_exact_preserves_baseline_math(self):
  cfg,prepared,p=self.recipe();self.assertTrue(profile.binding(p,cfg,prepared));self.assertIn('32,16',p['args']);self.assertIn('f16',p['args']);self.assertEqual(p['env']['KEEP_MATH'],cfg['env']['KEEP_MATH'])
 def test_math_argument_drift(self):
  cfg,prepared,p=self.recipe();p['args'][p['args'].index('--kv-type')+1]='q8';self.assertRaises(ValueError,profile.binding,p,cfg,prepared)
 def test_cache_drift(self):
  cfg,prepared,p=self.recipe();p['args'][p['args'].index('--prompt-cache')+1]='1';self.assertRaises(ValueError,profile.binding,p,cfg,prepared)
 def test_environment_drift(self):
  cfg,prepared,p=self.recipe();p['env']['KEEP_MATH']='changed';self.assertRaises(ValueError,profile.binding,p,cfg,prepared)
 def test_image_drift(self):
  cfg,prepared,p=self.recipe();p['image']='sha256:FOREIGN';self.assertRaises(ValueError,profile.binding,p,cfg,prepared)
 def test_pack_and_device_drift(self):
  for key,value in [('pack','/FOREIGN'),('cards',[0])]:
   cfg,prepared,p=self.recipe();p[key]=value;self.assertRaises(ValueError,profile.binding,p,cfg,prepared)
 def test_eager_presence_and_duplicate_baseline(self):
  for key in ('STRATA_VERIFY_EAGER','STRATA_CKPT_REREAD'):
   cfg,_,_=self.recipe();cfg['env'][key]='0';self.assertRaises(ValueError,profile.configuration,cfg,4)
  cfg,_,_=self.recipe();cfg['args']+=['--prompt-cache','0'];self.assertRaises(ValueError,profile.configuration,cfg,4)
 def test_snapshot_exact_bytes_not_normalized(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plan';raw=b'{ "x": [1,2], "flag": true }\n';p.write_bytes(raw);s=snapshots.Snapshot(p);s.write(Path(d)/'owned');snapshots.write_snapshot(Path(d)/'child',s.plan,s.raw);self.assertEqual((Path(d)/'child').read_bytes(),raw);self.assertEqual(s.verify(),hashlib.sha256(raw).hexdigest())
 def test_original_plan_changed_during_health(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plan';p.write_text('{"x":1}');s=snapshots.Snapshot(p);p.write_text('{"x":2}');self.assertRaises(ValueError,s.verify)
 def test_inmemory_plan_changed_before_child(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plan';p.write_text('{"x":1}');s=snapshots.Snapshot(p);s.plan['x']=True;self.assertRaises(ValueError,s.verify)
 def test_wrong_raw_child_plan_rejected(self):
  with tempfile.TemporaryDirectory() as d:self.assertRaises(ValueError,snapshots.write_snapshot,Path(d)/'out',{'x':1},b'{"x":true}')
 def test_duplicate_and_nonfinite_plan_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'plan'
   for raw in ('{"x":1,"x":2}','{"x":NaN}'):
    p.write_text(raw);self.assertRaises(ValueError,snapshots.Snapshot,p)
 def test_api_rejected_before_prelease_source_io_or_semantics(self):
  def require(ok,message):
   if not ok:raise ValueError(message)
  fake=SimpleNamespace(require=require,sha=lambda path:self.fail('API must fail before source hash'))
  self.assertRaises(ValueError,prelease.cheap,dict(schema=4,harness_generation=49,slots=4,kind='api'),fake)
  with patch.object(ctrl,'topology_baselines',side_effect=AssertionError('must not enter baseline')):self.assertRaises(ValueError,ctrl.prepare,SimpleNamespace(kind='api'))
 def test_source_mutation_after_cheap_caught_before_device(self):
  events=[]
  def require(ok,message):
   if not ok:raise ValueError(message)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);driver=root/'driver';driver.write_text('CPU');dep=root/'helper';dep.write_text('OLD');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
   p=dict(schema=4,harness_generation=49,slots=4,kind='native',env={},driver_sha256=sha(driver),dependency_sha256={'helper':sha(dep)})
   def full(plan,*,pack_epoch=None,sdk_epoch=None):events.append('full');require(sha(dep)==plan['dependency_sha256']['helper'],'changed before device')
   fake=SimpleNamespace(require=require,sha=sha,__file__=str(driver),HERE=root,DEPENDENCIES=['helper'],source_observers_off=lambda e:None,c1=SimpleNamespace(leased=lambda c:events.append('lease')),manifest_binding=full)
   prelease.cheap(p,fake);self.assertEqual(events,[]);dep.write_text('NEW');self.assertRaises(ValueError,prelease.full_leased,p,fake);self.assertEqual(events,['lease','full'])
 def test_parent_snapshot_transition_and_one_parent_full_only(self):
  s=(ctrl.HERE/'qualify_batch_numerical_v49.py').read_text();ast.parse(s);self.assertEqual(s.count('current_chain=full_leased('),1);self.assertNotIn("plan=read(a.plan)",s);self.assertIn('a.expected_plan_sha256==admitted.sha256',s);self.assertLess(s.index('current_chain=full_leased'),s.index("pre=health('pre')"));self.assertIn('admitted.verify(plan)',s);self.assertIn("out/'admitted-plan.snapshot.json'",s)
  execution=Path(ctrl.__file__).read_text();self.assertIn('manifest_binding(plan, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)',execution);self.assertIn('admitted.verify(plan)',execution)
 def test_child_native_and_serial_exact_bytes(self):
  for name in ('run_batch_numerical_pilot_v49.py','run_batch_serial_controls_v49.py'):
   s=(ctrl.HERE/name).read_text();ast.parse(s);self.assertIn('write_snapshot(',s);self.assertNotIn('c1.write_snapshot',s)
  audit=(ctrl.HERE/'audit_batch_numerical_suite_v49.py').read_text();self.assertIn('external_path.read_bytes()==child_path.read_bytes()',audit.replace(' ',''))
 def test_owned_snapshot_join_rejects_posthealth_and_child_mutations(self):
  from audit_batch_numerical_suite_v49 import snapshot_plan_join
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'child').mkdir();paths=[root/'original',root/'admitted-plan.snapshot.json',root/'input-plan.snapshot.json',root/'child/plan.snapshot.json'];raw=b'{ \"x\":1 }\n';digest=hashlib.sha256(raw).hexdigest()
   for path in paths:path.write_bytes(raw)
   parent={'plan':str(paths[0]),'admitted_plan_path':str(paths[1]),'admitted_plan_sha256':digest,'plan_sha256':digest};child={'plan_sha256':digest};self.assertEqual(snapshot_plan_join(root,parent,{'x':1},child),digest)
   for path in paths:
    path.write_bytes(b'{\"x\":1}')
    self.assertRaises(ValueError,snapshot_plan_join,root,parent,{'x':1},child);path.write_bytes(raw)
 def test_actual_case_corpus_unchanged(self):
  for n in (4,6):
   new=case.prepare(n);old=ctrl.read(ctrl.HERE/('batch-numerical-case'+str(n)+'-source35-v2.json'))
   for key in ('tokens','messages','api_token_ids','actual_counter_policy','tokenizer_sha256','cancel_index','native_diagnostic_max_new_by_request','api_max_new_by_request'):self.assertEqual(new[key],old[key])
   self.assertEqual(new['harness_generation'],49)
if __name__=='__main__':unittest.main()
