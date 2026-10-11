"""Actual derived normal43 prepare/dispatch with tiny owned dependencies only."""
import ast,copy,json,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source43_normal_model_intake_v1 as intake
import run_source43_normal_model_cli_v1 as a
import validate_source43_normal_model_cli_v1 as r
from test_source41_model_actor_cpu_v5 import Controls as OldShapes
class Controls(unittest.TestCase):
 def fixture(self):return OldShapes().fixture()
 def test_fresh43_mode0_and_actual40_control_only(self):
  m,cfg,inputs=self.fixture();left,x=a.phase_recipe(a.PHASES[0],m,cfg,inputs,'a'*64);right,y=a.phase_recipe(a.PHASES[1],m,cfg,inputs,'a'*64)
  self.assertEqual([x['engine_generation'],y['engine_generation']],[40,43]);self.assertEqual(x['requests'],y['requests']);self.assertTrue(all(v['max_new']==64 and v['fresh']==1 and v['pin']is None for v in x['requests']));self.assertEqual(y['native_env']['STRATA_QSA_INDEXER_MATERIALIZED_HALF'],'0');self.assertFalse(y['observer_enabled']);self.assertTrue(right['source'].endswith('120045Z-kfiohqmn/source'))
  with self.assertRaises(ValueError):a.phase_recipe('source41-original-fidelity',m,cfg,inputs,'a'*64)
 def test_only_declared_native_environment_change(self):
  m,cfg,inputs=self.fixture();_,x=a.phase_recipe(a.PHASES[0],m,cfg,inputs,'a'*64);_,y=a.phase_recipe(a.PHASES[1],m,cfg,inputs,'a'*64);env=dict(y['native_env']);self.assertEqual(env.pop('STRATA_QSA_INDEXER_MATERIALIZED_HALF'),'0');self.assertEqual(env,x['native_env']);self.assertEqual(x['native_argv'],y['native_argv'])
 def test_real_prepare_routes_all_fresh43_dependencies(self):
  import prefill_chunk64_512_control_v2 as shape
  import qualify_indexer_materialized43_masked_v2 as masked
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);here=root/'contract';here.mkdir();template=root/'prepared';template.mkdir();m,cfg,inputs=self.fixture();(template/'prepared.json').write_text(json.dumps(m));(template/'server-config.json').write_text(json.dumps(cfg));rows=[];expected=[]
   for i in range(4):
    name='UD-Q4_K_XL/shard'+str(i);path=root/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'CPU_TINY');st=path.stat();sig=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns];digest=hashlib.sha256(path.read_bytes()).hexdigest();expected.append({'path':name,'size':st.st_size,'sha256':digest});rows.append({'path':str(path),'passed':True,'sha256':digest,'expected_sha256':digest,'bytes':st.st_size,'stat_before':sig,'stat_after':sig})
   lock=here/'model-lock.json';lock.write_text(json.dumps({'revision':'CPU_TINY_REV','destination':'.','files':expected}));identity=root/'identity.json';identity.write_text(json.dumps({'passed':True,'lock_sha256':intake.sha(lock),'model_revision':'CPU_TINY_REV','rows':rows}));ids=root/'ids.json';ids.write_text(json.dumps(inputs))
   with patch.object(a,'closure',return_value='CPU_SYNTHETIC_PLAN'),patch.object(intake,'ROOT',root),patch.object(intake,'HERE',here),patch.object(shape,'validate_prepared',return_value=m),patch.object(intake,'sdk_binding',return_value={'generation':43,'engine_receipt_sha256':'a'*64})as sdk,patch.object(intake,'upload_binding',return_value={'actual43upload':True})as up,patch.object(masked,'finalized_binding',return_value={'actual43masked':True})as mask:
    plan=intake.prepare(template,ids,root/'up',root/'oracle',root/'pack',root/'masked',identity,root/'out')
   self.assertEqual(plan['source43'],{'generation':43,'engine_receipt_sha256':'a'*64});self.assertEqual(plan['fresh43_masked_component'],{'actual43masked':True});self.assertTrue((root/'out/plan.json').is_file());sdk.assert_called_once();up.assert_called_once();mask.assert_called_once()
 def test_private_main_source_namespace_and_no_fidelity_branch(self):
  n=a.ports();self.assertIs(n['intake'],intake);self.assertIs(n['PLAN'],a.PLAN);self.assertIn(('off-equivalence',),n['main'].__code__.co_consts);self.assertNotIn('validate_source41_model_cli_v3',n['main'].__code__.co_consts);self.assertIs(n['PHASES'],a.PHASES)
 def test_public_module_dependencies_and_full4_failure_close_dispatch(self):
  attrs={n.attr for n in ast.walk(ast.parse(Path(r.original.__file__).read_text()))if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='a'};self.assertTrue(all(hasattr(a,name)for name in attrs),attrs)
  namespace=r.ports();events=[]
  def entry(root):events.append('entry');raise RuntimeError('actual tiny entry failure')
  namespace['_finalized_binding']=entry
  def scan(lock_path,lock,shards,path,after):events.append('close');Path(path).write_text('{}');self.assertIs(type(after),float);return {'passed':True}
  with tempfile.TemporaryDirectory()as t,patch.object(r.c,'read',return_value={'destination':'CPU_NONMODEL','files':[{'path':'UD-Q4_K_XL/shard'+str(i)}for i in range(4)]}),patch.object(a.life.q.upload_v2,'full_buffered_identity',side_effect=scan),patch.object(intake,'guard',side_effect=lambda *args:events.append('pages')):
   with self.assertRaisesRegex(RuntimeError,'actual tiny entry failure'):namespace['finalized_binding'](Path(t))
  self.assertEqual(events,['entry','close','pages'])
if __name__=='__main__':unittest.main()
