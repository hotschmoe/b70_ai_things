"""Actual private runtime/intake/reader paths with tiny controlled dependencies."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_source43_num10_zero_v1 as a
import validate_source43_num10_zero_v1 as r
import source43_num10_intake_v1 as intake
import source43_num10_numeric_v1 as numeric
from test_source43_num10_case_cpu_v1 import Controls as Shapes
class Controls(unittest.TestCase):
 def fixture(self):
  m,cfg=Shapes().fixture();m.update(pack='/CPU_SYNTHETIC_PACK',runtime={'image':'CPU_SYNTHETIC_IMAGE'},profile='chunk64');return m,cfg,{'short':[248045,7],'prefill_about512':[248045,8]}
 def test_only_num10_flag_changes_between_43mode1_arms(self):
  m,cfg,inputs=self.fixture();left,x=a.phase_recipe('source43-num10-OFF',m,cfg,inputs,'a'*64);right,y=a.phase_recipe('source43-num10-ON',m,cfg,inputs,'a'*64)
  self.assertEqual(left,right);self.assertEqual(x['requests'],y['requests']);ex=dict(x['native_env']);ey=dict(y['native_env']);self.assertEqual(ex.pop('STRATA_LAYER0_Q8_DIAG'),'0');self.assertEqual(ey.pop('STRATA_LAYER0_Q8_DIAG'),'1');self.assertEqual(ex,ey);self.assertEqual(x['engine_generation'],43)
 def test_private_main_phase_roster_and_lifecycle_scope(self):
  n=a.derived();constants=n['main'].__code__.co_consts;self.assertIn(('source43-num10-OFF','source43-num10-ON'),constants);self.assertNotIn(('source40-control','source41-off'),constants);self.assertIn(('num10-equivalence',),constants);self.assertIs(n['intake'],intake);self.assertIs(n['admission'],a.admission);self.assertIsNot(n,vars(a.original))
  self.assertIn('layer0',n['run_phase'].__code__.co_consts[-1]if False else str(n['run_phase'].__code__.co_consts))
 def test_reader_actual43_and_required_full49_NUM10_collection(self):
  n=r.derived();self.assertIs(n['a'],a);self.assertIn('captured_requests',n['_finalized_binding'].__code__.co_consts);self.assertIn('actual_full49_observer_pairs',n['_finalized_binding'].__code__.co_consts);self.assertNotIn('validate_source41_model_cli_v3',str(n['_finalized_binding'].__code__.co_consts))
 def test_unknown_phase_before_launch(self):
  m,cfg,inputs=self.fixture()
  with self.assertRaises(ValueError):a.phase_recipe('source41-original-fidelity',m,cfg,inputs,'a'*64)
 def test_intake_own_recipe_authenticates_literal_first_id(self):
  m,cfg,inputs=self.fixture();self.assertEqual(intake.recipes(m,cfg,inputs,{} )['1']['requests'][0]['ids'],[248045]);inputs['short'][0]=7
  with self.assertRaises(ValueError):intake.recipes(m,cfg,inputs,{})
 def test_raw_numeric_exact_eight_length_literals_only(self):
  node=next(n for n in ast.parse(Path(numeric.original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='numeric');self.assertEqual(sum(type(n.value)is int and n.value==4 for n in ast.walk(node)if isinstance(n,ast.Constant)),8)
  with self.assertRaises(ValueError):numeric.numeric({},Path('/CPU_SYNTHETIC'),[(0,24),(24,48)])
if __name__=='__main__':unittest.main()

class BehavioralReaderControls(unittest.TestCase):
 def test_actual_reader_collects_both_arms_before_cpu_closing(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);(root/'report.json').write_text(json.dumps({'source_plan_sha256':'a'*64}))
   for phase in('source43-num10-OFF','source43-num10-ON'):
    p=root/phase;p.mkdir();(p/'requests.json').write_text(json.dumps([{'ordinal':i,'raw':{'ids':[248045],'fresh':1,'pin':None}}for i in range(1,5)]))
   old_derived=r.derived
   # Substitute only the inherited heavyweight final entry. The actual new
   # collection/49-comparison wrapper and per-arm reads execute unchanged.
   text=Path(r.__file__).read_text();node=next(n for n in ast.walk(ast.parse(text))if isinstance(n,ast.FunctionDef)and n.name=='with_actual_raw')
   namespace={**vars(r),'original_final':lambda _:{'passed':True}}
   exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[CPU-actual-new-reader-wrapper]','exec'),namespace)
   def compare(left,right):
    self.assertEqual(set(left),{'raw','meta'});self.assertEqual(left,right);return {'bitwise_equal':True}
   with patch.object(numeric,'numeric',return_value={'logits':['CPU_SYNTHETIC_HEAD'],'residuals':[]})as nums,patch.object(a.case,'collect',return_value={'frame':{'pid':1},'observed':[]})as layer,patch('layer0_numerical_qualification_v10.compare_numeric',side_effect=compare)as comparison:
    result=namespace['with_actual_raw'](root)
   self.assertEqual(nums.call_count,8);self.assertEqual(layer.call_count,4);self.assertEqual(comparison.call_count,4);self.assertEqual(len(result['captured_requests']),4)

class BehavioralIntakeControls(unittest.TestCase):
 def test_actual_prepare_executes_all_fresh43_dependencies_and_writes_truthful_plan(self):
  import hashlib
  import prefill_chunk64_512_control_v2 as shape
  import qualify_indexer_materialized43_masked_v2 as masked
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);here=root/'contract';here.mkdir();template=root/'prepared';template.mkdir();m,cfg,inputs=Controls().fixture()
   (template/'prepared.json').write_text(json.dumps(m));(template/'server-config.json').write_text(json.dumps(cfg));rows=[];expected=[]
   for i in range(1,5):
    name='UD-Q4_K_XL/shard'+str(i)+'.gguf';path=root/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'CPU_TINY_ORIGINAL');stat=path.stat();digest=hashlib.sha256(path.read_bytes()).hexdigest();sig=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns];expected.append({'path':name,'size':stat.st_size,'sha256':digest});rows.append({'path':str(path),'passed':True,'sha256':digest,'expected_sha256':digest,'bytes':stat.st_size,'stat_before':sig,'stat_after':sig})
   lock=here/'model-lock.json';lock.write_text(json.dumps({'revision':'CPU_SYNTHETIC_REV','destination':'.','files':expected}));identity=root/'identity.json';identity.write_text(json.dumps({'passed':True,'lock_sha256':intake.sha(lock),'model_revision':'CPU_SYNTHETIC_REV','rows':rows}));ids=root/'ids.json';ids.write_text(json.dumps(inputs))
   with patch.object(a,'closure',return_value='CPU_SYNTHETIC_PLAN'),patch.object(intake,'ROOT',root),patch.object(intake,'HERE',here),patch.object(shape,'validate_prepared',return_value=m),patch.object(intake,'sdk_binding',return_value={'generation':43})as sdk,patch.object(intake,'upload_binding',return_value={'actual43upload':True})as up,patch.object(masked,'finalized_binding',return_value={'actual43masked':True})as mask:
    plan=intake.prepare(template,ids,root/'up',root/'oracle',root/'pack',root/'masked',identity,root/'out')
   self.assertEqual(plan['source43'],{'generation':43});self.assertEqual(plan['fresh43_upload'],{'actual43upload':True});self.assertEqual(plan['fresh43_masked_component'],{'actual43masked':True});self.assertFalse(plan['source40_alias_claimed_for_source43']);self.assertTrue((root/'out/plan.json').is_file());sdk.assert_called_once();up.assert_called_once();mask.assert_called_once()
