import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_source41_host_latency_v2 as a
import validate_source41_host_latency_v2 as r
from test_source41_model_actor_cpu_v3 import Controls
class HostControls(unittest.TestCase):
 def test_actual41_mode0_and_only_h36_toggle(self):
  m,cfg,inputs=Controls().fixture();left=a.phase_recipe(a.PHASES[0],m,cfg,inputs,'a'*64);right=a.phase_recipe(a.PHASES[1],m,cfg,inputs,'a'*64)
  self.assertEqual(left[0],right[0]);self.assertNotIn('combined_generation',left[0]);self.assertEqual(left[1]['requests'],right[1]['requests']);self.assertEqual(left[1]['native_argv'],right[1]['native_argv'])
  self.assertTrue(all(p['materialized_mode']==0 and p['observer_enabled']is False for p in(left[1],right[1])))
  le=left[1]['native_env'];re=right[1]['native_env'];delta={k:v for k,v in re.items()if le.get(k)!=v};self.assertEqual(delta,{'STRATA_DECODE_TIMING':'1','STRATA_CRITICAL_PATH_TRACE':'1','STRATA_CRITICAL_PATH_TRACE_DIR':'/results/host-traces','STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256':'a'*64})
 def test_numeric_and_qsa_observers_remain_off(self):
  m,cfg,inputs=Controls().fixture()
  for phase in a.PHASES:
   _,p=a.phase_recipe(phase,m,cfg,inputs,'a'*64)
   for key in('STRATA_FIDELITY_DIAG','STRATA_PREFIX30','STRATA_PLE_INPUT33','STRATA_QSA3_TARGET','STRATA_BATCH_FIDELITY_DIAG'):self.assertEqual(p['native_env'].get(key,'0'),'0')
   self.assertEqual(p['native_env']['STRATA_QSA_INDEXER_MATERIALIZED_HALF'],'0')
 def test_exact_named_source_derivation_and_own_namespace(self):
  self.assertEqual(a._PORTS['derivation_counts'],{'choices':1,'phases':1,'prerequisite':1,'comparison_phase':1,'comparison_name':1,'trace_directory':1,'unused_ARM':1});self.assertEqual(r._PORTS['derivation_counts'],{'phases':1,'off_scope':1,'comparison':1,'scope_flag':2})
  self.assertEqual(a._PORTS['__file__'],a.__file__);self.assertIs(r._PORTS['a'],a);self.assertIs(a._PORTS['phase_recipe'],a.phase_recipe);self.assertIs(a._PORTS['run_phase'],a.run_phase)
 def test_wrong_phase_and_unsafe_math_refused(self):
  m,cfg,inputs=Controls().fixture()
  with self.assertRaises(ValueError):a.phase_recipe('original-fidelity',m,cfg,inputs,'a'*64)
  cfg=copy.deepcopy(cfg);cfg['env']['STRATA_QSA3_TARGET']='1'
  with self.assertRaises(ValueError):a.phase_recipe(a.PHASES[0],m,cfg,inputs,'a'*64)
 def test_repeated_ids_finish_gate_preserved_labelled_h36(self):
  controls=Controls();rows,p=controls.rows();
  for row in rows:
   lp='LP -1.25 '+' '.join(str(i)+':-1.25'for i in range(20));row['raw']['LP']=[lp];row['raw']['stdout'].insert(1,lp);row['raw']['stdout_seconds']=[.1,.15,.2]
  views=a.contract.requests(rows,p['requests']);proof=a.equivalence(views,copy.deepcopy(views));self.assertTrue(proof['H36_OFF_ON_equivalence_passed']);self.assertNotIn('OFF_equivalence_passed',proof)
  bad=copy.deepcopy(views);bad[1]['output_ids']=[8]
  with self.assertRaises(ValueError):a.equivalence(views,bad)
 def test_changed_lp_alone_rejects_crossarm_and_repeat(self):
  controls=Controls();rows,p=controls.rows()
  for row in rows:
   lp='LP -1.25 '+' '.join(str(i)+':-1.25'for i in range(20));row['raw']['LP']=[lp];row['raw']['stdout'].insert(1,lp);row['raw']['stdout_seconds']=[.1,.15,.2]
  views=a.contract.requests(rows,p['requests']);bad=copy.deepcopy(views);bad[1]['LP']=['LP changed']
  with self.assertRaises(ValueError):a.equivalence(views,bad)
  with self.assertRaises(ValueError):a.equivalence(bad,bad)
 def test_on_collector_failure_is_not_host_pass(self):
  with tempfile.TemporaryDirectory()as t:
   out=Path(t);directory=out/a.PHASES[1];directory.mkdir();(directory/'host-traces').mkdir();(directory/'requests.json').write_text('[{"raw":{}}]\n')
   with patch.dict(a._PORTS,{'native_phase':lambda *args,**kwargs:{'passed':True,'terminal':{}}}),patch.object(a.collector,'native_namespace',return_value=1),patch.object(a.collector,'lp20',return_value=[]),patch.object(a.c,'read',side_effect=lambda p:[]if p.name=='command.json'else [{'raw':{}}]),patch.object(a.collector,'collect',side_effect=ValueError('overflow owner clock refused')):
    with self.assertRaises(ValueError):a.run_phase(out,a.PHASES[1],None,None,None,'a'*64,None,None)
 def test_lp_disabled_is_omitted_not_zero_and_recipe_only(self):
  m,cfg,inputs=Controls().fixture();_,p=a.preregister_lp_disabled(m,cfg,inputs,'a'*64,a.PHASES[1]);self.assertTrue(all('logprobs='not in line for line in p['planned_protocol_lines']));self.assertFalse(p['LPdisabled_runtime_enabled']);self.assertFalse(p['LP0_is_disabled']);self.assertFalse(p['original_LP20_runtime_control_changed'])
if __name__=='__main__':unittest.main()
