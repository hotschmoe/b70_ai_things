import copy,unittest
from pathlib import Path
from unittest.mock import patch
import source52_clean_serial_cache_contract_v1 as contract
import validate_source52_clean_serial_cache_latency_v1 as reader
B=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
class Controls(unittest.TestCase):
 def rows(self,phase):
  d=B/'source52-serial-cache-latency-v3-run-v1'/phase;rows=reader.c.read(d/'requests.json');recipe=reader.c.read(d/'recipe.json')
  # Synthetic observer-off projection of closed actual protocol, not a clean runtime proof.
  for row in rows:row['raw']['stderr']=[x for x in row['raw']['stderr']if not x.startswith(('SFD ','PCL ','PDIAG '))]
  return rows,recipe['requests']
 def test_actual_DONE_source_shape_and_counter_refusals(self):
  rows,wants=self.rows('natural-p2048-B');v=contract.requests(rows,wants);self.assertEqual([x['new_prompt_rows']for x in v],[2048,512,487,487,512,487])
  for position in(8,14):
   bad=copy.deepcopy(rows);parts=bad[2]['raw']['done'].split();parts[position]=str(int(parts[position])+1);bad[2]['raw']['done']=' '.join(parts);bad[2]['raw']['stdout'][-1]=bad[2]['raw']['done']
   with self.assertRaises(ValueError):contract.requests(bad,wants)
 def test_crossarm_outputs_and_observer_refused(self):
  cold,cw=self.rows('natural-p2048-A');hot,hw=self.rows('natural-p2048-B');self.assertTrue(contract.equivalence(contract.requests(cold,cw),contract.requests(hot,hw))['capture_free_native_protocol']);hot[0]['raw']['stderr'].append('PCL {}')
  with self.assertRaises(ValueError):contract.requests(hot,hw)
 def test_real_recipe_command_no_capture_delta(self):
  decl=reader.c.read(B/'source52-serial-cache-v3-actual-construction-agent.json');recipe=decl['commands'][0]['recipe'];ns=reader.actor.NS;report={'SDK_root':'/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww','SDK_receipt_sha256':'8f6c09aa60493dd6403771cd10592bdb780947110c3035379b4493d0a4298d94','API_prepared_root':str(B/'source43-api-mode1-baseline-v3-prepared-v1'),'API_prepared_sha256':'562a2a25bdc4522f377909ac37d29f5e2de4f021ea0712e20e263a727f3d4298','role_input':'prefill_about512','producer_pid':584759,'producer_start_ticks':1,'source_plan_sha256':'synthetic-source-only-auth-stub'}
  def runtime(*args):ns['_RUNTIME']=recipe['tool_runtime']
  with patch.dict(ns,{'set_api':lambda *x:None,'set_runtime':runtime}),patch.object(reader.actor,'closure',return_value=report['source_plan_sha256']):
   with reader.configured_actor(report,recipe)as active:
    plan=reader.c.read(B/'source43-normal-cli-v1-prepared-v1/plan.json');m=reader.c.read(Path(plan['template_path'])/'prepared.json');cfg=reader.c.read(Path(plan['template_path'])/'server-config.json');inputs=reader.c.read(plan['inputs_path']);descriptor,r=active.phase_recipe(active.PHASES[0],m,cfg,inputs,'36e8c5e9d3dae19fb77722f13e70cbaf156ba81c8cf805ba71a81b113c42234c');self.assertEqual(r['native_argv'],recipe['native_argv']);self.assertEqual(r['requests'],recipe['requests']);self.assertEqual({k for k in r['native_env']if r['native_env'][k]!=recipe['native_env'][k]},{'STRATA_FIDELITY_DIAG','STRATA_PREFIX_DIAG','STRATA_PREFIX_LIFECYCLE_DIAG'});self.assertFalse(r['observer_enabled']);name,cmd=active.life.contract.native_command(B/'source52-clean-serial-cache-v1-run-v1'/active.PHASES[0],descriptor,r,584759,'36e8c5e9d3dae19fb77722f13e70cbaf156ba81c8cf805ba71a81b113c42234c');self.assertIn('STRATA_FIDELITY_DIAG=0',cmd);self.assertIs(active.contract.requests,contract.requests)
if __name__=='__main__':unittest.main()
