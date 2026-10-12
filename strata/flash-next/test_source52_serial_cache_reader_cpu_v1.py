import unittest,tempfile,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import validate_source52_serial_cache_latency_v3 as a
class Controls(unittest.TestCase):
 def test_actual_boundary_private_exports(self):
  active=SimpleNamespace(**{**vars(a.actor),**a.actor.NS});ports=a.boundary_reader_ports(active);self.assertIn('finalized_binding',ports)
  for name in('admission','phase_recipe','life','intake','contract','Protocol','stat_snapshot','campaign_stats','fresh','time'):self.assertTrue(hasattr(active,name),name)
  self.assertIs(active.contract.requests,a.actor.requests);self.assertIs(active.contract.equivalence,a.actor.equivalence);self.assertEqual(active.__file__,a.actor.__file__)
 def test_full_head_real_extent_and_wrong_argmax(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp).resolve();d=root/'phase';(d/'captures').mkdir(parents=True);p=d/'captures/head.f32';x=np.zeros(248320,dtype='<f4');x[846]=1;p.write_bytes(x.tobytes());digest=hashlib.sha256(p.read_bytes()).hexdigest();report={'artifacts':{'phase/captures/head.f32':digest}};meta={'vectors':[{'phase':'first_logits_before_sampler','path':str(p),'bytes':993280,'sha256':digest}]};self.assertTrue(a.head_binding(root,report,d,meta,{'output_ids':[846]})['emitted_is_argmax'])
   with self.assertRaises(ValueError):a.head_binding(root,report,d,meta,{'output_ids':[847]})
   x[0]=np.nan;p.write_bytes(x.tobytes());digest=hashlib.sha256(p.read_bytes()).hexdigest();report['artifacts']['phase/captures/head.f32']=digest;meta['vectors'][0]['sha256']=digest
   with self.assertRaises(ValueError):a.head_binding(root,report,d,meta,{'output_ids':[846]})
 def test_real_recipe_with_metadata_auth_stubs(self):
  b=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010');decl=a.c.read(b/'source52-serial-cache-v3-actual-construction-agent.json');recipe=decl['commands'][0]['recipe'];ns=a.actor.NS;parent=a.actor.parent
  report={'SDK_root':'/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww','SDK_receipt_sha256':'8f6c09aa60493dd6403771cd10592bdb780947110c3035379b4493d0a4298d94','API_prepared_root':str(b/'source43-api-mode1-baseline-v3-prepared-v1'),'API_prepared_sha256':'562a2a25bdc4522f377909ac37d29f5e2de4f021ea0712e20e263a727f3d4298','role_input':'prefill_about512','producer_pid':584759,'producer_start_ticks':1,'source_plan_sha256':'synthetic-source-only-auth-stub'}
  def runtime(*args):ns['_RUNTIME']=recipe['tool_runtime']
  with patch.dict(ns,{'set_api':lambda *x:None,'set_runtime':runtime}),patch.object(a.actor,'closure',return_value=report['source_plan_sha256']):
   with a.configured_actor(report,recipe)as active:
    plan=a.c.read(b/'source43-normal-cli-v1-prepared-v1/plan.json');m=a.c.read(Path(plan['template_path'])/'prepared.json');cfg=a.c.read(Path(plan['template_path'])/'server-config.json');inputs=a.c.read(plan['inputs_path']);descriptor,current=active.phase_recipe(active.PHASES[0],m,cfg,inputs,'36e8c5e9d3dae19fb77722f13e70cbaf156ba81c8cf805ba71a81b113c42234c');name,command=active.life.contract.native_command(b/'source52-serial-cache-latency-v3-run-v1'/active.PHASES[0],descriptor,current,584759,'36e8c5e9d3dae19fb77722f13e70cbaf156ba81c8cf805ba71a81b113c42234c');self.assertEqual(current,recipe);self.assertIn('--prompt-cache 3',' '.join(command));self.assertEqual(len(current['requests']),6)
 def test_actual_fixture_metadata_B487(self):
  p=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source52-serial-cache-tokenizer-v2/cache-fixture.json');v=a.c.read(p);self.assertEqual(a.c.sha(p),'27685dc74865eccd54cff98346351b2ba18a0aa13391fdefe96233b305aed1c7');self.assertEqual((len(v['A']),len(v['B']),v['pin']),(2048,2023,1536));self.assertEqual((len(v['A'])-v['pin'],len(v['B'])-v['pin']),(512,487));self.assertEqual(v['A'][:1536],v['B'][:1536])
if __name__=='__main__':unittest.main()
