import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import native62_hc_rms_raw_v2 as raw
import native62_hc_rms_comparison_v2 as compare
import run_native62_hc_rms_ON_v2 as actor
import test_source59_mode0_fidelity_cpu_v2 as constructor
import native_source41_cli_protocol_v4 as pipe
class Controls(unittest.TestCase):
 def test_explicit_MAIN_actor_over_unconfigured_import(self):
  # Reproduce the split: canonical imported actor has no configured SDK.
  old=raw.prior._namespace['actor'];self.assertIsNone(old.parent.fresh)
  actual=SimpleNamespace(parent=SimpleNamespace(fresh=SimpleNamespace(SDK=Path('/mnt/vm_8tb/b70/build/strata-native62-incremental61-v1-20261013T020818Z-wcsa7gom'),PLAN=actor.HERE/'native62-HC-perf-explicit-sqrt-engine-proposal-v1.json')),c=actor.c,require=actor.require,requests=actor.requests)
  fn=raw.ports(actual);self.assertIs(fn.__globals__['actor'],actual);fields=fn.__globals__['field_contract']();self.assertEqual(len(fields),33);self.assertEqual(sum(x['bytes']for x in fields),7023048);self.assertIsNone(old.parent.fresh)
 def test_actual_protected_constructor_once(self):
  invoke=constructor.Constructor().call(actor.Protocol);v=json.loads(Path('/tmp/b70-native62-hc-rms-genuine-construction-v1.json').read_text())['commands'][1]
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);actor.c.write(p/'recipe.json',v['recipe']);active=[]
   with patch.object(pipe.subprocess,'Popen',side_effect=OSError('no-launch MAIN ctor sentinel'))as call:
    with self.assertRaisesRegex(OSError,'no-launch MAIN ctor sentinel'):invoke(v['native_command'],p,lambda:None,active,v['recipe'])
   self.assertEqual(call.call_count,1);self.assertEqual(active,[])
 def test_real_consume_three_front_two_underlying(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'field';p.write_bytes(b'\0'*16);self.assertEqual(compare.consume(p,16,compare.sha(p)),p.read_bytes())
   with self.assertRaises(ValueError):compare.consume(p,16,'0'*64)
 def test_ON_only_math_flag_and_private_callbacks(self):
  self.assertEqual(actor.PHASES,('native62-hc-rms-ON',));self.assertIs(actor.BASE_RUN_PHASE.__globals__,actor.NS);self.assertIs(actor.NS['set_sdk'].__globals__,vars(actor.parent));v=json.loads(Path('/tmp/b70-native62-hc-rms-genuine-construction-v1.json').read_text())['commands'][1]['recipe']
  with patch.object(actor.parent,'phase_recipe',return_value=({},copy.deepcopy(v))):_,r=actor.phase_recipe(actor.PHASES[0],{},{},{},'0'*64)
  self.assertEqual(r['native_env']['STRATA_HC_EXPLICIT_SQRT_DIVIDE'],'1');self.assertEqual(r['explicit_sqrt_divide'],1);self.assertEqual(len(r['requests']),4)
if __name__=='__main__':unittest.main()
