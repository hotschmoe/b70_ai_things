#!/usr/bin/env python3
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from full48_reference_inventory_fixture_v1 import inventory_contract,INVENTORY,SyntheticFull48Fixture

class Full48Tests(unittest.TestCase):
 def test_actual_header_only_catalog_complete_and_mixedformats(self):
  result=inventory_contract();self.assertEqual(result['source_tensors'],1224);self.assertEqual(result['unsupported_decoder_formats'],[]);self.assertEqual(len(result['gdn_layers']),36);self.assertEqual(len(result['qsa_layers']),12);self.assertEqual(result['ffn_exception_map']['Q5_K_gate_up_layers'],[2]);self.assertFalse(result['native_full48_ready'])
 def test_missing_original_role_refused(self):
  data=json.loads(INVENTORY.read_text())
  for file in data['files']:
   if '/UD-Q4_K_XL/' in file['path']:file['tensors']=[t for t in file['tensors'] if t['name']!='blk.47.hc_ffn_inject.weight']
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'CPU-mock-header.json';path.write_text(json.dumps(data))
   with self.assertRaises(ValueError):inventory_contract(path)
 def test_synthetic_all48_schedule_owned_everyprefix(self):
  for length in (1,2,4,8):
   result=SyntheticFull48Fixture().run(list(range(length)));self.assertEqual(len(result['gdn_states']),36);self.assertEqual(len(result['qsa_states']),12)
   for row in result['trace']:self.assertEqual([v['layer'] for v in row['layers']],list(range(48)));self.assertEqual(len({v['state_owner'] for v in row['layers']}),48)
   self.assertFalse(result['real_original_weights_used']);self.assertFalse(result['actual_native_storage_routes_executed']);self.assertFalse(result['full_model_math_qualified'])
 def test_fresh_history_repeat_and_divergent_suffix(self):
  fixture=SyntheticFull48Fixture();a=fixture.run([1,2,3,4]);b=fixture.run([1,2,3,4]);c=fixture.run([1,2,3,5]);self.assertTrue(np.array_equal(a['trace'][-1]['logits_toy16'],b['trace'][-1]['logits_toy16']));self.assertFalse(np.array_equal(a['trace'][-1]['logits_toy16'],c['trace'][-1]['logits_toy16']))
 def test_layer_state_arrays_do_not_alias(self):
  result=SyntheticFull48Fixture().run([1,2]);states=result['gdn_states'];a=states[0].copy();states[1].flat[0]+=1;self.assertTrue(np.array_equal(states[0],a));self.assertEqual(result['last_two'],[1,2]);self.assertEqual(result['ple_history'].shape,(9,16))

if __name__=='__main__':unittest.main()
