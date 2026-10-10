"""Original input versus final mutated headers; tiny source only, no compiler."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import c139_header_payload_binding_v2 as headers
import prepare_full_cache_memory39_v1 as producer
class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='C139-headers-CPU-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);_,rows=producer.reconstruct()
  for name,text in rows.items():
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
  self.plan=json.loads((headers.HERE/'full-cache-memory39-engine-build-plan-v1.json').read_bytes())
 def test_final66_and_original30_independent_payloads(self):
  r=headers.binding(self.plan,self.root);self.assertEqual(len(r['original_payloads']),30);changed=[x['path'] for x in r['original_payloads'] if x['modified_by_later_ordered_patches']];self.assertEqual(len(changed),2);self.assertFalse(r['older_source37_runtime_proof_transferred'])
 def test_mutated_final_header_rejected_even_if_original_payload_true(self):
  p=self.root/'sycl/include/strata/core/batch_fidelity_contract.hpp';p.write_text(p.read_text()+'\n// CPU mutation\n')
  with self.assertRaises(ValueError):headers.binding(self.plan,self.root)
 def test_changed_original_payload_declaration_and_order_reject(self):
  for mode in ('sha','order'):
   p=copy.deepcopy(self.plan)
   if mode=='sha':p['added_header_payloads'][0]['sha256']='0'*64
   else:p['added_header_payloads'][0],p['added_header_payloads'][1]=p['added_header_payloads'][1],p['added_header_payloads'][0]
   with self.assertRaises(ValueError):headers.binding(p,self.root)
 def test_independently_retained_original_bytes_changed_refused(self):
  original=headers.sha
  with patch.object(headers,'sha',side_effect=lambda p:'0'*64 if Path(p)==headers.SOURCE37_ROOT/'sycl/include/strata/core/batch_fidelity_contract.hpp' else original(p)):
   with self.assertRaises(ValueError):headers.binding(self.plan,self.root)
if __name__=='__main__':unittest.main()
