import hashlib,json,unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import current59_full48_mode0_reference_v3 as v
from current59_full48_mode0_reference_v2 import ConsumerProjector as Old
INVENTORY=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/current-gguf-inventory.json')
def inventory_rows(x):
 if isinstance(x,dict):
  if all(k in x for k in('name','type','shape_ggml_order')):yield x
  for y in x.values():yield from inventory_rows(y)
 elif isinstance(x,list):
  for y in x:yield from inventory_rows(y)
class Provider:
 def __init__(self,t):self.t=t;self.reader=SimpleNamespace(tensors={t['name']:(None,t,None)});self.rows_requested=[]
 def shape(self,name):return tuple(self.t['shape_ggml_order'])
 def rows(self,name,indices):self.rows_requested.append((name,indices));return np.zeros((1,2560),dtype='<f4')
class Backend:
 def rows(self,w,x):return np.sum(w*x,axis=1,dtype=np.float32)
class Controls(unittest.TestCase):
 def metadata(self):
  b=INVENTORY.read_bytes();self.assertEqual(hashlib.sha256(b).hexdigest(),'8267da37e921300a348813ecb985dc185cde87d1918fb69d65e6aa1013b3b9cb');return [t for t in inventory_rows(json.loads(b))if t['name'].endswith('.ffn_gate_inp_shexp.weight')and not t['name'].startswith('blk.48.')]
 def test_actual_inventory_scalar_route_and_prior_failure(self):
  tensors=self.metadata();self.assertEqual(len(tensors),48)
  for t in tensors:
   self.assertEqual(t['shape_ggml_order'],[2560]);self.assertEqual(t['type'],'F32')
  p=Provider(tensors[0]);x=np.ones(2560,dtype='<f4')
  with self.assertRaises(IndexError):Old(p,Backend()).project(p.t['name'],x,weight='bf16')
  got=v.ConsumerProjector(p,Backend()).project(p.t['name'],x,weight='bf16');self.assertEqual(got.shape,(1,));self.assertEqual(p.rows_requested,[(p.t['name'],[0])])
 def test_no_arbitrary_flatten(self):
  t=dict(self.metadata()[0]);t['shape_ggml_order']=[2560,1];p=Provider(t)
  with self.assertRaises(ValueError):v.ConsumerProjector(p,Backend()).project(t['name'],np.ones(2560),weight='bf16')
 def test_original_exception_has_exact_role(self):
  p=Provider(self.metadata()[0]);problem=RuntimeError('owned source read failed')
  with patch.object(p,'rows',side_effect=problem):
   with self.assertRaises(RuntimeError)as caught:v.ConsumerProjector(p,Backend()).project(p.t['name'],np.ones(2560),weight='bf16')
  self.assertIs(caught.exception,problem);self.assertEqual(problem.reference_role_context['original_shape'],[2560]);self.assertEqual(problem.reference_role_context['role'],p.t['name'])
 def test_runtime_traceback_and_failure_close(self):
  import run_current59_full48_reference_v3 as run
  import run_source44_num38_original_reference_v2 as closing
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);identity=root/'identity.json';identity.write_text('{}');a=SimpleNamespace(allow_unqualified_estimates=True,native_root=root,output=root/'out',model_identity=identity,expected_identity_sha256=run.sha(identity))
   problem=IndexError('actual failing role');problem.reference_role_context={'role':'blk.0.ffn_gate_inp_shexp.weight','original_shape':[2560]}
   with patch.object(run,'source_binding',return_value='0'*64),patch.object(run,'full_targets',side_effect=problem),patch.object(closing,'closing_identity',return_value={'passed':True})as close:
    with self.assertRaises(ValueError):run.execute(a)
   report=json.loads((a.output/'report.json').read_text());self.assertFalse(report['passed']);self.assertIn('IndexError: actual failing role',report['failure']['traceback']);self.assertEqual(report['failure']['projection_role'],problem.reference_role_context);self.assertEqual(close.call_count,1)
if __name__=='__main__':unittest.main()
