"""Tiny CPU owned-only controls; no compiled helper/model execution."""
import inspect,unittest
from unittest.mock import patch
import numpy as np
import full48_owned_hc_fma_refinement_v1 as r
from full48_owned_composition_storage_v2 import RouteAwareFull48OwnedComposition as old
class Tests(unittest.TestCase):
 def model(self):
  m=r.OwnedHcFmaRefinement.__new__(r.OwnedHcFmaRefinement)
  m.invoke=lambda op,k,n,mode,a,b,c=None:np.frombuffer(r.hc.host.fma_f32(r.hc.raw(a),r.hc.raw(b),r.hc.raw(c)),dtype='<f4').copy() if k*n<=10240 else None
  return m
 def test_write_fused_negative(self):
  m=self.model();block=np.full(2560,1+2**-23,dtype='<f4');residual=np.full((4,2560),-1,dtype='<f4')
  with patch.object(r.hc,'sigmoid_candidate',return_value=(1-2**-23)/2):out=m.hc_write(residual,block,np.zeros(4))
  self.assertTrue(np.all(out==np.float32(-2**-46)));self.assertTrue(np.all(np.float32(block*np.float32(1-2**-23))-1==0))
 def test_write_shape(self):
  for residual,block,inject in [(np.zeros((1,2560)),np.zeros(2560),np.zeros(4)),(np.zeros((4,2560)),np.zeros(2),np.zeros(4)),(np.zeros((4,2560)),np.zeros(2560),np.zeros(3))]:
   with self.assertRaises(ValueError):self.model().hc_write(residual,block,inject)
 def test_nonfinite(self):
  with self.assertRaises(ValueError):r.finite(np.array([np.nan]),(1,))
 def test_qualification_first(self):
  with patch.object(r.qualification,'finalized_binding',side_effect=ValueError('unqualified')) as gate,patch.object(old,'__init__') as original:
   with self.assertRaises(ValueError):r.OwnedHcFmaRefinement(None,'0'*64,[],{},'/unused','/unused')
   gate.assert_called_once();original.assert_not_called()
 def test_native_input_interface(self):
  self.assertEqual(list(inspect.signature(r.OwnedHcFmaRefinement.tokens).parameters),['self','token_ids'])
  for ids in ([True],[-1],[248320],[0,0,0],['captured']):
   with self.assertRaises(ValueError):self.model().tokens(ids)
 def test_non_hc_inherited(self):
  self.assertNotIn('__init__',r.OwnedHcFmaRefinement.__dict__.get('gdn',{}));self.assertTrue(issubclass(r.OwnedHcFmaRefinement,old));source=inspect.getsource(r.OwnedHcFmaRefinement.tokens);self.assertIn('super().tokens(ids)',source)
 def test_projection_owned_tile(self):
  m=self.model();seen=[]
  class P:
   def shape(self,n):return (32,3)
   def rows(self,n,ids):seen.append((n,list(ids)));return np.zeros((len(ids),32))
  m.p=P();m.projector=type('Proj',(),{'tile':512})();m.invoke=lambda op,k,n,mode,a,b,c=None:np.arange(n,dtype='<f4');self.assertEqual(m.projection('owned.weight',np.zeros(32)).shape,(3,));self.assertEqual(seen,[('owned.weight',[0,1]),('owned.weight',[2])])
if __name__=='__main__':unittest.main()
