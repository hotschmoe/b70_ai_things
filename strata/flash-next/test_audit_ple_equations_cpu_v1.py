#!/usr/bin/env python3
import unittest
import numpy as np
from audit_ple_equations_cpu_v1 import source_stage_estimate
from original_math_scalar import ple_postprojection,metrics

class PleSourceTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(123);self.key=rng.normal(0,.2,(4,2560));self.value=rng.normal(0,.2,2560);self.residual=rng.normal(0,.2,(4,2560));self.gamma=[rng.normal(1,.1,(4,2560)) for _ in range(3)];self.history=np.zeros((9,10240));self.weights=rng.normal(0,.2,(10240,4))
 def compute(self,history=None,weights=None):return source_stage_estimate(self.key,self.value,self.residual,self.gamma,self.history if history is None else history,self.weights if weights is None else weights)
 def test_frozen_math_equivalent_bounded_synthetic_with_small_storage_seams(self):
  old=ple_postprojection(self.key.astype('<f4'),self.value.astype('<f4'),self.residual.astype('<f4'),*[g.astype('<f4') for g in self.gamma],self.history,self.weights.astype('<f4').T,lane='declared_storage');new=self.compute();self.assertLess(metrics(new['result'],old['result'])['nmse'],1e-12);self.assertFalse(new['native_reducer_or_intrinsic_qualified'])
 def test_zero_history_only_newest_tap(self):
  expected=self.compute();wrong=self.weights.copy();wrong[:,:3]+=100;self.assertTrue(np.array_equal(expected['result'],self.compute(weights=wrong)['result']));wrong=self.weights.copy();wrong[:,3]+=.1;self.assertFalse(np.array_equal(expected['result'],self.compute(weights=wrong)['result']))
 def test_direct_gated_residual_required(self):
  out=self.compute();wrong=self.residual+out['conv_activation'];self.assertFalse(np.allclose(out['result'],wrong));self.assertTrue(np.allclose(out['result'],self.residual+out['gated']+out['conv_activation'],rtol=1e-6,atol=1e-7))
 def test_channel_tap_transpose_rival_changes_result(self):
  wrong=self.weights.reshape(4,10240).T;self.assertFalse(np.array_equal(self.compute()['result'],self.compute(weights=wrong)['result']))
 def test_dilation_history_and_normalized_nextrow(self):
  history=np.arange(9,dtype=float)[:,None]*np.ones((9,10240));out=self.compute(history);self.assertTrue(np.array_equal(out['next_history'][-1],out['normalized'].reshape(-1)));self.assertTrue(np.array_equal(out['next_history'][:-1],history[1:]));history2=history.copy();history2[[1,2,4,5,7,8]]+=100;self.assertTrue(np.array_equal(out['result'],self.compute(history2)['result']));history2=history.copy();history2[3]+=1;self.assertFalse(np.array_equal(out['result'],self.compute(history2)['result']))
 def test_nonfinite_and_geometry_rejected(self):
  bad=self.weights.copy();bad[0,0]=np.nan
  with self.assertRaises(ValueError):self.compute(weights=bad)
  with self.assertRaises(ValueError):self.compute(weights=self.weights.T)

if __name__=='__main__':unittest.main()
