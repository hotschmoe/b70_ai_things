import unittest
import numpy as np
from original_head_distribution_v1 import distributions

class DistributionTests(unittest.TestCase):
    def test_identical_extreme_logits_and_shift_invariance(self):
        x=np.array([10000.,9999.,-10000.,100.])
        same=distributions(x,x,2)
        self.assertEqual(same['KL_original_to_native'],0)
        shifted=distributions(x,x+10000,2)
        self.assertEqual(shifted['total_variation'],0)
        self.assertTrue(shifted['argmax_equal'])

    def test_swapped_winner_negative_control(self):
        result=distributions([10,0,-10],[0,10,-10],2)
        self.assertFalse(result['argmax_equal'])
        self.assertGreater(result['KL_original_to_native'],9)
        self.assertGreater(result['total_variation'],.99)
        self.assertFalse(result['quality_qualified'])
        self.assertIsNone(result['tolerance_gate'])

    def test_uniform_analytic_kl(self):
        result=distributions([0,0],[np.log(3),0],2)
        expected=.5*np.log(.5/.75)+.5*np.log(.5/.25)
        self.assertAlmostEqual(result['KL_original_to_native'],expected)
        self.assertAlmostEqual(result['total_variation'],.25)

    def test_tie_order_is_token_id(self):
        self.assertEqual(distributions([1,1,1],[1,1,1],2)['original_top_ids'],[0,1])

    def test_nonfinite_or_mismatched_vector_refused(self):
        for a,b in [([1,np.nan],[1,2]),([1,2],[1,np.inf]),([1,2],[1,2,3])]:
            with self.assertRaises(ValueError):distributions(a,b,2)

if __name__=='__main__':unittest.main()
