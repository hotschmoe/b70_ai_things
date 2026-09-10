"""CPU checks for the independently specified, measured XPU scalar semantics."""
import unittest
import torch
from oracle import normalized_reference


class ScalarReferenceTests(unittest.TestCase):
    def test_measured_fixture_differences(self):
        gen=torch.Generator().manual_seed(8123)
        for scale, half_count, byte_count in ((.029828752790178572,7837,156),
                                               (.02104317801339286,6441,0)):
            scale=float(torch.tensor(scale))
            value=torch.randn(67,4,256,generator=gen).half()
            cpu=value.clone().div_(torch.tensor(scale))
            observed=normalized_reference(value,scale,'xpu')
            self.assertEqual(int((cpu.view(torch.int16)!=observed.view(torch.int16)).sum()),half_count)
            self.assertEqual(int((cpu.to(torch.float8_e4m3fn).view(torch.uint8)!=observed.to(torch.float8_e4m3fn).view(torch.uint8)).sum()),byte_count)

    def test_power2_and_rounding_boundary(self):
        value=torch.tensor([-.80517578125,.50732421875,0.,1.],dtype=torch.float16)
        for scale in (.125,.25):
            self.assertTrue(torch.equal(normalized_reference(value,scale,'xpu'),value/scale))
        scale=float(torch.tensor(.029828752790178572))
        self.assertEqual(normalized_reference(value,scale,'xpu')[:2].tolist(),[-26.984375,17.0])
        self.assertEqual(normalized_reference(value,scale,'cpu')[:2].tolist(),[-27.0,17.015625])

    def test_unrepresentable_effective_scale_rejected(self):
        for scale in (1e-12,1e10,float('nan'),-1.):
            with self.assertRaises(AssertionError):
                normalized_reference(torch.ones(1,dtype=torch.float16),scale,'xpu')


if __name__=='__main__':
    unittest.main()
