import math,unittest
from fractions import Fraction
import independent_f32_sqrt_divide_rne_v1 as r
class Controls(unittest.TestCase):
 def test_analytical_exact_powers(self):
  for n in(-120,-30,0,30,120):
   x=2.0**n;self.assertEqual(r.sqrt_f32_rne(x),2.0**(n//2));self.assertEqual(r.sqrt_then_divide_f32_rne(x),2.0**(-n//2))
 def test_rational_even_midpoint(self):
  for lower in(0x3f800000,0x3f800001):
   midpoint=(Fraction(r.value(lower))+Fraction(r.value(lower+1)))/2
   self.assertEqual(r.bits(r.round_positive(lambda q:q-midpoint)),lower if lower%2==0 else lower+1)
 def test_current_exact_arg_and_midpoint_proof(self):
  x=r.value(0x3859f0be);low=r.rsqrt_f32_rne(x);self.assertEqual(r.bits(low),0x430aba1e);self.assertEqual(r.sqrt_then_divide_f32_rne(x),low);mid=(Fraction(low)+Fraction(r.value(r.bits(low)+1)))/2;self.assertGreater(Fraction(x)*mid*mid,1)
 def test_nonpositive_or_nonfinite_refused(self):
  for x in(0,-1,float('inf'),float('nan')):
   with self.assertRaises(ValueError):r.sqrt_then_divide_f32_rne(x)
if __name__=='__main__':unittest.main()
