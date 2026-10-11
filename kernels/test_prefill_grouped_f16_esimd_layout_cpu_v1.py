"""One CPU source-layout control, not an ESIMD compile/device/math qualification."""
import re,unittest
from pathlib import Path

class Test(unittest.TestCase):
 def test_actual_B_index_packing_and_disjoint_tail_output(self):
  source=Path(__file__).with_name('prefill_grouped_f16_esimd_v1.hpp').read_text()
  # Evaluate the actual C++ B index expression only as bounded integer algebra.
  expression=re.search(r'b\[([^\]]+)\]=wr\[kk\]',source).group(1).replace('/','//')
  self.assertTrue(re.fullmatch(r'[kc0-9/()%+* ]+',expression))
  packed=[None]*256
  W=[[1000*c+k for k in range(16)]for c in range(16)]
  for c in range(16):
   for kk in range(16):
    index=eval(expression,{'__builtins__':{}},{'c':c,'kk':kk})
    self.assertIsNone(packed[index]);packed[index]=W[c][kk]
  # Independent hardware-format description: each 32-bit word carries a
  # vertical pair B[2pair,col], B[2pair+1,col], where B is transposed W.
  reference=[W[col][2*pair+half]for pair in range(8)for col in range(16)for half in range(2)]
  self.assertEqual(packed,reference)
  A=[[r-k for k in range(16)]for r in range(8)]
  unpacked=[[packed[2*(pair*16+col)+half]for pair in range(8)for half in range(2)]for col in range(16)]
  self.assertEqual([[sum(A[r][k]*unpacked[c][k]for k in range(16))for c in range(16)]for r in range(8)],[[sum(A[r][k]*W[c][k]for k in range(16))for c in range(16)]for r in range(8)])
  # Distinct source coordinates make a row-major-B mistake observable.
  self.assertNotEqual(packed,[W[col][k]for k in range(16)for col in range(16)])
  for count in [1,3,8]:
   for n in [16,1280,2560]:
    rows=[1,2,7,9,13,31,63,64][:count];observed=set();start=0
    for expert,ne in enumerate(rows):
     tiles=(ne+7)//8*(n//16)
     for tile in range(start,start+tiles):
      local=tile-start;m0=local//(n//16)*8;n0=local%(n//16)*16
      for r in range(8):
       if m0+r>=ne:continue
       for c in range(16):
        key=(expert,m0+r,n0+c);self.assertNotIn(key,observed);observed.add(key)
     start+=tiles
    self.assertEqual(observed,{(g,r,c)for g,ne in enumerate(rows)for r in range(ne)for c in range(n)})
if __name__=='__main__':unittest.main()
