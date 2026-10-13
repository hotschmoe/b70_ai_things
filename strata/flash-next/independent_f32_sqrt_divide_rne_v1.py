"""Exact positive finite F32 sqrt-RNE and reciprocal-RNE, no hardware samples.
Integer/rational comparisons decide neighbors and ties; no approximate seed.
This defines explicit sqrt then divide, NOT an OpenCL rsqrt emulation.
"""
import math,struct
from fractions import Fraction

def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
def bits(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def value(raw):return struct.unpack('<f',struct.pack('<I',raw))[0]
def positive(argument):
 x=f32(argument)
 if not math.isfinite(x)or x<=0:raise ValueError('Positive finite own F32 operand required')
 return Fraction(x)
def round_positive(compare):
 # compare(q): negative/zero/positive when q is below/equal/above exact root.
 lo,hi=0,0x7f7fffff
 while lo<hi:
  mid=(lo+hi+1)//2
  if compare(Fraction(value(mid)))<=0:lo=mid
  else:hi=mid-1
 if compare(Fraction(value(lo)))==0:return value(lo)
 upper=lo+1
 if upper>0x7f7fffff:raise ValueError('Bounded finite F32 result required')
 cmp=compare((Fraction(value(lo))+Fraction(value(upper)))/2)
 return value(upper if cmp<0 else lo if cmp>0 else lo if lo%2==0 else upper)
def sqrt_f32_rne(argument):
 x=positive(argument)
 return round_positive(lambda q:q*q-x)
def reciprocal_f32_rne(argument):
 x=positive(argument)
 return round_positive(lambda q:q*x-1)
def rsqrt_f32_rne(argument):
 x=positive(argument)
 return round_positive(lambda q:q*q*x-1)
def sqrt_then_divide_f32_rne(argument):return reciprocal_f32_rne(sqrt_f32_rne(argument))
