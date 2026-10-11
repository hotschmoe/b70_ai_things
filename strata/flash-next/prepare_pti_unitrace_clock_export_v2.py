"""Exact original u64 clock mirror; identical output-only C++ patch to V1."""
import math
from prepare_pti_unitrace_clock_export_v1 import *
def normalize(row):
 for k,v in row.items():require(type(v)is int and 0<=v<(1<<64),'typed u64 normalization input '+k)
 maxu=(1<<64)-1;mask=row['timer_mask'];scale=struct.unpack('=d',struct.pack('=Q',row['ns_per_cycle_ieee754_u64']))[0]
 require(row['timer_frequency_hz']>0 and math.isfinite(scale)and scale==float(1000000000)/float(row['timer_frequency_hz']),'exact frequency-derived normalization scalar')
 start=row['raw_kernel_start_ticks']&mask;end=row['raw_kernel_end_ticks']&mask;submit=row['submit_device_ticks']&mask
 delta=(start-submit if start>submit else mask-submit+1+start)&maxu
 duration=(end-start if start<=end else mask-start+1+end)&maxu
 def convert(v):
  value=float(v)*scale;require(math.isfinite(value)and 0<=value<(1<<64),'original unsigned cast must be representable');return int(value)
 host_start=(row['submit_host_ns']+convert(delta))&maxu
 return host_start,(host_start+convert(duration))&maxu
if __name__=='__main__':print(patch_text(),end='')
