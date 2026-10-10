#!/usr/bin/env python3
"""Synthetic raw recollection controls; no native execution/model payload."""
import array,tempfile
from pathlib import Path
import collect_gdn_state_transition35_v1 as c

def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Invalid control accepted')
with tempfile.TemporaryDirectory(prefix='gdn35-raw-cpu-') as td:
 raw=Path(td);log=raw/'native.log';lines=[]
 def blob(value,n):return array.array('f',[value]).tobytes()*n
 def write(name,contents):(raw/(name+'.f32')).write_bytes(contents)
 states=[bytes(c.ST*4)]+[blob(n,c.ST) for n in [1,2,3]];conv=[bytes(c.CV*4)]+[blob(n,c.CV) for n in [1,2,3]]
 h=blob(4,3*c.C);y=b''.join(blob(n+4,c.V) for n in range(3));write('t1-normalized-qkv',h);write('t1-output',y)
 for n in range(1,4):write('t1-state-after-'+str(n),states[n]);write('t1-conv-after-'+str(n),conv[n])
 for keep in range(3):
  label='t2-keep'+str(keep)
  cases=[('forward-state-unmodified','forward-state',c.ST,states[0]),('forward-conv-unmodified','forward-conv',c.CV,conv[0]),('forward-normalized-qkv','forward-normalized-qkv',2*c.C,h[:2*c.C*4]),('forward-output','forward-output',2*c.V,y[:2*c.V*4]),('accepted-state','committed-state',c.ST,states[keep]),('accepted-conv','committed-conv',c.CV,conv[keep]),('carry-state','carry-state',c.ST,states[keep+1]),('carry-conv','carry-conv',c.CV,conv[keep+1]),('carry-output','carry-output',c.V,y[keep*c.V*4:(keep+1)*c.V*4])]
  for component,suffix,n,data in cases:
   write(label+'-'+suffix,data);lines.append(f'GDN35_COMPARE case={label} component={component} words={n} differing=0 first={n} max_abs=0 bitwise=1')
 negative=bytearray(y[2*c.V*4:]);negative[:4]=blob(20,1);write('negative-modified-state-output',negative)
 lines+=['GDN35_NEGATIVE modified_state_output_detected=1','GDN35_RESULT comparisons=27 failures=0 synthetic_component_passed=1 model_math_qualified=0 all_owned_allocations_freed=1'];log.write_text('\n'.join(lines)+'\n')
 proof=c.collect(raw,log);assert proof['passed'] and len(proof['comparisons'])==27 and proof['negative_modified_state_detected'];assert proof['owned_allocations_free_independently_qualified'] is False and proof['model_math_qualified'] is False
 log.write_text('\n'.join(lines+[lines[0]])+'\n');reject(lambda:c.collect(raw,log));log.write_text('\n'.join(lines)+'\n')
 path=raw/'t2-keep2-carry-output.f32';before=path.read_bytes();wrong=bytearray(before);wrong[:4]=blob(99,1);path.write_bytes(wrong);reject(lambda:c.collect(raw,log));path.write_bytes(before)
 path=raw/'negative-modified-state-output.f32';before=path.read_bytes();path.write_bytes(y[2*c.V*4:]);reject(lambda:c.collect(raw,log));path.write_bytes(before)
 path=raw/'t2-keep2-committed-state.f32';before=path.read_bytes();path.write_bytes(before[:-4]);reject(lambda:c.collect(raw,log));path.write_bytes(before)
 path=raw/'t2-keep1-forward-output.f32';before=path.read_bytes();bad=bytearray(before);bad[:4]=blob(float('nan'),1);path.write_bytes(bad);reject(lambda:c.collect(raw,log));path.write_bytes(before)
 log.write_text('\n'.join(lines[:-1])+'\n');reject(lambda:c.collect(raw,log))
print('PASS CPU raw collector GDN35:27 full raw bitwise positives, duplicate log, modified raw, ineffective negative, truncated state, nonfinite output, missing terminal summary negatives; fullmodel/free/health remain explicitly unqualified. No device/native/model execution.')
