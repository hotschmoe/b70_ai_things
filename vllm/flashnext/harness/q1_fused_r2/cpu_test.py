"""CPU checks exact compiled key helper and threshold/output semantics."""
from pathlib import Path
import subprocess
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
BUILD=Path('/mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r2')
from oracle import CANARY,cases,packed,reference
rng=np.random.default_rng(20261021)
special=[0,0x80000000,0x3f800000,0xbf800000,0x7f800000,0xff800000,
         0x7fc00001,0x7fffffff,0xffc00001,0xffffffff,1,0x80000001,
         0x7f7fffff,0xff7fffff]
pairs=[(bits,i) for bits in special for i in (0,1,2047,147455,0xfffffffe)]
pairs += [(int(bits),int(i)) for bits,i in zip(rng.integers(0,2**32,4096,dtype=np.uint32),
                                              rng.integers(0,147456,4096,dtype=np.uint32))]
lines=''.join(f'{bits} {i}\n' for bits,i in pairs)
proc=subprocess.run([str(BUILD/'cpu_keys')],input=lines,text=True,capture_output=True,check=True)
got=list(map(int,proc.stdout.splitlines()))
expected=[]
for bits,i in pairs:
    order=bits ^ (0xffffffff if bits & 0x80000000 else 0x80000000)
    u=((order<<32)|(0xffffffff ^ i)) ^ (1<<63)
    expected.append(u if u<2**63 else u-2**64)
assert got==expected
assert all(value>-(2**63) for value in got)
def bitonic_ids(selected,width,sort_n=4096):
    buf=np.full(sort_n,2147483647,dtype=np.int32)
    buf[:width]=(np.uint64(0xffffffff) ^ (selected[:width].view(np.uint64) & np.uint64(0xffffffff))).astype(np.int32)
    idx=np.arange(sort_n,dtype=np.int32)
    k=2
    stages=0
    while k<=sort_n:
        j=k>>1
        while j:
            low=idx[(idx^j)>idx]
            high=low^j
            a=buf[low].copy();b=buf[high].copy()
            ascending=(low&k)==0
            buf[low]=np.where(ascending,np.minimum(a,b),np.maximum(a,b))
            buf[high]=np.where(ascending,np.maximum(a,b),np.minimum(a,b))
            stages+=1
            j>>=1
        k<<=1
    assert stages==78
    return buf[:width]
count=0
for name,bits,n,width in cases():
    n=max(0,min(n,len(bits)))
    key=packed(bits[:n]) ^ np.uint64(1<<63)
    mapped=key.view(np.int64)
    values=np.sort(mapped)[::-1]
    for selected_k in (0,1,17,2051):
        w=max(0,min(width,n,2051,selected_k))
        out=np.full(2051,CANARY,dtype=np.int32)
        if w:
            out[:w]=bitonic_ids(values,w)
        np.testing.assert_array_equal(out,reference(bits,n,min(width,selected_k)),err_msg=name)
        count+=1
print(f'FUSED_R2_CPU_PASS compiled_key_pairs={len(pairs)} bitonic_cases={count} stages=78 SLM=16384 invalidMIN=strict guarded_tails=exact',flush=True)
