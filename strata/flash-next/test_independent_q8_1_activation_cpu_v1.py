#!/usr/bin/env python3
"""Independent packet properties / official CPU variant; no GPU/model payloads."""
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import numpy as np
from independent_q8_1_activation_v1 import encode,decode,quantization_cost,require_packet_exact,NATIVE,OFFICIAL,QFUSE


def f32(v):
    if abs(v)>float(np.finfo(np.float32).max):return math.copysign(math.inf,v)
    return struct.unpack('<f',struct.pack('<f',v))[0]


def scalar_native(raw):
    values=struct.unpack('<'+'f'*(len(raw)//4),raw);result=bytearray()
    for start in range(0,len(values),32):
        x=values[start:start+32];maximum=max(abs(v) for v in x);scale=min(65504.,f32(maximum/127))
        if maximum and not scale:raise ValueError('Unsupported underflow')
        lanes=list(x)
        for mask in (16,8,4,2,1):lanes=[f32(lanes[i]+lanes[i^mask]) for i in range(32)]
        total=max(-65504.,min(65504.,lanes[0]));result.extend(struct.pack('<ee',scale,total))
        for v in x:
            q=0 if maximum==0 else f32(v/scale);code=math.copysign(math.floor(abs(q)+.5),q)
            result.append(int(max(-127,min(127,code)))&255)
    return bytes(result)


def main():
    here=Path(__file__).resolve().parent;foundation=json.loads((here/'original-gguf-reference-foundation-plan-v1.json').read_bytes())
    lib=Path(foundation['official_cpu_scalar_library']);assert hashlib.sha256(lib.read_bytes()).hexdigest()==foundation['official_cpu_scalar_library_sha256']
    official=Path(foundation['official_gguf_py']).parent;controls=[];cases=[];rng=np.random.default_rng(20261009)
    with tempfile.TemporaryDirectory(prefix='official-q81-cpu-') as temporary:
        work=Path(temporary);binary=work/'official-q81'
        compile_cmd=['icpx','-std=c++17','-O2','-I'+str(official/'ggml/include'),'-I'+str(official/'ggml/src'),str(here/'official_q8_1_cpu_probe_v1.cpp'),str(lib),'-lm','-lpthread','-ldl','-o',str(binary)]
        command=['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
                 '-v',str(official.parent)+':'+str(official.parent)+':ro','-v',str(here)+':'+str(here)+':ro','-v',str(work)+':'+str(work),
                 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7',shlex.join(compile_cmd)]
        compiled=subprocess.run(command,capture_output=True,text=True)
        if compiled.returncode:raise RuntimeError(compiled.stdout+compiled.stderr)
        profiles={'zero':np.zeros(32,dtype=np.float32),'ties':np.array([127.,.5,-.5,1.5,-1.5]+[0]*27,dtype=np.float32),
                  'cancel':np.array([1e5,-1e5,.01,-.02]*8,dtype=np.float32),'random':rng.normal(size=32).astype(np.float32),
                  'scale_clamp':np.full(32,1e9,dtype=np.float32),'sum_clamp':np.full(32,4096,dtype=np.float32),
                  'small_normal':rng.normal(size=32).astype(np.float32)*np.float32(1e-20)}
        official_count=0
        for profile,block in profiles.items():
            for rows,cols in [(1,640),(2,2560),(4,6144),(8,2560)]:
                raw=np.tile(block,rows*cols//32).astype('<f4').tobytes();packet=encode(raw,rows,cols)
                assert packet==scalar_native(raw),(profile,rows,cols)
                assert require_packet_exact(raw,packet,rows,cols)['packet_exact']
                decoded=decode(packet,rows,cols);assert np.isfinite(decoded['scale']).all() and np.isfinite(decoded['stored_sum']).all()
                cases.append({'profile':profile,'rows':rows,'cols':cols,'packet_bytes':len(packet),'exact_independent_scalar':True,'quantization_cost':quantization_cost(raw,packet,rows,cols)})
                if profile in ('zero','ties','cancel','random','small_normal'):
                    (work/'input.bin').write_bytes(raw);subprocess.run([str(binary),str(work/'input.bin'),str(work/'output.bin')],check=True,capture_output=True)
                    assert encode(raw,rows,cols,OFFICIAL)==(work/'output.bin').read_bytes();official_count+=1
        ties=decode(encode(profiles['ties'].tobytes(),1,32),1,32)['codes'][0];assert list(ties[:5])==[127,1,-1,2,-2];controls.append('half_away_not_even')
        raw=profiles['random'].tobytes();assert encode(raw,1,32)!=encode(raw,1,32,OFFICIAL);controls.append('official_integer_sum_not_native_raw_sum')
        # Find an explicit F32 division-versus-reciprocal half-boundary witness.
        witness=None
        for attempt in range(2048):
            amax=np.float32(10**rng.uniform(-3,3));d=np.float32(amax/np.float32(127));j=int(rng.integers(0,126));x=np.float32(d*np.float32(j+.5));block=np.zeros(32,dtype=np.float32);block[0]=amax;block[1]=x
            native=decode(encode(block.tobytes(),1,32),1,32)['codes'];official_codes=decode(encode(block.tobytes(),1,32,OFFICIAL),1,32)['codes']
            if not np.array_equal(native,official_codes):witness={'amax':float(amax),'x':float(x),'native_code':int(native[0,1]),'official_code':int(official_codes[0,1])};break
        assert witness is not None
        (work/'input.bin').write_bytes(block.tobytes());subprocess.run([str(binary),str(work/'input.bin'),str(work/'output.bin')],check=True,capture_output=True)
        assert encode(block.tobytes(),1,32,OFFICIAL)==(work/'output.bin').read_bytes();official_count+=1
        controls.append('division_vs_reciprocal_code_witness')
        normal=profiles['random'].tobytes();assert encode(normal,1,32,QFUSE)==encode(normal,1,32,NATIVE);controls.append('qfuse_matches_only_finite_unclamped_range')
        overflow=profiles['sum_clamp'].tobytes();qfused=decode(encode(overflow,1,32,QFUSE),1,32)
        assert not np.isfinite(qfused['stored_sum']).all() and np.isfinite(decode(encode(overflow,1,32),1,32)['stored_sum']).all();controls.append('qfuse_unclamped_sum_is_distinct')
        # Stored half scale must never be fed back into code generation.
        half_scale_witness=False
        for attempt in range(512):
            block=rng.normal(size=32).astype(np.float32);packet=encode(block.tobytes(),1,32);d16=np.float32(decode(packet,1,32)['scale'][0])
            quotient=block/d16;wrong=np.clip(np.sign(quotient)*np.floor(np.abs(quotient.astype(np.float64))+.5),-127,127).astype(np.int8)
            if not np.array_equal(wrong,decode(packet,1,32)['codes'][0]):half_scale_witness=True;break
        assert half_scale_witness;controls.append('codes_use_f32_scale_not_stored_half')
        reduction=np.array([1e8]+[1.]*30+[-1e8],dtype=np.float32)
        sequential=np.float32(0)
        for value in reduction:sequential=np.float32(sequential+value)
        assert float(np.float16(sequential))!=decode(encode(reduction.tobytes(),1,32),1,32)['stored_sum'][0];controls.append('xor_tree_not_sequential_sum')
        valid=encode(raw,1,32);corrupt=bytearray(valid);corrupt[4]^=1
        for label,fn in [('altered_packet',lambda:require_packet_exact(raw,bytes(corrupt),1,32)),('short_extent',lambda:encode(raw[:-1],1,32)),
                         ('wrong_shape',lambda:encode(raw,1,31)),('nonfinite_input',lambda:encode(np.full(32,np.nan,dtype='<f4').tobytes(),1,32)),
                         ('nan_reduction_from_finite_input',lambda:encode(np.tile(np.repeat(np.array([np.finfo(np.float32).max,-np.finfo(np.float32).max],dtype=np.float32),8),2).astype('<f4').tobytes(),1,32)),
                         ('nonzero_scale_underflow',lambda:encode(np.full(32,np.nextafter(np.float32(0),np.float32(1)),dtype='<f4').tobytes(),1,32))]:
            try:fn()
            except ValueError:controls.append(label)
            else:raise AssertionError('Invalid finite packet admitted '+label)
        for label,offset in [('altered_scale',0),('altered_sum',2)]:
            corrupted=bytearray(valid);corrupted[offset]^=1
            try:require_packet_exact(raw,bytes(corrupted),1,32)
            except ValueError:controls.append(label)
            else:raise AssertionError('Wrong packet field admitted')
        two_raw=profiles['random'].tobytes()+profiles['ties'].tobytes();two_packet=encode(two_raw,2,32)
        try:require_packet_exact(two_raw,two_packet[36:]+two_packet[:36],2,32)
        except ValueError:controls.append('swapped_row_packets')
        else:raise AssertionError('Wrong row order admitted')
        result={'mode':'CPU_INDEPENDENT_Q8_1_PACKET_CONTRACT_ONLY','passed':True,'cases':cases,'case_count':28,'official_cpu_variant_cases':official_count,
            'negative_and_rival_controls':controls,'division_witness':witness,'source_sha256':hashlib.sha256((here/'independent_q8_1_activation_v1.py').read_bytes()).hexdigest(),
            'official_scalar_library_sha256':foundation['official_cpu_scalar_library_sha256'],'compile_devices_exposed':False,'compile_image':'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','official_probe_binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'actual_model_payloads_read':False,'gpu_executed':False,
            'raw_fused_hidden_observed':False,'full_model_math_qualified':False}
        (here/'independent-q8-1-activation-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},sort_keys=True))


if __name__=='__main__':main()
