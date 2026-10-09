#!/usr/bin/env python3
"""Vectorized decoder triangulation with frozen scalar and official gguf-py."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np
from original_gguf_reference import decode as scalar
from original_gguf_vector_decoder_v2 import decode,GEOMETRY


def main():
    here=Path(__file__).resolve().parent;plan=json.loads((here/'original-gguf-reference-foundation-plan-v1.json').read_bytes())
    for path,expected in plan['official_sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    sys.path.insert(0,plan['official_gguf_py']);from gguf import quants,GGMLQuantizationType
    rng=np.random.default_rng(28);checks=[]
    for kind,(elements,width) in GEOMETRY.items():
        packets=[]
        for b in range(16):
            packet=bytearray(rng.integers(0,256,width,dtype=np.uint8).tobytes())
            if kind=='F32':packet=bytearray(struct.pack('<f',(b-7)*.123456789))
            elif kind=='BF16':packet=bytearray(struct.pack('<H',0x3f80+b))
            else:
                packet[:2]=struct.pack('<e',(-1 if b%2 else 1)*2**(b%11-6))
                if kind in ('Q5_1','Q4_K','Q5_K'):packet[2:4]=struct.pack('<e',2**(b%9-10))
            packets.append(packet)
        raw=b''.join(packets)
        for lane in ('original_fp64','ggml_f32'):
            assert np.array_equal(decode(raw,kind,lane),scalar(raw,kind,lane)),(kind,lane)
        expected=quants.dequantize(np.frombuffer(raw,dtype=np.uint8).reshape(16,width),GGMLQuantizationType[kind]).reshape(-1)
        assert np.array_equal(decode(raw,kind,'ggml_f32').astype(np.float32),expected),kind
        try:decode(raw[:-1],kind)
        except ValueError:pass
        else:raise AssertionError('Partial block admitted')
        checks.append(kind)
    receipt={'mode':'CPU_VECTOR_DECODER_TRIANGULATION_ONLY','passed':True,'formats':checks,'blocks':112,'lanes':['original_fp64','ggml_f32'],'official_python_and_frozen_scalar_equal':True,'reference_source_sha256':hashlib.sha256((here/'original_gguf_vector_decoder_v2.py').read_bytes()).hexdigest(),'actual_model_payloads_read':False,'gpu_executed':False,'full_model_math_qualified':False}
    (here/'original-vector-decoder-cpu-receipt-v2.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,sort_keys=True))


if __name__=='__main__':main()
