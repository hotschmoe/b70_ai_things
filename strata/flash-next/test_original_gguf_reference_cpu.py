#!/usr/bin/env python3
"""Bounded independent decoder / role controls; no full-model forward or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import subprocess
import os
import shlex
import tempfile
import numpy as np
from original_gguf_reference import decode,GEOMETRY,OriginalGguf
from test_original_reader_admission_cpu import run as reader_controls

HERE=Path(__file__).resolve().parent


def main():
    plan=json.loads((HERE/'original-gguf-reference-foundation-plan-v1.json').read_bytes())
    for path,expected in plan['official_sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    sys.path.insert(0,plan['official_gguf_py'])
    from gguf import quants,GGMLQuantizationType
    library=Path(plan['official_cpu_scalar_library']);assert hashlib.sha256(library.read_bytes()).hexdigest()==plan['official_cpu_scalar_library_sha256']
    work=tempfile.TemporaryDirectory(prefix='original-ggml-cpu-');directory=Path(work.name);binary=directory/'decoder'
    official=Path(plan['official_gguf_py']).parent
    command=['icpx','-std=c++17','-O2','-I'+str(official/'ggml/include'),'-I'+str(official/'ggml/src'),str(HERE/'original_ggml_decode_probe.cpp'),str(library),'-lm','-lpthread','-ldl','-o',str(binary)]
    command=['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
             '-v',str(official.parent)+':'+str(official.parent)+':ro','-v',str(HERE)+':'+str(HERE)+':ro',
             '-v',str(directory)+':'+str(directory),
             'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7',shlex.join(command)]
    built=subprocess.run(command,capture_output=True,text=True)
    if built.returncode:raise RuntimeError(built.stdout+built.stderr)
    rng=np.random.default_rng(20261009);rows=[];controls=[]
    for kind,(elements,width) in GEOMETRY.items():
        for case in range(12):
            packet=bytearray(rng.integers(0,256,width,dtype=np.uint8).tobytes())
            if kind=='F32':packet=bytearray(rng.normal(size=1).astype('<f4').tobytes())
            elif kind=='BF16':packet=bytearray(struct.pack('<H',0x3f80+case))
            else:
                packet[:2]=struct.pack('<e',(-1 if case%2 else 1)*2**(case%9-5))
                if kind in ('Q5_1','Q4_K','Q5_K'):packet[2:4]=struct.pack('<e',2**(case%7-8))
            got=decode(bytes(packet),kind,'ggml_f32').astype(np.float32)
            expected=quants.dequantize(np.frombuffer(packet,dtype=np.uint8).reshape(1,width),GGMLQuantizationType[kind]).reshape(-1)
            assert np.array_equal(got,expected),(kind,case,np.max(np.abs(got-expected)))
            if kind not in ('F32','BF16'):
                (directory/'packet.bin').write_bytes(packet)
                subprocess.run([str(binary),kind,str(directory/'packet.bin'),str(directory/'official.bin')],check=True,capture_output=True)
                direct=np.frombuffer((directory/'official.bin').read_bytes(),dtype='<f4')
                assert np.array_equal(got,direct),(kind,case,'official C scalar differs')
            original=decode(bytes(packet),kind)
            assert np.isfinite(original).all() and original.size==elements
        rows.append({'format':kind,'official_python_decoder_byte_equivalent_f32':True,'blocks':12,'official_c_scalar_checked':kind not in ('F32','BF16')})
    # Independent labelled packets make plausible rival indexing/formulas observable.
    q5=struct.pack('<eeI',0.5,-2,1<<20)+bytes([0x21]*16)
    out=decode(q5,'Q5_1');assert out[20]==7 and out[4]==-1.5
    assert not np.array_equal(out,np.concatenate([out[16:],out[:16]]));controls.append('q5_1_half_order')
    assert not np.array_equal(out,out+2);controls.append('q5_1_minimum_omitted')
    q8=struct.pack('<e',0.25)+bytes([0x80]+[0]*31);assert decode(q8,'Q8_0')[0]==-32;controls.append('q8_signedness')
    iq=struct.pack('<e',1)+bytes([0xf0]+[0]*15);assert decode(iq,'IQ4_NL')[0]==-127 and decode(iq,'IQ4_NL')[16]==113;controls.append('iq4_nonlinear_table')
    q4=bytearray(144);q4[:4]=struct.pack('<ee',0.5,0);q4[4]=0x80;q4[12]=2;q4[80]=3
    assert decode(q4,'Q4_K')[128]==51;controls.append('q4_k_split_scale_bits')
    q5k=bytearray(176);q5k[:4]=struct.pack('<ee',0.25,0);q5k[13]=1;q5k[16]=32;q5k[112]=0x10
    assert decode(q5k,'Q5_K')[160]==4.25;controls.append('q5_k_high_bit_group')
    lanes=struct.pack('<eeI',65504,2**-24,0)+bytes([0xff]*16)
    assert not np.array_equal(decode(lanes,'Q5_1'),decode(lanes,'Q5_1','ggml_f32'));controls.append('original_dequant_vs_f32_storage')
    for kind,(_,width) in GEOMETRY.items():
        try:decode(b'\0'*(width-1),kind)
        except ValueError:pass
        else:raise AssertionError('Partial packet admitted '+kind)
    controls.append('partial_blocks_all_formats')
    # Geometry / complete-role rejection uses inventory metadata only, no model reads.
    inventory=json.loads(Path(plan['inventory']).read_bytes());probe=OriginalGguf.__new__(OriginalGguf);probe.manifest=plan
    probe.files=[f for f in inventory['files'] if '/UD-Q4_K_XL/' in f['path']]
    probe.tensors={t['name']:(f,t,[]) for f in probe.files for t in f['tensors']};probe.validate_roles()
    for label,mutate in [('missing_role',lambda p:p.tensors.pop('blk.0.attn_qkv.weight')),
                         ('wrong_shape',lambda p:p.tensors['blk.0.attn_qkv.weight'][1]['shape_ggml_order'].__setitem__(0,1280)),
                         ('wrong_type',lambda p:p.tensors['blk.1.ple_value.weight'][1].__setitem__('type','BF16')),
                         ('wrong_architecture',lambda p:p.files[0]['metadata']['general.architecture'].__setitem__('value','qwen3next'))]:
        bad=copy.deepcopy(probe);mutate(bad)
        try:bad.validate_roles()
        except ValueError:controls.append(label)
        else:raise AssertionError('Wrong role/geometry admitted '+label)
    for kind in ('Q4_K','Q5_0','Q6_K'):
        bad=copy.deepcopy(probe);bad.tensors['blk.48.mtp_forbidden.weight']=({}, {'type':kind,'shape_ggml_order':[2560,640]},[])
        try:bad.validate_roles()
        except ValueError:controls.append('mtp_extra_'+kind)
        else:raise AssertionError('MTP sidecar role admitted')
    result={'mode':'CPU_REFERENCE_FOUNDATION_ONLY','passed':True,'formats':rows,'decoder_blocks':84,'official_c_scalar_blocks':60,'cpu_probe_binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'official_scalar_library_sha256':plan['official_cpu_scalar_library_sha256'],'negative_controls':controls,'tensor_role_count':plan['tensor_count'],'gpu_executed':False,'actual_payloads_read':False,'cpu_compile_image':'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','compile_devices_exposed':False,'decoder_source_sha256':hashlib.sha256((HERE/'original_gguf_reference.py').read_bytes()).hexdigest(),'cpu_probe_source_sha256':hashlib.sha256((HERE/'original_ggml_decode_probe.cpp').read_bytes()).hexdigest(),'full_model_math_qualified':False}
    result.update(reader_controls())
    (HERE/'original-gguf-reference-foundation-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
