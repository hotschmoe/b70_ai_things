#!/usr/bin/env python3
"""Actual independent Q8 packet/reference helpers; no SYCL or GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def main():
    with tempfile.TemporaryDirectory(prefix='mirror-q8-reference-') as tmp:
        tmp = Path(tmp)
        (tmp / 'stage_mirror_q8_reference.hpp').write_bytes((HERE / 'stage_mirror_q8_reference.hpp').read_bytes())
        (tmp / 'test.cpp').write_text(r'''
#include "stage_mirror_q8_reference.hpp"
#include <cassert>
#include <cstdio>
int main(){using namespace mirror_reference;
    int finite=0;for(unsigned bits=0;bits<65536;++bits) if((bits&0x7c00)!=0x7c00){float value=half_value(uint16_t(bits));assert(half_bits(value)==bits);++finite;}assert(finite==63488);
    std::vector<float> x(32);x[0]=-127;x[1]=127;auto q=q8_1(x);assert(q.size()==36 && load_half(q.data())==0x3c00 && load_half(q.data()+2)==0 && q[4]==129 && q[5]==127);for(size_t i=6;i<q.size();++i) assert(q[i]==0);
    std::vector<float> zero(64);auto z=q8_1(zero);for(auto b:z) assert(b==0);
    std::vector<float> high(32,100000.0f);auto h=q8_1(high);assert(load_half(h.data()+2)==0x7bff);for(size_t i=4;i<h.size();++i) assert(h[i]==127);
    std::vector<float> values(32);for(int i=0;i<32;++i) values[i]=(i%2 ? -1:1)*float(i+1)*0.00100013f;auto packet=q8_1(values);std::vector<float> weights(32,1.0f);double expected=0;float d=half_value(load_half(packet.data()));for(int i=0;i<32;++i){int code=packet[4+i];if(code>=128) code-=256;expected+=double(d)*code;}assert(dot(weights.data(),nullptr,8,packet.data(),32)==expected);
    // Type7 Q5_1 does not use the original activation sum. An unused original
    // row pointer cannot affect that selected format's independent reference.
    assert(dot(weights.data(),nullptr,7,packet.data(),32)==expected);
    // Optional type6 Q5_0 adds the declared original-sum correction.
    uint8_t row[22]={};row[0]=0;row[1]=0x40;double correction=32.0*(expected-half_value(load_half(packet.data()+2)));assert(dot(weights.data(),row,6,packet.data(),32)==expected+correction);
    auto same=metric(weights,weights);assert(same.finite && same.nmse==0 && same.linf==0);auto corrupt=weights;corrupt[0]+=0.01f;auto bad=metric(corrupt,weights);assert(bad.nmse>1e-6 || bad.linf>1e-4);
    bool rejected=false;try{q8_1(std::vector<float>(33));}catch(const std::runtime_error&){rejected=true;}assert(rejected);
    std::printf("PASS independent Q8_1 finite-half, source packet, sum/clamp, selectedQ5_1/Q8_0 dot and optionalQ5_0 correction references\n");}
''')
        command = 'g++ -std=c++17 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all -I/work /work/test.cpp -o /work/test && /work/test'
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '--entrypoint','/bin/bash','-v',str(tmp)+':/work',IMAGE,'-lc',command],check=True)
    print(json.dumps({'passed':True,'mode':'CPU_ONLY','reference_source_sha256':hashlib.sha256((HERE/'stage_mirror_q8_reference.hpp').read_bytes()).hexdigest(),'gpu_qualified':False},sort_keys=True))


if __name__ == '__main__':
    main()
