#!/usr/bin/env python3
"""CPU-only observer contracts with explicit mock backend copies/callbacks."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/mnt/vm_8tb/github/llama.cpp-flashnext')
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def main():
    plan=json.loads((ROOT/'llamacpp/flash-next/activation-logits-observer-build-plan.json').read_text())
    with tempfile.TemporaryDirectory(prefix='llama-activation-cpu-') as temp:
        work=Path(temp)
        for f in ['src/llama-context.cpp','src/llama-memory-recurrent.cpp','ggml/src/ggml-sycl/ggml-sycl.cpp',
                  'tools/server/server-context.cpp','tests/test-backend-ops.cpp']:
            p=work/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((SOURCE/f).read_bytes())
        for spec in plan['patches']:
            path=ROOT/spec['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
            subprocess.run(['git','apply','--check',str(path)],cwd=work,check=True)
            subprocess.run(['git','apply',str(path)],cwd=work,check=True)
        # Actual ggml public declarations, with backend operations mocked explicitly.
        for path in (SOURCE/'ggml/include').glob('*.h'):
            (work/'ggml/include'/path.name).write_bytes(path.read_bytes())
        out=work/'captures';out.mkdir();(work/'arm').write_text('CPU fixture\n')
        (work/'observer.cpp').write_text(r'''
#include "flashnext-activation-diag.h"
#include <cassert>
#include <fstream>
using namespace flashnext_activation_diag;
int reads=0,sets=0,asks=0,answers=0;
ggml_backend_sched_eval_callback installed=nullptr;void* installed_data=nullptr;
extern "C" {
void ggml_backend_tensor_get(const ggml_tensor* t,void* data,size_t offset,size_t size) {
    ++reads;std::memcpy(data,static_cast<const uint8_t*>(t->data)+offset,size);
}
void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t,ggml_backend_sched_eval_callback cb,void* data) {
    ++sets;installed=cb;installed_data=data;
}
size_t ggml_nbytes(const ggml_tensor* t) {
    return size_t(t->ne[0])*4+size_t(t->ne[1]-1)*t->nb[1]+size_t(t->ne[2]-1)*t->nb[2]+size_t(t->ne[3]-1)*t->nb[3];
}
bool ggml_is_contiguous(const ggml_tensor* t) {
    size_t stride=4;for(int d=0;d<4;++d){if(t->nb[d]!=stride)return false;stride*=size_t(t->ne[d]);}return true;
}
}
bool original(ggml_tensor* t,bool ask,void*) {if(ask){++asks;return !std::strcmp(t->name,"other");}++answers;return true;}
ggml_tensor tensor(const char* name,int64_t a,int64_t b,int64_t c,int64_t d,void* data) {
    ggml_tensor t{};t.type=GGML_TYPE_F32;t.op=GGML_OP_ADD;t.buffer=reinterpret_cast<ggml_backend_buffer_t>(1);
    t.ne[0]=a;t.ne[1]=b;t.ne[2]=c;t.ne[3]=d;size_t stride=4;
    for(int i=0;i<4;++i){t.nb[i]=stride;stride*=size_t(t.ne[i]);}
    std::snprintf(t.name,sizeof t.name,"%s",name);t.data=data;return t;
}
int main(int argc,char** argv) {
    assert(argc==2);const bool on=!std::strcmp(argv[1],"on");
    int32_t ids[]={11,12},pos[]={0,1},seq[]={0},batch[]={10,11};int8_t flags[]={0,1};
    const void* ctx=reinterpret_cast<const void*>(0xabc);
    decode(ctx);begin(ctx,2,ids,pos,1,1,seq,batch,flags,1,true);
    std::vector<uint32_t> values(2*10240,0x3f800000u);
    values[10240]=0x80000000u;values[10241]=0x7fc12345u;values[10242]=0xff800000u;
    auto hc=tensor("hc_init",2560,4,2,1,values.data());
    read_plan p;assert(layout(&hc,HC_INPUT,2,1,p) && p.offset==40960 && p.bytes==40960);
    auto bad=hc;bad.flags|=GGML_TENSOR_FLAG_PARAM;assert(!layout(&bad,HC_INPUT,2,1,p));
    bad=hc;bad.type=GGML_TYPE_F16;assert(!layout(&bad,HC_INPUT,2,1,p));
    bad=hc;bad.nb[0]=8;assert(!layout(&bad,HC_INPUT,2,1,p));
    bad=hc;bad.nb[1]+=4;assert(!layout(&bad,HC_INPUT,2,1,p));
    bad=hc;bad.nb[2]=std::numeric_limits<size_t>::max();bad.ne[2]=3;assert(!layout(&bad,HC_INPUT,3,1,p));
    auto state=tensor("state_predelta-1",128,128,48,1,nullptr);
    assert(layout(&state,STATE_S,2,1,p) && p.bytes==3145728);
    state.ne[3]=2;assert(!layout(&state,STATE_S,2,1,p));
    auto R=tensor("cache_r_l1",9,10240,1,1,nullptr);
    auto P=tensor("cache_ple_r_l1",32,10240,1,1,nullptr);
    auto rv=tensor("conv_state_at-1",9,10240,1,1,nullptr);rv.view_src=&R;
    auto pv=tensor("conv_state_at-1",32,10240,1,1,nullptr);pv.view_src=&P;
    if(!on) {
        assert(!current().armed);callback_guard guard(nullptr,&original,nullptr);assert(!guard.active);
        assert(!reads && !sets);std::puts("PASS default-off no backend copy or callback replacement");return 0;
    }
    assert(select(&rv)==STATE_R && select(&pv)==STATE_P);
    auto other=tensor("other",1,1,1,1,nullptr);
    {
        callback_guard guard(nullptr,&original,nullptr);assert(guard.active && sets==1);
        assert(installed(&hc,true,installed_data));assert(installed(&hc,false,installed_data));
        assert(reads==1 && current().observed[HC_INPUT]==1 && answers==0);
        assert(installed(&other,true,installed_data));assert(installed(&other,false,installed_data));
        assert(answers==1 && asks==2 && reads==1);
    }
    assert(sets==2 && installed==&original);
    // Shape failures and PARAM flags must not submit a backend read.
    current().seen={};bad=hc;bad.flags|=GGML_TENSOR_FLAG_PARAM;capture(&bad);assert(reads==1);
    // Exact canonical LE bytes, including signed zero/NaN payload/Inf bits.
    auto file=std::filesystem::directory_iterator(std::getenv("FLASHNEXT_ACTIVATION_DIAG_DIR"));
    std::string name;for(const auto& entry:file)if(entry.path().extension()==".f32")name=entry.path();
    assert(!name.empty());std::ifstream input(name,std::ios::binary);std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),{});
    assert(bytes.size()==40960 && bytes[0]==0 && bytes[3]==0x80 && bytes[4]==0x45 && bytes[7]==0x7f && bytes[11]==0xff);
    // Original host accessor capture is tied to actual output/batch correlation.
    outputs(ctx);current().logit=false;
    sample_begin(ctx,11,0,42,0);std::vector<float> logits(248320,1.0f);host_logits(ctx,logits.data(),248320,11,0);
    assert(current().first_done);sample_end(ctx,7);
    const int before=reads;current().seen={};used().store(MAX_TOTAL);capture(&hc);assert(reads==before);
    begin(ctx,2,ids,pos,2,1,seq,batch,flags,1,true);assert(!current().valid);
    std::puts("PASS CPU layout/parameter/overflow/callback-chain/correlation/canonical-byte/quota contracts");
}
'''.replace('#include <fstream>','#include <fstream>\n#include <filesystem>'))
        command=('g++ -std=c++17 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all -I/work/ggml/include '
                 '/work/observer.cpp -o /work/observer\n'
                 '/work/observer off\n'
                 'FLASHNEXT_ACTIVATION_DIAG=1 FLASHNEXT_ACTIVATION_DIAG_ARM=/work/arm '
                 'FLASHNEXT_ACTIVATION_DIAG_DIR=/work/captures /work/observer on')
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '--entrypoint','/bin/bash','-v',str(work)+':/work',IMAGE,'-lc','set -e\n'+command],check=True)
        files=list(out.glob('*.f32'));assert len(files)==2
        assert sorted(p.stat().st_size for p in files)==[40960,993280]
    print('No real backend, full model, SYCL compilation or diagnostic output-equivalence execution performed.')


if __name__=='__main__':main()
