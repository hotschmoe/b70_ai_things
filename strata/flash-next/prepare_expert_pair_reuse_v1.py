"""Pure source construction for original-quant E2 weight reuse; no compilation."""
from pathlib import Path
import argparse,difflib,hashlib,json
SOURCE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww/source/sycl/src/kernels/cuda/iq_kernels.dp.cpp')
HERE=Path(__file__).resolve().parent
TARGET='sycl/src/kernels/cuda/iq_kernels.dp.cpp'
def once(source,old,new):
 if source.count(old)!=1:raise ValueError('Unique actual52 source seam required '+old[:90])
 return source.replace(old,new,1)
def construct(source):
 a=source.index('template<int TY> struct S27;');b=source.index('\ntemplate <int TG, bool LX, int RPW, bool TS>',a)
 typed=source[a:b].replace('S27','ExpertPairV1')
 helper='''
// Original type load/apply bodies from the disabled S27 reference, no AMD launch/LDS machinery.
'''+typed+'''
template <int TY, int LANES>
__dpct_inline__ void row_dot_pair_v1(const uint8_t* row,const block_q8_1* x0,const block_q8_1* x1,int nb,int sub,float* out){
    using F=Fmt<TY>;using P=ExpertPairV1<TY>;
    float a=0.0f,b=0.0f;
    for(int k=sub;k<nb*F::ipb;k+=LANES){
        const int kbx=k/F::ipb,iqs=F::step*(k%F::ipb);
        const auto weights=P::load(row,kbx,iqs);
        a+=P::apply(weights,x0+kbx*(F::qk/32),iqs);
        b+=P::apply(weights,x1+kbx*(F::qk/32),iqs);
    }
    out[0]=lanes_sum<LANES>(a);out[1]=lanes_sum<LANES>(b);
}
'''
 anchor='template <int TY, int LANES>\n__dpct_inline__ void row_entries'
 source=once(source,anchor,helper+'\n'+anchor)
 source=once(source,'const void* grid = nullptr) {\n    int e = e0;','const void* grid = nullptr, bool pair_reuse = false) {\n    if constexpr(TY==12 || TY==13){\n        if(pair_reuse && e1-e0==2){\n            float paired[2];row_dot_pair_v1<TY,LANES>(wr,x+size_t(ent_idx[e0])*x_stride,x+size_t(ent_idx[e0+1])*x_stride,nb,sub,paired);\n            if(sub==0){dst[size_t(e0)*dst_stride]=paired[0];dst[size_t(e0+1)*dst_stride]=paired[1];}\n            return;\n        }\n    }\n    int e = e0;')
 a=source.index('__dpct_inline__ void native_gu_kernel(');b=source.index('    const int ng = *n_groups;',a)
 signature=source[a:b]
 source=once(source,signature,signature.replace('float *__restrict__ up) {','float *__restrict__ up, bool pair_reuse = false) {'))
 source=once(source,'(is_up ? up : gate) + r, (size_t) L.n_ff, grid);','(is_up ? up : gate) + r, (size_t) L.n_ff, grid, pair_reuse);')
 source=once(source,'NativeExpertLayout L, float *__restrict__ out) {\n    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();\n    const int ng = *n_groups;   // #363: the group stride, as native_gu_kernel','NativeExpertLayout L, float *__restrict__ out, bool pair_reuse = false) {\n    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();\n    const int ng = *n_groups;   // #363: the group stride, as native_gu_kernel')
 anchor='''        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        int e = e0;
        if constexpr (Multi<TD>::has) {'''
 replacement='''        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if constexpr(TD==7 || TD==8){
            if(pair_reuse && e1-e0==2){
                float paired[2];row_dot_pair_v1<TD,LN>(wr,hq+size_t(e0)*hb,hq+size_t(e0+1)*hb,nb,sub,paired);
                if(sub==0){out[size_t(ent_dst[e0])*L.n_embd+r]=paired[0];out[size_t(ent_dst[e0+1])*L.n_embd+r]=paired[1];}
                continue;
            }
        }
        int e = e0;
        if constexpr (Multi<TD>::has) {'''
 source=once(source,anchor,replacement)
 source=once(source,'inline int gu_lanes()','inline bool expert_pair_reuse_enabled_v1(){static const bool on=[](){const char* v=std::getenv("STRATA_EXPERT_PAIR_REUSE");return v && v[0]==\'1\' && v[1]==\'\\0\';}();return on;}\ninline int gu_lanes()')
 for role,param in [('gu','float *gate, float *up'),('down','float *out')]:
  anchor=param+') {\n    constexpr int ROWS = 256 / LN;'
  source=once(source,anchor,param+') {\n    const bool pair_reuse=expert_pair_reuse_enabled_v1();\n    constexpr int ROWS = 256 / LN;')
 source=once(source,'native_gu_kernel<TG, LN>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);','native_gu_kernel<TG, LN>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, pair_reuse);')
 source=once(source,'native_down_kernel<TD, LN>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);','native_down_kernel<TD, LN>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, pair_reuse);')
 return source

def prepare(output):
 plan=json.loads((HERE/'expert-pair-reuse-source-proposal-v1.json').read_bytes())
 raw=SOURCE.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=plan['original_source_sha256']:raise ValueError('Actual original52 source changed')
 for path,want in plan['own_source_sha256'].items():
  if hashlib.sha256((HERE.parents[1]/path).read_bytes()).hexdigest()!=want:raise ValueError('Candidate own source changed '+path)
 patched=construct(raw.decode('ascii'))
 if hashlib.sha256(patched.encode('ascii')).hexdigest()!=plan['expected_patched_sha256']:raise ValueError('Exact candidate construction changed')
 output=Path(output)
 if not output.is_absolute()or output.resolve()!=output or output.exists():raise ValueError('New canonical source output required')
 output.mkdir();p=output/'iq_kernels.dp.cpp';p.write_text(patched,encoding='ascii')
 proof={'schema':1,'original_path':str(SOURCE),'original_sha256':hashlib.sha256(raw).hexdigest(),'patched_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'patched_path':str(p),'source_proposal_sha256':hashlib.sha256((HERE/'expert-pair-reuse-source-proposal-v1.json').read_bytes()).hexdigest(),'compile_executed':False,'GPU_executed':False,'model_payload_read':False,'exported_API_changed':False}
 (output/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='ascii');return proof
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.output),ensure_ascii=True))
