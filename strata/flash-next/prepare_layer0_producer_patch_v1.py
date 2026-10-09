#!/usr/bin/env python3
"""NEW0023 producer draft against frozen20+21; never edits either source."""
import argparse
import difflib
import hashlib
from pathlib import Path
from prepare_layer0_capture_patch_v1 import construct as construct21,BASE,once
HERE=Path(__file__).resolve().parent


def construct():
    _,source21,added21=construct21();base={**source21,**added21}
    for p in ['sycl/src/kernels/cuda/shared_expert.dp.cpp','sycl/src/kernels/cuda/iq_kernels.dp.cpp']:base[p]=(BASE/p).read_text()
    changed={}
    p='sycl/include/strata/core/layer0_numerical_observer.hpp';changed[p]=(HERE/'layer0_numerical_observer_v2.hpp').read_text()
    p='sycl/src/kernels/cuda/shared_expert.dp.cpp';s=base[p]
    s=once(s,'int64_t n_ff, void* stream, const void* x_q8_1_ready, int lfuse) {','int64_t n_ff, void* stream, const void* x_q8_1_ready, int lfuse, const numerical_diag::hooks* observer) {')
    s=once(s,'    const int n = (int) (n_ff * n_tok);\n    if (fused_swiglu_q81_enabled()) {','    if(observer && observer->shared_gu)observer->shared_gu(observer->user,gate,up,n_tok,int(n_ff),stream);\n    const int n = (int) (n_ff * n_tok);\n    if (fused_swiglu_q81_enabled()) {')
    s=once(s,'    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);','    if(observer && observer->shared_hq)observer->shared_hq(observer->user,nw.q8_1,n_tok,int(n_ff),stream);\n    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);')
    changed[p]=s
    p='sycl/src/kernels/cuda/iq_kernels.dp.cpp';s=base[p]
    s=once(s,'int64_t grid_groups) {\n    if (cap_groups <= 0 || cap_entries <= 0) return;', 'int64_t grid_groups, const numerical_diag::hooks* observer) {\n    if(observer && g_exp_phase!=0)throw std::invalid_argument("Layer0 producer bench phase is unsupported");\n    if (cap_groups <= 0 || cap_entries <= 0) return;')
    anchor='    check("native_expert_grouped/gu");\n    const long long nh ='
    inserted='''    check("native_expert_grouped/gu");
    numerical_diag::expert_view observed{gate,up,hq,grp_ptr,grp_start,n_groups,ent_dst,ent_tok,cap_entries,L.n_ff};
    if(observer && observer->expert_gu)observer->expert_gu(observer->user,observed,stream);
    const long long nh ='''
    s=once(s,anchor,inserted)
    anchor='    check("native_expert_grouped/swiglu");\n    }'
    s=once(s,anchor,'''    check("native_expert_grouped/swiglu");
    if(observer && observer->expert_hq)observer->expert_hq(observer->user,observed,stream);
    }''');changed[p]=s
    p='sycl/src/core/verify.cpp';s=base[p]
    anchor='                shared_expert_multi(n, xm, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data,'
    before='''                numerical_diag::hooks shared_hooks;
                if(l0_capture) {
                    shared_hooks.user=layer0_snapshot_.get();
                    shared_hooks.shared_gu=[](void*user,const float*g,const float*u,int rows,int ffn,void*stream){static_cast<layer0_diag::snapshot*>(user)->observe_shared_gu(g,u,rows,ffn,*strata::q_of(stream));};
                    shared_hooks.shared_hq=[](void*user,const void*hq,int rows,int ffn,void*stream){static_cast<layer0_diag::snapshot*>(user)->observe_shared_hq(hq,rows,ffn,*strata::q_of(stream));};
                }
'''
    s=once(s,anchor,before+anchor)
    anchor='                                    (sg_ready ? 1 : 0) | (lfuse_on(n) && g_lfuse_pair() ? 2 : 0));'
    s=once(s,anchor,'                                    (sg_ready ? 1 : 0) | (lfuse_on(n) && g_lfuse_pair() ? 2 : 0),l0_capture?&shared_hooks:nullptr);')
    anchor='                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,'
    before='''                struct ProducerContext {layer0_diag::snapshot* snapshot; numerical_diag::entry_binding binding;};
                ProducerContext context{};numerical_diag::hooks expert_hooks;
                const bool l0_producer=layer0_first_&&layer0_snapshot_&&T==1&&l==0&&grp==0;
                if(l0_producer) {
                    context.snapshot=layer0_snapshot_.get();
                    context.binding={ids_,hits_.d_res+l*g.n_expert,hits_.stage_mirror?hits_.stage_mirror->layer_table(l):nullptr,slot_off_d_,hits_.cache_base,hits_.n_slots,hits_.blob};
                    expert_hooks.user=&context;
                    expert_hooks.expert_gu=[](void*user,const numerical_diag::expert_view&view,void*stream){auto&c=*static_cast<ProducerContext*>(user);c.snapshot->observe_expert(view,c.binding,false,*strata::q_of(stream));};
                    expert_hooks.expert_hq=[](void*user,const numerical_diag::expert_view&view,void*stream){auto&c=*static_cast<ProducerContext*>(user);c.snapshot->observe_expert(view,c.binding,true,*strata::q_of(stream));};
                }
'''
    s=once(s,anchor,before+anchor)
    anchor='                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, dst_buf, cs, gy);'
    s=once(s,anchor,'                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, dst_buf, cs, gy,l0_producer?&expert_hooks:nullptr);')
    changed[p]=s
    added={'sycl/include/strata/kernels/layer0_producer_hooks.hpp':(HERE/'layer0_producer_hooks_v1.hpp').read_text(),
           'sycl/include/strata/kernels/shared_expert.hpp':(HERE/'layer0_shared_expert_shadow_v1.hpp').read_text(),
           'sycl/include/strata/kernels/iq_kernels.hpp':(HERE/'layer0_iq_kernels_shadow_v1.hpp').read_text()}
    return base,changed,added


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError('Preserve existing0023 generation')
    base,changed,added=construct();diff=[]
    for path,text in {**changed,**added}.items():diff.extend(difflib.unified_diff(base.get(path,'').splitlines(True),text.splitlines(True),fromfile='a/'+path if path in base else '/dev/null',tofile='b/'+path))
    args.output.write_text(''.join(diff));print(hashlib.sha256(args.output.read_bytes()).hexdigest()+' '+str(args.output))


if __name__=='__main__':main()
