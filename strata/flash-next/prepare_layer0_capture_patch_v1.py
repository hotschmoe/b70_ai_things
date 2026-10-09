#!/usr/bin/env python3
"""Reconstruct0021 against immutable20; writes only a NEW patch draft."""
import argparse
import difflib
import hashlib
from pathlib import Path

BASE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T112318Z-4uag6j5x/source')
HERE=Path(__file__).resolve().parent


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Source anchor differs: '+old[:96])
    return text.replace(old,new,1)


def construct():
    paths=['sycl/include/strata/core/verify.hpp','sycl/src/core/verify.cpp','sycl/src/program/generate.cpp']
    original={p:(BASE/p).read_text() for p in paths};changed=dict(original)
    p=paths[0];s=changed[p]
    s=once(s,'namespace fidelity_diag { class snapshot; }','namespace fidelity_diag { class snapshot; }\nnamespace layer0_diag { class snapshot; }')
    s=once(s,'    void release_fidelity_observer();','    void release_fidelity_observer();\n    void select_layer0_first(bool on);\n    bool dump_layer0_first(int rows,int64_t pos,int token,std::string& err);\n    void release_layer0_observer();')
    s=once(s,'    bool fidelity_first_=false;','    bool fidelity_first_=false;\n    std::shared_ptr<layer0_diag::snapshot> layer0_snapshot_;\n    bool layer0_first_=false;\n    dpct::experimental::command_graph_exec_ptr layer0_exec_[4] = {};')
    changed[p]=s
    p=paths[1];s=changed[p]
    s=once(s,'#include "strata/core/fidelity_observer.hpp"','#include "strata/core/fidelity_observer.hpp"\n#include "strata/core/layer0_numerical_observer.hpp"')
    anchor='    for(auto& graph:fidelity_exec_) { if(graph) delete graph; graph=nullptr; }\n    fidelity_snapshot_.reset();\n    hits_.stage_mirror.reset();'
    s=once(s,anchor,'    for(auto& graph:layer0_exec_) { if(graph)delete graph;graph=nullptr; }\n    for(auto& graph:fidelity_exec_) { if(graph) delete graph; graph=nullptr; }\n    layer0_snapshot_.reset();layer0_first_=false;\n    fidelity_snapshot_.reset();\n    hits_.stage_mirror.reset();')
    s=once(s,'    if (cs_) cs_->wait();\n    if (sh_cs_) sh_cs_->wait();','    if (cs_) cs_->wait();\n    if (sh_cs_) sh_cs_->wait();\n    for(auto& graph:layer0_exec_) { if(graph)delete graph;graph=nullptr; }\n    layer0_snapshot_.reset();')
    anchor='    if (fidelity_diag::settings().enabled) {'
    insert='''    if (layer0_diag::settings().enabled && lb_==0 && le_>0) {
        if (!native_hc_requested() || g.n_embd!=2560 || g.hc!=4 || g.n_layers!=48 ||
            g.ssm_state_size!=128 || g.ssm_k_heads!=16 || g.ssm_v_heads!=48 || g_qfuse() || !strata::kernels::cpu::expert_layout().native || !one_token_self_commit() || std::getenv("STRATA_VERIFY_EAGER")) {
            err="layer0 diagnostic: unsupported geometry/native/QFUSE/selfcommit/eager contract";return false;
        }
        layer0_snapshot_=std::make_shared<layer0_diag::snapshot>(*cs_,device_);
    }
'''
    s=once(s,anchor,insert+anchor)
    s=once(s,'        bool native_gr_ok=true;','''        const bool l0_capture=layer0_first_ && layer0_snapshot_ && T==1 && l==0 && grp==0;
        if(l0_capture)layer0_snapshot_->copy("residual_input",Rt(0),40960,*cs);
        bool native_gr_ok=true;''')
    s=once(s,'                    HcNativeArgs a; a.eps=EPS;','''                    if(l0_capture && half==1)layer0_snapshot_->copy("residual_after_attn",Rt(0),40960,*cs);
                    HcNativeArgs a; a.eps=EPS;''')
    s=once(s,'                    if (!native_hc_submit(a,cs,err)) { native_gr_ok=false; return false; }','''                    if (!native_hc_submit(a,cs,err)) { native_gr_ok=false; return false; }
                    if(l0_capture && half==0) {
                        layer0_snapshot_->copy("attn_hc_normalized",a.xn,40960,*cs);
                        layer0_snapshot_->copy("attn_hc_low",a.lo,1280,*cs);
                        layer0_snapshot_->copy("attn_hc_gate",a.gate,40960,*cs);
                        layer0_snapshot_->copy("attn_hc_inject",a.inject_out,16,*cs);
                        layer0_snapshot_->copy("attn_mixed",a.mixed,10240,*cs);
                    }
                    if(l0_capture && half==1)layer0_snapshot_->copy("ffn_mixed",a.mixed,10240,*cs);''')
    s=once(s,'                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;\n                float* qkv = qkv_L_',"""                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                if(l0_capture) {
                    layer0_snapshot_->copy("gdn_state_before",state,3145728,*cs);
                    layer0_snapshot_->copy("gdn_conv_before",conv,122880,*cs);
                }
                float* qkv = qkv_L_""")
    s=once(s,'                if (!q8_attn) native_quantize_q8_1(xm, xq_, (int) N, n, cs);\n                native_mmvq(wqkv->native_type', '''                if (!q8_attn) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                if(l0_capture)layer0_snapshot_->copy("attn_input_q81",xq_,2880,*cs);
                native_mmvq(wqkv->native_type''')
    # Capture exactly after all producers, before out-projection/next input overwrites.
    anchor='                if (!g_qfuse()) native_quantize_q8_1(y_ + (size_t) tb * ZV, xq_, (int) ZV, n, cs);   // STRATA_QFUSE: done above'
    after='''
                if(l0_capture) {
                    layer0_snapshot_->copy("gdn_qkv",qkv,40960,*cs);
                    layer0_snapshot_->copy("gdn_z",z_,24576,*cs);
                    layer0_snapshot_->pair("gdn_decay_beta",gate,beta,192,*cs);
                    layer0_snapshot_->copy("gdn_normalized_qkv",hb,40960,*cs);
                    layer0_snapshot_->copy("gdn_output_gated",y_,24576,*cs);
                    layer0_snapshot_->copy("gdn_output_q81",xq_,6912,*cs);
                    layer0_snapshot_->copy("gdn_state_after",state,3145728,*cs);
                    layer0_snapshot_->copy("gdn_conv_after",conv,122880,*cs);
                }'''
    s=once(s,anchor,anchor+after)
    anchor='                native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);'
    s=once(s,anchor,anchor+'\n                if(l0_capture)layer0_snapshot_->copy("gdn_block_output",bo_,10240,*cs);')
    # Router has completed before planning and preserves its raw logits/selected rows.
    anchor='        auto make_plan=[&](uint32_t* skip,uint32_t ring,uint32_t* plan_err) {'
    if s.count(anchor)!=1:raise ValueError('router anchor')
    s=s.replace(anchor,'''        if(l0_capture) {
            layer0_snapshot_->copy("router_logits",logits_,2048,*cs);
            layer0_snapshot_->copy("router_ids",ids_,40,*cs);
            layer0_snapshot_->copy("router_weights",w_,40,*cs);
        }
'''+anchor,1)
    # Input native packet only after qdedup/quantize completed, before any expert reads.
    anchor='        stamp(l, 18, grp);\n        return true;'
    s=once(s,anchor,'        if(l0_capture)layer0_snapshot_->copy("ffn_input_q81",nat_xq_,2880,*cs);\n'+anchor)
    anchor='        stamp(l, 24, grp);\n        if (native_hc_requested()) {'
    s=once(s,anchor,'''        if(layer0_first_ && layer0_snapshot_ && T==1 && l==0 && grp==0)
            layer0_snapshot_->copy("ffn_block_output",bo_,10240,*cs);
'''+anchor)
    anchor='            if (fidelity_first_ && fidelity_snapshot_ && T==1 && grp==0) fidelity_snapshot_->layer(int(l),Rt(0),*cs);'
    s=once(s,anchor,'''            if(layer0_first_ && layer0_snapshot_ && T==1 && l==0 && grp==0)
                {layer0_snapshot_->copy("residual_after_ffn",Rt(0),40960,*cs);layer0_snapshot_->stamp(*cs);}
'''+anchor)
    # Two independent variants for resident/doorbell. Normal graphs remain unchanged.
    old='fidelity_first_ && T==1 ? fidelity_exec_[ar_off_?1:0] : (ar_off_ ? exec_nr_[T] : exec_[T])'
    if s.count(old)!=2:raise ValueError('graph variant anchors differ')
    s=s.replace(old,'layer0_first_ && T==1 ? layer0_exec_[(ar_off_?1:0)+(fidelity_first_?2:0)] : ('+old+')')
    anchor='void Verifier::release_fidelity_observer() {'
    helper='''void Verifier::select_layer0_first(bool on) {
    layer0_first_=on && layer0_snapshot_!=nullptr;
    if(next_)next_->select_layer0_first(on);
}
bool Verifier::dump_layer0_first(int rows,int64_t pos,int token,std::string& err) try {
    if(layer0_snapshot_ && !layer0_snapshot_->dump((ar_off_?1:0)+(fidelity_first_?2:0),rows,pos,token,err))return false;
    return next_==nullptr || next_->dump_layer0_first(rows,pos,token,err);
} catch(const std::exception& e){err=std::string("layer0 diagnostic: ")+e.what();return false;}
void Verifier::release_layer0_observer() {
    const OnDevice on_device(device_);
    if(layer0_snapshot_) {
        cs_->wait_and_throw();
        for(auto& graph:layer0_exec_) { if(graph)delete graph;graph=nullptr; }
        layer0_snapshot_->release();layer0_snapshot_.reset();layer0_first_=false;
    }
    if(next_)next_->release_layer0_observer();
}

'''
    s=once(s,anchor,helper+anchor)
    s=once(s,'void Verifier::release_fidelity_observer() {','void Verifier::release_fidelity_observer() {\n    if(layer0_snapshot_)release_layer0_observer(); // Combined graphs retire before old snapshot storage.')
    s=once(s,'    std::string rerr;\n    const bool ok = record_window(T, cs_, rerr);','    std::string rerr;\n    if(layer0_first_&&layer0_snapshot_&&T==1)layer0_snapshot_->begin_roster((ar_off_?1:0)+(fidelity_first_?2:0));\n    const bool ok = record_window(T, cs_, rerr);')
    s=once(s,'    const dpct::err0 ie = DPCT_CHECK_ERROR(\n        exec_t = new sycl::ext::oneapi::experimental::command_graph<','    if(layer0_first_&&layer0_snapshot_&&T==1)layer0_snapshot_->seal_roster((ar_off_?1:0)+(fidelity_first_?2:0));\n    const dpct::err0 ie = DPCT_CHECK_ERROR(\n        exec_t = new sycl::ext::oneapi::experimental::command_graph<')
    s=once(s,'    if (!staged_) stage_inputs(T, tokens, pos0);\n    staged_ = false;','    if (!staged_) stage_inputs(T, tokens, pos0);\n    staged_ = false;\n    if(layer0_first_&&layer0_snapshot_&&T==1)layer0_snapshot_->arm((ar_off_?1:0)+(fidelity_first_?2:0),T,pos0,tokens[0],*cs_);')
    changed[p]=s
    p=paths[2];s=changed[p]
    s=once(s,'#include "strata/core/fidelity_observer.hpp"','#include "strata/core/fidelity_observer.hpp"\n#include "strata/core/layer0_numerical_observer.hpp"')
    anchor='            strata::core::fidelity_diag::begin(ids,!geni && o.batch<=1 && o.pipeline_windows==0 && o.mtp.empty());'
    s=once(s,anchor,anchor+'\n            strata::core::layer0_diag::begin(ids,!geni && o.batch<=1 && o.pipeline_windows==0 && o.mtp.empty());')
    anchor='            strata::core::fidelity_diag::resumed(resume);'
    s=once(s,anchor,anchor+'\n            strata::core::layer0_diag::resumed(resume);')
    anchor='                ver.select_fidelity_first(produced_n==0 && strata::core::fidelity_diag::current().active && T==1);'
    s=once(s,anchor,anchor+'\n                ver.select_layer0_first(produced_n==0 && strata::core::layer0_diag::current().active && T==1 && strata::core::layer0_diag::current().reused==0 && strata::core::layer0_diag::first_position(strata::core::layer0_diag::current().ids.size(),T,p,window[0],strata::core::layer0_diag::current().ids));')
    anchor='                ver.select_fidelity_first(false);'
    s=once(s,anchor,'''                if(produced_n==0 && !ver.dump_layer0_first(T,p,window[0],err)) { std::printf("ERR %s\\n",err.c_str());return 1; }
                ver.select_layer0_first(false);
'''+anchor)
    anchor='            if (strata::core::fidelity_diag::settings().enabled) {'
    s=once(s,anchor,'            if(strata::core::layer0_diag::settings().enabled)ver.release_layer0_observer();\n'+anchor)
    changed[p]=s
    added={'sycl/include/strata/core/layer0_numerical_contract.hpp':(HERE/'layer0_numerical_contract_v1.hpp').read_text(),
           'sycl/include/strata/core/layer0_numerical_observer.hpp':(HERE/'layer0_numerical_observer_v1.hpp').read_text()}
    return original,changed,added


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError('Preserve existing0021 draft')
    original,changed,added=construct();diff=[]
    for path,text in {**changed,**added}.items():
        diff.extend(difflib.unified_diff(original.get(path,'').splitlines(True),text.splitlines(True),fromfile='a/'+path if path in original else '/dev/null',tofile='b/'+path))
    args.output.write_text(''.join(diff));print(hashlib.sha256(args.output.read_bytes()).hexdigest()+' '+str(args.output))


if __name__=='__main__':main()
