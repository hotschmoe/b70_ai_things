"""NEW default-OFF layer3 targets over frozen source39; source preparation only."""
import difflib, hashlib, json
from pathlib import Path
import prepare_full_cache_memory39_v1 as previous
ROOT=previous.ROOT;HERE=previous.HERE
BASE=HERE/'full-cache-memory39-engine-build-plan-v1.json'
BASE_SHA='86bc189bbb2805ffcb989974146fe8263a00beeb109cd44ef41af77eb1db6940'
HEADER=HERE/'layer3_qsa_target_observer_v1.hpp'
H='sycl/include/strata/core/layer3_qsa_target_observer.hpp'
PATCH=HERE/'patches/0040-default-off-native-layer3-qsa-target-observer.patch'
sha=previous.sha;once=previous.once

def reconstruct():
 if sha(BASE)!=BASE_SHA:raise ValueError('Frozen source39 engine plan changed')
 _,old=previous.reconstruct();plan=json.loads(BASE.read_bytes())
 for path,text in old.items():
  if hashlib.sha256(text.encode()).hexdigest()!=plan['expected_patched_source_sha256'][path]:raise ValueError('Source39 reconstruction changed '+path)
 new=dict(old);new[H]=HEADER.read_text()
 path='sycl/include/strata/core/verify.hpp';s=new[path]
 s=once(s,'#include "strata/core/prefix_residual_capture.hpp"','#include "strata/core/prefix_residual_capture.hpp"\n#include "strata/core/layer3_qsa_target_observer.hpp"')
 s=once(s,'    std::shared_ptr<prefix_residual30::snapshot> prefix30_snapshot_;','    std::shared_ptr<prefix_residual30::snapshot> prefix30_snapshot_;\n    std::shared_ptr<qsa3_target::snapshot> qsa3_snapshot_;');new[path]=s
 path='sycl/src/core/verify.cpp';s=new[path]
 s=once(s,'    prefix30_snapshot_.reset(); // Only retired serialfidelity/combined graphs reference P30.','    prefix30_snapshot_.reset(); // Only retired serialfidelity/combined graphs reference P30.\n    qsa3_snapshot_.reset(); // Only the retired P30/fidelity graphs can reference these targets.')
 s=once(s,'    prefix30_snapshot_.reset(); // All serial graphs retired; owning queue still alive.','    prefix30_snapshot_.reset(); // All serial graphs retired; owning queue still alive.\n    qsa3_snapshot_.reset();')
 anchor='    if (std::getenv("STRATA_VERIFY_DEBUG") != nullptr) {   // SYCL port: the per-layer residual ladder (token 0)'
 s=once(s,anchor,'''    if(qsa3_target::settings().enabled) {
        if(!native_hc_requested()||!native_qsa_enabled()||!native_rope_enabled()||!native_qsa_indexer_enabled()||g.n_embd!=2560||g.hc!=4||g.n_layers!=48||max_t!=2||g_qfuse()||std::getenv("STRATA_VERIFY_EAGER")||std::getenv("STRATA_ROPE_TABLE")||std::getenv("STRATA_NO_NORM_ROPE")||std::getenv("STRATA_ATTN_LANECELL")||layer0_diag::settings().enabled){err="QSA3 unsupported source geometry/native/flags";return false;}
        qsa3_snapshot_=std::make_shared<qsa3_target::snapshot>(*cs_,device_,int(lb_),int(le_));
    }
'''+anchor)
 anchor='        const int tb = tb_[grp], te = te_[grp], n = te - tb;\n        if(prefix30_prompt_ && prefix30_snapshot_)prefix30_snapshot_->copy_rows("input",int(l),Rt(tb),n,tb,T,*cs);'
 s=once(s,anchor,anchor+'\n        const bool q3_capture=qsa3_snapshot_&&(prefix30_prompt_||fidelity_first_)&&l==3;\n        if(q3_capture&&(grp!=0||n!=T||batch_rec_))throw std::invalid_argument("QSA3 unsupported split/batch source ownership");')
 anchor='                    if(l0_capture && half==1)layer0_snapshot_->copy("ffn_mixed",a.mixed,10240,*cs);'
 s=once(s,anchor,anchor+'\n                    if(q3_capture&&half==0){qsa3_snapshot_->copy("hc_residual",a.R,t,1,T,*cs);qsa3_snapshot_->copy("hc_mixed",a.mixed,t,1,T,*cs);}')
 anchor='                const bool fuse_nr = native_qsa_enabled() && native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);'
 s=once(s,anchor,anchor+'''
                if(q3_capture){if(!fuse_nr||!dec_batch||st.kv_rot||st.kv_int8||st.kv_q4||st.kv_hybrid)throw std::invalid_argument("QSA3 unsupported actual fused/KV source route");
                    qsa3_snapshot_->copy("q_gamma",wqn->data,0,1,T,*cs);qsa3_snapshot_->copy("k_gamma",wkn->data,0,1,T,*cs);qsa3_snapshot_->copy("indexer_q_gamma",wiqn->data,0,1,T,*cs);qsa3_snapshot_->copy("indexer_k_gamma",wikn->data,0,1,T,*cs);
                    qsa3_snapshot_->state("indexer_before",st,*cs);qsa3_snapshot_->copy("page_before",st.page_table,0,1,T,*cs);qsa3_snapshot_->copy("K_physical_page0_before",st.k_pool,0,1,T,*cs);qsa3_snapshot_->copy("V_physical_page0_before",st.v_pool,0,1,T,*cs);
                }''')
 # The same stamp occurs in GDN and QSA. Scope replacements to QSA branch only.
 begin=s.index('                // ======================= QSA =======================')
 end=s.index('        } catch (const std::exception& e) {',begin);block=s[begin:end]
 anchor='                if (!q8_attn) native_quantize_q8_1(xm, xq_, (int) N, n, cs);'
 block=once(block,anchor,anchor+'\n                if(q3_capture)qsa3_snapshot_->copy("mixed_q81",xq_,tb,n,T,*cs);')
 anchor='                stamp(l, 7, grp);'
 block=once(block,anchor,'                if(q3_capture)qsa3_snapshot_->copy("indexer_key_projected",idx_raw+tb*ID,tb,n,T,*cs);\n'+anchor)
 anchor='                if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);'
 block=once(block,anchor,'                if(q3_capture){qsa3_snapshot_->copy("k_projected",kcur_+tb*NKV*HD,tb,n,T,*cs);qsa3_snapshot_->copy("v_projected",vcur_+tb*NKV*HD,tb,n,T,*cs);}\n'+anchor)
 anchor='                stamp(l, 8, grp);';block=once(block,anchor,'                if(q3_capture)qsa3_snapshot_->copy("k_fused_RoPE",kcur_+tb*NKV*HD,tb,n,T,*cs);\n'+anchor)
 anchor='                stamp(l, 9, grp);';block=once(block,anchor,'                if(q3_capture)qsa3_snapshot_->state("indexer_after",st,*cs);\n'+anchor)
 anchor='                if (qb) {\n                    if (fuse_nr) {';block=once(block,anchor,'                if(q3_capture)qsa3_snapshot_->copy("q_full_projected",qfull_+tb*NH*2*HD,tb,n,T,*cs);\n'+anchor)
 anchor='                    norm_rope(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ);'
 block=once(block,anchor,'                    if(q3_capture)qsa3_snapshot_->copy("indexer_query_projected",qidx_+tb*IQ*ID,tb,n,T,*cs);\n'+anchor)
 anchor='                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);'
 block=once(block,anchor,'                    if(q3_capture)qsa3_snapshot_->copy("indexer_query_projected",qx,t,1,T,*cs);\n'+anchor)
 anchor='                stamp(l, 10, grp);';block=once(block,anchor,'                if(q3_capture){qsa3_snapshot_->copy("q_fused_RoPE",qcur_+tb*NH*HD,tb,n,T,*cs);qsa3_snapshot_->copy("indexer_query_fused_RoPE",qidx_+tb*IQ*ID,tb,n,T,*cs);}\n'+anchor)
 anchor='                stamp(l, 12, grp);';block=once(block,anchor,'                if(q3_capture){for(int t=tb;t<te;++t){qsa3_snapshot_->copy("step",step_+t*kStepCount,t,1,T,*cs);qsa3_snapshot_->copy("selected_ids_capacity4",sel_+(size_t)t*cap_,t,1,T,*cs);}qsa3_snapshot_->copy("page_resolved",st.page_table,0,1,T,*cs);qsa3_snapshot_->copy("K_physical_page0_after_resolve",st.k_pool,0,1,T,*cs);qsa3_snapshot_->copy("V_physical_page0_after_resolve",st.v_pool,0,1,T,*cs);}\n'+anchor)
 anchor='                stamp(l, 13, grp);';block=once(block,anchor,'                if(q3_capture)qsa3_snapshot_->copy("attention",attn_+tb*NH*HD,tb,n,T,*cs);\n'+anchor)
 anchor='                stamp(l, 14, grp);';block=once(block,anchor,'                if(q3_capture)qsa3_snapshot_->copy("gated",attn32_+tb*NH*HD,tb,n,T,*cs);\n'+anchor)
 anchor='                native_mmvq(wo->native_type, wo->native_data, xq_, bo_ + tb * N, (int) (NH * HD), (int) N, n, cs);'
 block=once(block,anchor,'                if(q3_capture)qsa3_snapshot_->copy("gated_q81",xq_,tb,n,T,*cs);\n'+anchor+'\n                if(q3_capture)qsa3_snapshot_->copy("output_projected",bo_+tb*N,tb,n,T,*cs);')
 s=s[:begin]+block+s[end:]
 anchor='    if(prefix30_prompt_ && prefix30_snapshot_)prefix30_snapshot_->stamp_rows(T,*cs);'
 s=once(s,anchor,anchor+'\n    if(qsa3_snapshot_&&(prefix30_prompt_||fidelity_first_))qsa3_snapshot_->record_end(T,*cs);')
 anchor='    if(prefix30_prompt_ && prefix30_snapshot_)prefix30_snapshot_->begin_rows(T);'
 s=once(s,anchor,anchor+'\n    if(qsa3_snapshot_&&(prefix30_prompt_||fidelity_first_))qsa3_snapshot_->record_begin(T);')
 anchor='    if(prefix30_snapshot_){prefix30_snapshot_->release();prefix30_snapshot_.reset();}'
 s=once(s,anchor,anchor+'\n    qsa3_snapshot_.reset(); // P30/fidelity graphs retired before target memory.')
 anchor='    if(!prefix30_snapshot_->begin("prompt_verifier",pos,rows,err))return false;'
 s=once(s,anchor,anchor+'\n    if(qsa3_snapshot_)qsa3_snapshot_->begin("prompt_verifier",pos,rows);')
 anchor='    if(!prefix30_snapshot_->dump(err))return false;';s=once(s,anchor,anchor+'\n    if(qsa3_snapshot_)qsa3_snapshot_->dump();')
 anchor='    if(next_)next_->select_fidelity_first(on);'
 s=once(s,anchor,'    if(fidelity_first_&&qsa3_snapshot_)qsa3_snapshot_->begin("verifier",int64_t(fidelity_diag::current().ids.size())-1,1);\n'+anchor)
 anchor='    if(prefix30_snapshot_ && !prefix30_snapshot_->dump(err))return false;'
 s=once(s,anchor,anchor+'\n    if(qsa3_snapshot_)qsa3_snapshot_->dump();');new[path]=s
 return old,new

def patch_bytes():
 old,new=reconstruct()
 return ''.join(''.join(difflib.unified_diff(old.get(n,'').splitlines(True),new[n].splitlines(True),fromfile='a/'+n if n in old else '/dev/null',tofile='b/'+n))for n in sorted(new)if old.get(n)!=new[n]).encode('ascii')

def build_plan():
 _,new=reconstruct();p=json.loads(BASE.read_bytes())
 if PATCH.read_bytes()!=patch_bytes():raise ValueError('New QSA target patch differs')
 p['derived_from_plan']={'path':str(BASE.relative_to(ROOT)),'sha256':BASE_SHA}
 p['generation']='source39 plus defaultOFF actual layer3 QSA comparison targets40'
 p['status']='SOURCE/CPU proposal; fresh build/runtime unobserved'
 p['patches'].append({'path':str(PATCH.relative_to(ROOT)),'sha256':sha(PATCH)})
 p['added_header_payloads'].append({'path':H,'sha256':sha(HEADER),'consumed_include':'strata/core/layer3_qsa_target_observer.hpp','resolution':'NEW bounded SYCL native target observer'})
 p['expected_patched_source_sha256']={n:hashlib.sha256(text.encode()).hexdigest()for n,text in new.items()}
 p['overlay_files']=sorted(new);p['new_source_contracts']['0040']={'default_OFF':True,'layer':3,'prefix4_2_1_1_only':True,'captured_values_are_math_inputs':False,'internal_norm_score_softmax_values_observed':False,'source39_runtime_transfer':False}
 p['required_before_model_run'].append('Fresh source40 SDK/source/header/ABI/ELF/oracle/upload/strong one+pair baseline and genuine ON-OFF head+all48 equivalence/graph nonce/field ownership/normal teardown/health/newfour/pages required; no old39 runtime proof transfer.')
 return p

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 if args.output.exists():raise ValueError('New source-plan output required')
 args.output.write_text(json.dumps(build_plan(),indent=2)+'\n')
 print(json.dumps({'plan_sha256':sha(args.output),'actual_compile_or_GPU_run':False}))
