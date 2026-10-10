#!/usr/bin/env python3
"""CPU source reconstruction only. No build, git, Docker, GPU or model reads."""
import difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl/source')
HEADER='sycl/include/strata/program/host_critical_path_trace.hpp'
G='sycl/src/program/generate.cpp'; V='sycl/src/core/verify.cpp'
SHA={G:'27e52a23debf493b57654a7ca871859b2655c5d8bb11c51be9e473d5ba557a39',V:'cdd837fd3dd31873094587e549623bae887765c48d8c4de7f542185f52c4d84f'}
NS='strata::program::host_trace36'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(s,a,b,count=1):
 if s.count(a)!=count:raise ValueError('Exact source anchor count differs: '+a[:100])
 return s.replace(a,b)
def scope(kind):return NS+'::Scope h36_'+kind+'('+NS+'::'+kind+');'
def reconstruct():
 old={p:(SDK/p).read_text() for p in [G,V]}
 for p in old:
  if sha(SDK/p)!=SHA[p]:raise ValueError('Frozen35 source changed')
 new=dict(old)
 for p in [G,V]:new[p]='#include "strata/program/host_critical_path_trace.hpp"\n'+new[p]
 s=new[G]
 s=replace(s,'    if (!have_tokens && o.serve) {', '    if (!'+NS+'::init(o.serve && o.batch==0 && o.pipeline_windows==0 && o.mtp.empty() && o.layer_split.empty() && o.split_device.empty() && o.suffix_draft==0 && o.lookup_chain==0)) { std::fprintf(stderr,"host trace36 requires bounded native serial graph mode and valid DIR/binding\\n"); return 2; }\n    if (!have_tokens && o.serve) {')
 s=replace(s,'            const Clock::time_point r0 = Clock::now();','            const Clock::time_point r0 = Clock::now();\n            '+NS+'::RequestScope h36_request(n,req_fresh,req_rid,req_slot_generation,max_new);\n            '+NS+'::Scope h36_reset('+NS+'::Reset);')
 # End reset/selection coverage explicitly; lifetime scope is replaced with a bounded block below.
 s=s.replace(NS+'::Scope h36_reset('+NS+'::Reset);','const uint64_t h36_reset_id='+NS+'::begin('+NS+'::Reset);')
 s=replace(s,'            conversations.limit_reuse(read_from);','            '+NS+'::end('+NS+'::Reset,h36_reset_id);\n            conversations.limit_reuse(read_from);')
 s=replace(s,'                const bool sp_ok = win ? read_windows(at, to, err) : read_part(at, to, err);','                bool sp_ok;\n                { '+scope('PromptSpan')+' sp_ok = win ? read_windows(at, to, err) : read_part(at, to, err); }')
 s=replace(s,'                    std::printf("T %d%s\\n",(int)outv[(size_t)i],','                    '+NS+'::token((int)outv[(size_t)i],p+i+1);\n                    const uint64_t h36_printf_id='+NS+'::begin('+NS+'::TPrintf);\n                    std::printf("T %d%s\\n",(int)outv[(size_t)i],')
 s=replace(s,'                    if (req_logprobs >= 0) {   // row i', '                    '+NS+'::end('+NS+'::TPrintf,h36_printf_id);\n                    if (req_logprobs >= 0) {   // row i')
 s=replace(s,'                std::fflush(stdout);\n                ++rounds;', '                { '+scope('StdoutFlush')+' std::fflush(stdout); }\n                ++rounds;')
 s=replace(s,'                        if (!ver.copy_logits(i, lrow.data())) {','                        bool h36_copy_ok;\n                        { '+scope('LPReadback')+' h36_copy_ok=ver.copy_logits(i,lrow.data()); }\n                        if (!h36_copy_ok) {')
 s=replace(s,'                            float mx = -INFINITY;','                            '+scope('LPCPU')+'\n                            float mx = -INFINITY;')
 s=replace(s,'            std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld%s%s\\n",','            '+'const uint64_t h36_DoneEmit_id='+NS+'::begin('+NS+'::DoneEmit);\n            std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld%s%s\\n",')
 s=replace(s,'            if (admit_slot >= 0) {   // --batch: BADM','            '+NS+'::end('+NS+'::DoneEmit,h36_DoneEmit_id);\n            '+NS+'::completed((int)produced_n);\n            if (admit_slot >= 0) {   // --batch: BADM')
 new[G]=s
 s=new[V]
 s=replace(s,'    if (exec_t != nullptr) return true;', '    if (exec_t != nullptr) { '+NS+'::graph_mark(reinterpret_cast<uintptr_t>(exec_t),false); return true; }\n    '+scope('Capture'))
 a=s.index('bool Verifier::capture(int T,'); b=s.index('bool Verifier::capture_commit(',a)
 part=replace(s[a:b],'    delete (graph);','    '+NS+'::graph_mark(reinterpret_cast<uintptr_t>(exec_t),true);\n    delete (graph);')
 part=replace(part,'    const dpct::err0 ie = DPCT_CHECK_ERROR(', '    const uint64_t h36_finalize_id='+NS+'::begin('+NS+'::GraphFinalize);\n    const dpct::err0 ie = DPCT_CHECK_ERROR(')
 part=replace(part,'    '+NS+'::graph_mark(reinterpret_cast<uintptr_t>(exec_t),true);','    '+NS+'::end('+NS+'::GraphFinalize,h36_finalize_id);\n    '+NS+'::graph_mark(reinterpret_cast<uintptr_t>(exec_t),true);')
 part=replace(part,'    const dpct::err0 us = DPCT_CHECK_ERROR(cs_->wait());','    const uint64_t h36_capture_wait_id='+NS+'::begin('+NS+'::CaptureWait);\n    const dpct::err0 us = DPCT_CHECK_ERROR(cs_->wait());\n    '+NS+'::end('+NS+'::CaptureWait,h36_capture_wait_id);')
 s=s[:a]+part+s[b:]
 s=replace(s,'bool Verifier::capture_commit(std::string &err) try {','bool Verifier::capture_commit(std::string &err) try {\n    '+NS+'::ContextRestore h36_commit_context;\n    '+scope('CommitCapture'))
 a=s.index('bool Verifier::capture_commit('); b=s.index('bool Verifier::run(int T,',a)
 part=replace(s[a:b],'    if (commit_exec_ != nullptr) return true;','    if (commit_exec_ != nullptr) { '+NS+'::graph_mark(reinterpret_cast<uintptr_t>(commit_exec_),false); return true; }')
 part=replace(part,'    delete (graph);','    '+NS+'::graph_mark(reinterpret_cast<uintptr_t>(commit_exec_),true);\n    delete (graph);')
 s=s[:a]+part+s[b:]

 s=replace(s,'    prefix_lifecycle::stage("verify_body",device_,lb_,le_,pos0,pos0+T,false);','    '+NS+'::Frame h36_frame(device_,lb_,le_,T,pos0,reinterpret_cast<uintptr_t>(cs_));\n    '+scope('Verify')+'\n    prefix_lifecycle::stage("verify_body",device_,lb_,le_,pos0,pos0+T,false);')
 s=replace(s,'    if (!staged_) stage_inputs(T, tokens, pos0);','    if (!staged_) { '+scope('StageInputs')+' stage_inputs(T,tokens,pos0); }')
 s=replace(s,'    if (do_ple) {\n        int32_t prev[2]', '    if (do_ple) {\n        '+scope('PlePrefetch')+'\n        int32_t prev[2]')
 s=replace(s,'    if (ple_precollected) {\n        const Clock::time_point tp', '    if (ple_precollected) {\n        '+scope('PleGather')+'\n        const Clock::time_point tp')
 # Delimit submit/wait around the existing statements, without retaining/querying events or adding a wait.
 s=replace(s,'    ple_input33::submit_begin(ple_input_publication);','    ple_input33::submit_begin(ple_input_publication);\n    '+'const uint64_t h36_ReplaySubmit_id='+NS+'::begin('+NS+'::ReplaySubmit);')
 s=replace(s,'    ple_input33::submit_returned(ple_input_publication,int(le));','    '+NS+'::end('+NS+'::ReplaySubmit,h36_ReplaySubmit_id);\n    ple_input33::submit_returned(ple_input_publication,int(le));')
 s=replace(s,'    trace_ev("SYNC", -1, -1, 0);','    trace_ev("SYNC", -1, -1, 0);\n    '+'const uint64_t h36_ReplayWait_id='+NS+'::begin('+NS+'::ReplayWait);')
 s=replace(s,'    trace_ev("SYNCED", -1, -1, (int64_t) se);','    '+NS+'::end('+NS+'::ReplayWait,h36_ReplayWait_id);\n    trace_ev("SYNCED", -1, -1, (int64_t) se);')
 a=s.index('bool Verifier::run(int T,'); b=s.index('bool Verifier::warm(',a)
 part=replace(s[a:b],'    if (head_sampling_ && (sampled || hist_d_ != nullptr)) {','    if (head_sampling_ && (sampled || hist_d_ != nullptr)) {\n        '+scope('Sample'))
 s=s[:a]+part+s[b:]
 s=replace(s,'bool Verifier::commit(int n_keep, std::string &err) try {','bool Verifier::commit(int n_keep, std::string &err) try {\n    '+NS+'::Frame h36_frame(device_,lb_,le_,last_t_,last_pos0_,reinterpret_cast<uintptr_t>(cs_));\n    '+scope('Commit'))
 s=replace(s,'bool Verifier::wait_commit(std::string &err) try {','bool Verifier::wait_commit(std::string &err) try {\n    '+NS+'::Frame h36_frame(device_,lb_,le_,last_t_,last_pos0_,reinterpret_cast<uintptr_t>(cs_));\n    '+scope('CommitWait'))
 s=replace(s,'    } else {\n        std::atomic_thread_fence(std::memory_order_seq_cst);\n        const dpct::err0 le =', '    } else {\n        '+NS+'::graph_mark(reinterpret_cast<uintptr_t>(commit_exec_),false);\n        std::atomic_thread_fence(std::memory_order_seq_cst);\n        const dpct::err0 le =')

 new[V]=s
 new[HEADER]=(ROOT/'strata/flash-next/host_critical_path36_header_v2.hpp').read_text()
 return old,new

def patch_bytes():
 old,new=reconstruct();out=[]
 for name in [G,V,HEADER]:
  out.extend(difflib.unified_diff(old.get(name,'').splitlines(True),new[name].splitlines(True),fromfile='a/'+name if name in old else '/dev/null',tofile='b/'+name))
 return ''.join(out).encode()
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Output must be new')
 a.output.write_bytes(patch_bytes());print(json.dumps({'actual_build':False,'patch_sha256':sha(a.output)}))
