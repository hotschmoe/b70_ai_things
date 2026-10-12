"""Pure default-OFF original native serve integration over pinned actual58 source."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SDK=Path('/mnt/vm_8tb/b70/build/strata-native58-fresh52-v2-20261012T204001Z-id823bdv')
PLAN=HERE/'original-mtp53-serve-source-plan-v3.json'
def once(s,a,b):
 if s.count(a)!=1:raise ValueError('Unique actual58 serve seam required '+a[:100])
 return s.replace(a,b,1)
def construct(source):
 s=source
 s='#include "strata/kernels/original_mtp53_serving_context.hpp"\n'+s
 anchor='    const char* batch_mtp_env = std::getenv("STRATA_BATCH_MTP");'
 s=once(s,anchor,'''    const char* original_mode_env=std::getenv("STRATA_ORIGINAL_MTP53_SERVING");
    const bool original_serving=original_mode_env && std::strcmp(original_mode_env,"1")==0;
    if(original_mode_env && std::strcmp(original_mode_env,"0") && std::strcmp(original_mode_env,"1")){std::fprintf(stderr,"original serving must0/1\\n");return 2;}
    if(original_serving && (!strata::core::original_mtp53::enabled() || !std::getenv("STRATA_ONE_TOKEN_COMMIT") || std::strcmp(std::getenv("STRATA_ONE_TOKEN_COMMIT"),"0") || !native_pack || !strata::core::native_hc_requested() || g.n_layers!=48 || g.n_embd!=2560 || g.hc!=4 || !o.serve || o.batch<2 || o.batch>6 || o.batch_groups!=1 || stages.size()!=1 || !last_st || last_st->dev!=1 || last_st->lb!=24 || last_st->le!=48 || o.spec!=2 || !o.mtp.empty() || o.batch_mtp || o.pipeline_windows!=0 || o.max_context!=4096 || o.adapt_every!=0 || !o.no_prefill_borrow || o.lookup_chain!=0 || o.suffix_draft!=0 || o.short_read!=0 || o.kv_grow || std::getenv("STRATA_VERIFY_MIRROR_STAGE_T1") && std::strcmp(std::getenv("STRATA_VERIFY_MIRROR_STAGE_T1"),"0") || !o.stop_eos || o.eos_ids!=std::vector<int64_t>{248044,248046})){std::fprintf(stderr,"original serving requires exact native paired24/24 greedy/spec2/private slots/no-stock-MTP scope\\n");return 2;}
'''+anchor)
 s=once(s,'    strata::core::MtpDrafter mtp;','''    std::unique_ptr<strata::kernels::original_mtp53::OriginalServeContext53> originalServe;
    uint64_t original_solo_generation=0;
    if(original_serving){const strata::core::OnDevice on(1);const char* path=std::getenv("STRATA_ORIGINAL_MTP53_MAIN_GGUF");const char* d=std::getenv("STRATA_ORIGINAL_MTP53_DRAFT_CONTRACT_SHA256");const char* t=std::getenv("STRATA_ORIGINAL_MTP53_TARGET_CONTRACT_SHA256");struct stat st{};if(!path || !d || !t || ::stat(path,&st)){std::fprintf(stderr,"original serving actual original file/contracts required\\n");return 2;}strata::kernels::original_mtp53::OriginalDraftIdentity53 identity{path,"8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231",uint64_t(st.st_dev),uint64_t(st.st_ino),uint64_t(st.st_size),int64_t(st.st_mtim.tv_sec)*1000000000+st.st_mtim.tv_nsec,int64_t(st.st_ctim.tv_sec)*1000000000+st.st_ctim.tv_nsec};originalServe.reset(new strata::kernels::original_mtp53::OriginalServeContext53(path,identity,dpct::get_in_order_queue(),o.batch,uint64_t(o.conversation_cache_mib)<<20,uint64_t(o.conversation_cache_min_free_mib)<<20,d,t));}
    strata::core::MtpDrafter mtp;''')
 # Existing stock loader guards/body remain untouched. Only target window capacity changes in new mode.
 s=s.replace('if (batch_mtp) ver.set_batch_graph_limit(64);','if (batch_mtp || original_serving) ver.set_batch_graph_limit(64);',1)
 s=s.replace('batch_mtp ? strata::kernels::kVerifyMaxT : std::max(o.spec, o.batch)','(batch_mtp || original_serving) ? strata::kernels::kVerifyMaxT : std::max(o.spec, o.batch)')
 anchor='        mem_mark("the verifier and the drafter\'s binding");'
 if s.count(anchor)!=2:raise ValueError('Actual serving/nonserve verifier markers required')
 s=s.replace(anchor,'        if(originalServe)originalServe->bind_target(ver,last_st->ver);\n'+anchor,1)
 anchor='            ckpt_why.clear();\n            if (o.prompt_cache <= 0 || L < 1) return true;'
 s=once(s,anchor,'''            ckpt_why.clear();
            if(originalServe){if(L>0 && L<int64_t(cur.size())){auto prefix=strata::kernels::original_mtp53::original_tokens32_53(std::vector<int64_t>(cur.begin(),cur.begin()+L));originalServe->publish_admission(ss,last_st->ss,g,prefix);}return true;}
            if (o.prompt_cache <= 0 || L < 1) return true;''')
 s=once(s,'        sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {','''        sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
            if(originalServe)originalServe->teacher(R_rows,int(T),p0,cur);''')
 # This original callback is moved to LAST sp at7832, preserving actual residual ownership.
 s=once(s,'            if(strict_batch_identity && admit_slot>=0)req_slot_generation=++slot_generations[(size_t)admit_slot];','''            if(strict_batch_identity && admit_slot>=0)req_slot_generation=++slot_generations[(size_t)admit_slot];
            if(originalServe){if(admit_slot<0)req_slot_generation=++original_solo_generation;originalServe->begin_admission({req_rid,req_slot_generation});}''')
 # Disable ONLY legacy cache lookup/parking under new original mode; stock path unchanged.
 s=s.replace('if (cache_prefix_ceiling>0 &&','if (!originalServe && cache_prefix_ceiling>0 &&')
 s=once(s,'            const auto parked=cache_prefix_ceiling>0?conversations.best','            const auto parked=!originalServe && cache_prefix_ceiling>0?conversations.best')
 s=once(s,'            // KV streaming: the drafter\'s ring may hold cells past `resume`','''            if(originalServe && !req_fresh){auto joint=originalServe->best(ids,std::min<int64_t>(cache_prefix_ceiling,n-1));if(joint){originalServe->restore_targets(joint,ss,last_st->ss,g);resume=joint->original().target_inputs.size();originalServe->restore_admission(joint,joint->original().target_inputs,ids[size_t(resume)]);std::fprintf(stderr,"MTP53_SERVE_REUSED rid=%llu gen=%llu tokens=%lld shifted_repair=1\\n",(unsigned long long)req_rid,(unsigned long long)req_slot_generation,(long long)resume);}}
            // KV streaming: the drafter's ring may hold cells past `resume`''')
 s=once(s,'                prefix_diag.evaluated(p,p+T,"target_verify");','''                if(originalServe){if(T!=1 || a!=0 || !ver.wait_commit(err))throw std::runtime_error("original solo admission exactT1 commit required");originalServe->admission_committed(p,outv[0],ver.final_R_all());}
                prefix_diag.evaluated(p,p+T,"target_verify");''')
 s=once(s,'            strata::core::prefix_lifecycle::state("committed_live",live,','''            if(originalServe && !cancelled && !live.empty())originalServe->publish_admission(ss,last_st->ss,g,live);
            strata::core::prefix_lifecycle::state("committed_live",live,''')
 s=once(s,'                    if (batch_mtp) {\n                        if (!slot_mtp[(size_t) admit_slot]','''                    if(originalServe){originalServe->move_admission_to_slot(admit_slot,{sl.request_id,sl.generation},sl.ids,sl.x);sl.draft[0]=originalServe->slots().proposal(admit_slot,{sl.request_id,sl.generation});sl.draft_ready=true;}
                    if (batch_mtp) {
                        if (!slot_mtp[(size_t) admit_slot]''')
 s=once(s,'        auto park_slot=[&](int b,std::string& e)->bool{','        auto park_slot=[&](int b,std::string& e)->bool{\n            if(originalServe)return true; // complete compound already captured at its actual terminal boundary')
 s=once(s,'                            sl.partial = true;','''                            if(originalServe){originalServe->move_admission_to_slot(ys,{sl.request_id,sl.generation},sl.ids,ids[size_t(q)]);originalServe->slots().finish(ys,{sl.request_id,sl.generation},sl.ids.size());originalServe->publish_slot(ys,{sl.request_id,sl.generation},*bslot_ss[0][size_t(ys)],*bslot_ss[1][size_t(ys)],g,sl.ids);}
                            sl.partial = true;''')
 # Explicit new compound protocol; old file commands stay unchanged when mode OFF.
 anchor='            if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {'
 s=once(s,anchor,'''            if(originalServe && (line.rfind("ORIGINAL_SAVE ",0)==0 || line.rfind("ORIGINAL_LOAD ",0)==0)){
                try{strata::core::SessionFileIdentity identity;if(!session_identity(identity,err,[]{}))throw std::runtime_error(err);std::array<strata::core::SessionFileIdentity,2> identities{identity,identity};
                    if(line.rfind("ORIGINAL_SAVE ",0)==0){originalServe->save_file(line.substr(14),identities);std::printf("ORIGINAL_SAVED compound=1\\n");}
                    else{std::istringstream input(line.substr(14));std::string digest,path;input>>digest;std::getline(input,path);if(!path.empty() && path.front()==' ')path.erase(0,1);std::array<strata::core::SessionReadLimits,2> limits;const strata::core::QsaState no_draft{};for(int i=0;i<2;++i){const auto& state=i==0?ss:last_st->ss;if(!strata::core::conversation_session_read_limits(limits[i],state,g,no_draft,4096,64,err))throw std::runtime_error(err);limits[i].max_file_bytes=strata::core::session_read_max_file_bytes(limits[i]);limits[i].admit=[&](uint64_t need,std::string& why){if(!strata::core::conversation_memory_admit(strata::core::conversation_available_memory(),need,uint64_t(o.conversation_cache_min_free_mib)<<20)){why="original compound load RAM refused";return false;}return true;};}originalServe->load_file(path,digest,identities,limits);std::printf("ORIGINAL_LOADED compound=1 activation=0\\n");}
                }catch(const std::exception& e){std::printf("ERR original compound file: %s\\n",e.what());}std::fflush(stdout);continue;
            }
            if(originalServe && (line.rfind("SAVE ",0)==0 || line.rfind("RESTORE ",0)==0)){std::printf("ERR original mode requires complete ORIGINAL_SAVE/ORIGINAL_LOAD schema\\n");std::fflush(stdout);continue;}
'''+anchor)
 # Packed batch body only: original model uses per-slot own proposal and actual tail width.
 begin=s.index('        auto batch_step =');end=s.index('            std::fflush(stdout);',begin);part=s[begin:end]
 part=part.replace('(batch_mtp ? 2 : 1)','(originalServe ? 1 : batch_mtp ? 2 : 1)')
 part=part.replace('                if (bs[(size_t) b].active) {','                if (bs[(size_t) b].active) {\n                    const int width53=originalServe && bs[size_t(b)].max_new-bs[size_t(b)].produced>=2 && bs[size_t(b)].p+2<=o.max_context?2:1;if(originalServe && S+width53>strata::kernels::kVerifyMaxT)continue;',1)
 part=part.replace('if (batch_mtp) {','if (batch_mtp || (originalServe && bs[(size_t)b].max_new-bs[(size_t)b].produced>=2 && bs[(size_t)b].p+2<=o.max_context)) {',1)
 part=part.replace('if (batch_mtp && A > 0)','if ((batch_mtp || originalServe) && A > 0)',1)
 anchor='            if (!(batch_mtp ? ver.commit_slot_prefixes(keep.data(), err) : ver.commit_slots(err))) {'
 replacement='''            if(originalServe){std::vector<strata::kernels::original_mtp53::OriginalSlotCompletedRows53> completed;for(int a=0;a<A;++a){const int b=active[a],i=first[a],width=(a+1<A?first[a+1]:S)-i;const auto& sl=bs[size_t(b)];completed.push_back({b,i,width,{sl.request_id,sl.generation},sl.p,width==2?tok[i+1]:-1,{outb[i],width==2?outb[i+1]:-1},int(sl.max_new-sl.produced),outb[i]==248044 || outb[i]==248046,sl.stop});}originalServe->slots().commit_and_catchup(completed,keep,err);}
            if (!originalServe && !(batch_mtp ? ver.commit_slot_prefixes(keep.data(), err) : ver.commit_slots(err))) {'''
 if part.count(anchor)!=1:raise ValueError('Actual packed commit seam required')
 part=part.replace(anchor,replacement,1)
 anchor='                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();'
 replacement='''                        if(originalServe){originalServe->slots().finish(b,{sl.request_id,sl.generation},sl.ids.size());originalServe->publish_slot(b,{sl.request_id,sl.generation},*bslot_ss[0][size_t(b)],*bslot_ss[1][size_t(b)],g,sl.ids);}
'''+anchor
 if part.count(anchor)!=1:raise ValueError('Actual terminal publication seam required')
 part=part.replace(anchor,replacement,1)
 # Refresh next original proposal only after actual catchup; terminal path never proposes.
 part=part.replace('                if (batch_mtp && sl.active) {','''                if(originalServe && sl.active){sl.draft[0]=originalServe->slots().proposal(b,{sl.request_id,sl.generation});sl.draft_ready=true;}
                if (batch_mtp && sl.active) {''',1)
 s=s[:begin]+part+s[end:]
 s=once(s,'            strata::core::full_cache_memory39::lifetime("BSTOP_applied"','''            if(originalServe && bs[(size_t)b].active)originalServe->slots().request_cancel(b,{bs[(size_t)b].request_id,bs[(size_t)b].generation});
            strata::core::full_cache_memory39::lifetime("BSTOP_applied"''')
 # Original owner retires before target mirror/source teardown and SDK checkedExit.
 anchor='        if (strata::program::host_trace36::enabled())\n            strata::program::host_trace36::exit_export();'
 s=once(s,anchor,'        if(originalServe)originalServe.reset();\n'+anchor)
 return s

def prepare(out):
 plan=json.loads(PLAN.read_bytes());root=HERE.parents[1]
 for name,want in plan['files'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=want:raise ValueError('Serve candidate source changed '+name)
 src=SDK/'source/sycl/src/program/generate.cpp';raw=src.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=plan['actual58_generate_sha256']:raise ValueError('Actual58 source changed')
 out=Path(out)
 if not out.is_absolute()or out.resolve()!=out or out.exists():raise ValueError('New canonical source input directory required')
 out.mkdir();p=out/'sycl/src/program/generate.cpp';p.parent.mkdir(parents=True);p.write_text(construct(raw.decode('ascii')),encoding='ascii')
 for dest,source in plan['resolved_overlay_sources'].items():
  target=out/dest;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/source).read_bytes())
 record={'source_plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'constructed_generate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_only':True,'compiled':False,'GPU_executed':False,'API_qualified':False};(out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.output),ensure_ascii=True))
