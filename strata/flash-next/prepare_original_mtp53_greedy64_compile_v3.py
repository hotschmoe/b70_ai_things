"""Prepare a standalone SDK52 native MAIN diagnostic, without executing it."""
import argparse,hashlib,json
from pathlib import Path
import prepare_original_mtp53_step_compile_v2 as component
HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-publisher-mtp53-greedy64-proposal-v3.json'
def transform(source):
 early='    strata::core::MtpDrafter mtp;'
 late='        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);'
 if source.count(early)!=1 or source.count(late)!=2:raise ValueError('Exact SDK52 MAIN anchors required')
 reserve='''    std::unique_ptr<strata::kernels::original_mtp53::LoadedMainDraft53> original53;
    const char* original53_path=std::getenv("STRATA_ORIGINAL_MTP53_MAIN_GGUF");
    const char* original53_mode=std::getenv("STRATA_ORIGINAL_MTP53_GREEDY");
    if(original53_mode){
        if((std::strcmp(original53_mode,"0")!=0 && std::strcmp(original53_mode,"1")!=0) || o.tokens.empty() || o.max_new!=64 || o.tokens.size()+64>256 || !o.stop_eos || o.eos_ids!=std::vector<int64_t>{248044,248046}){std::fprintf(stderr,"MTP53 greedy64 exact mode/prompt/budget/EOS required\\n");return 2;}
        const char* mirror=std::getenv("STRATA_VERIFY_MIRROR_STAGE_T1");
        if(!strata::core::original_mtp53::enabled() || !o.serve || !native_pack || o.spec!=2 || !o.mtp.empty() || o.batch!=0 || o.pipeline_windows!=0 || !stages.empty() || (mirror && std::strcmp(mirror,"1")==0)){
            std::fprintf(stderr,"MTP53 MAIN diagnostic requires explicit original draft, native serve/spec2, one stage, mirror0, no stock MTP/batch/pipeline\\n");return 2;
        }
        if(std::strcmp(original53_mode,"1")==0){
            if(!original53_path || !*original53_path){std::fprintf(stderr,"MTP53 original asset path required\\n");return 2;}
            original53.reset(new strata::kernels::original_mtp53::LoadedMainDraft53(original53_path,dpct::get_in_order_queue()));
        }
        std::fprintf(stderr,"MTP53 GREEDY mode=%s original_tensor_reserve_bytes=%llu plus own scratch/cache; no capacity waiver\\n",original53_mode,original53?2775258112ull:0ull);
    }
'''
 hook='''
        if(original53_mode){
            if(!stage_mirrors){std::fprintf(stderr,"MTP53 GREEDY requires stage mirrors for checked owned shutdown\\n");std::fflush(nullptr);std::_Exit(86);}
            drive.d.plan=ver.plan_sink();
            if(!strata::kernels::original_mtp53::original_greedy53(original53.get(),ver,o.tokens,int(o.max_new),win_pool_fn,win_pool_user) || drive.d.failed){
                // Failure is latched; avoid destructing target queues/state that may still be active.
                std::fflush(nullptr);std::_Exit(86);
            }
            try {
                dpct::get_current_device().queues_wait_and_throw();
                original53.reset(); // drain-protected draft owner retires before any target/context teardown
                if(strata::core::batch_fidelity::settings().enabled){ver.release_batch_observer();for(auto& stage:stages)stage->ver.release_batch_observer();}
                if(strata::core::layer0_diag::settings().enabled)ver.release_layer0_observer();
                if (strata::core::fidelity_diag::settings().enabled) {
                    ver.release_fidelity_observer();sp.release_fidelity_observer();
                    for(auto& stage:stages) {
                        const strata::core::OnDevice on(stage->dev);
                        dpct::get_current_device().queues_wait_and_throw();
                        stage->sp.release_fidelity_observer();
                    }
                }
                if (stage_mirrors) {
                    // Drain every stage context while it is live, then destroy graph
                    // references before dropping tables and their host allocations.
                    // Retire ordinary token/session graphs too, before source aliases
                    // or any mirror allocation can be dropped.
                    strata::core::token_graph_free(tgraph);
                    strata::core::session_graphs_free(gr);
                    std::vector<std::shared_ptr<strata::core::StageExpertMirror>> retiring;
                    if (first_stage_mirror) retiring.push_back(first_stage_mirror);
                    ver.release_stage_mirror_graphs(); ver_b.release_stage_mirror_graphs();
                    for (auto& stage:stages) {
                        if (stage->expert_mirror) retiring.push_back(stage->expert_mirror);
                        const strata::core::OnDevice on(stage->dev);
                        dpct::get_current_device().queues_wait_and_throw();
                        stage->ver.release_stage_mirror_graphs(); stage->ver_b.release_stage_mirror_graphs();
                        stage->expert_mirror.reset();
                    }
                    for(auto& stage_owners:bslot_ss)for(auto& owner:stage_owners)owner->trace_graphs_retired();
                    bslot_ss.clear(); // all batch graph references retired; owning-context frees
                    vh.stage_mirror.reset(); first_stage_mirror.reset();
                    // Drop source aliases before checked release; no graph or source
                    // may retain a usable table after the owning allocations free.
                    // These events follow actual global token/session and all
                    // stage verifier/batch graph destruction above; source aliases
                    // are separately retired inside close() before owning release.
                    for(auto& owner:retiring)owner->trace_serving_graphs_retired();
                    gguf_src.close();
                    for (auto& owner:retiring) owner->release();
                } else {
                    ver.release_slot_graphs();
                    for(auto& stage:stages){const strata::core::OnDevice on(stage->dev);stage->ver.release_slot_graphs();}
                    for(auto& stage_owners:bslot_ss)for(auto& owner:stage_owners)owner->trace_graphs_retired();
                    bslot_ss.clear();gguf_src.close();
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata shutdown: mirror release failed: %s\\n", e.what());
                std::fflush(stderr);
                std::_Exit(1);
            }
            if (strata::program::host_trace36::enabled())
                strata::program::host_trace36::exit_export();
            if (!strata::program::diagnostic_pti_finalize43()) {
                std::fprintf(stderr,"strata diagnostic PTI finalization failed\\n");
                std::fflush(stderr);
                std::_Exit(1);
            }
            std::fflush(stdout);
            std::fflush(stderr);
            std::_Exit(0);
        }
'''
 source=source.replace(early,reserve+early,1)
 source=source.replace(late,late+hook,1)
 return '#include "strata/kernels/original_mtp53_sustained_greedy.hpp"\n'+source

def prepare(root,receipt,out):
 plan=json.loads(PLAN.read_bytes())
 for absolute,want in plan['actual52_control_source_anchors'].items():
  if hashlib.sha256(Path(absolute).read_bytes()).hexdigest()!=want:raise ValueError('Exact SDK52 control/snapshot source changed '+absolute)
 for rel,want in plan['own_source_sha256'].items():
  if hashlib.sha256((HERE.parents[1]/rel).read_bytes()).hexdigest()!=want:raise ValueError('MAIN diagnostic own source changed '+rel)
 root=Path(root);source=root/'source/sycl/src/program/generate.cpp';raw=source.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=plan['actual52_generate_sha256']:raise ValueError('Exact SDK52 native MAIN translation unit required')
 record=component.prepare(root,receipt,out);out=Path(out);overlay=out/'overlay'
 target=overlay/'sycl/include/strata/kernels/original_mtp53_loaded_main_adapter.hpp';target.write_bytes((HERE/'original-mtp53/loaded_main_adapter_v6.hpp').read_bytes())
 (overlay/'sycl/include/strata/kernels/original_mtp53_phase_timing.hpp').write_bytes((HERE/'original-mtp53/phase_timing_v1.hpp').read_bytes())
 (overlay/'sycl/include/strata/kernels/original_mtp53_draft_step.hpp').write_bytes((HERE/'original-mtp53/draft_step_v4.hpp').read_bytes())
 (overlay/'sycl/include/strata/kernels/original_mtp53_sustained_greedy.hpp').write_bytes((HERE/'original-mtp53/sustained_greedy_v2.hpp').read_bytes())
 generated=overlay/'sycl/src/program/original_mtp53_main_diagnostic.cpp';generated.parent.mkdir(parents=True,exist_ok=True);generated.write_text(transform(raw.decode('ascii')),encoding='ascii')
 argv=list(record['compile_commands'][0]['argv']);argv[argv.index('/overlay/sycl/src/kernels/original_mtp53_draft_step_loaded_fixture.cpp')]='/overlay/sycl/src/program/original_mtp53_main_diagnostic.cpp';argv[-1]='/out/original-mtp53-greedy64'
 record.update({'compile_commands':[{'name':'original-mtp53-greedy64','argv':argv}],'main_proposal_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'native_generate_original_sha256':hashlib.sha256(raw).hexdigest(),'native_generate_diagnostic_sha256':hashlib.sha256(generated.read_bytes()).hexdigest(),'real_MAIN_executed':False,'draft_numerical_qualified':False,'full64_equivalence_qualified':False})
 component.write(out/'compile-recipe.json',record);return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
