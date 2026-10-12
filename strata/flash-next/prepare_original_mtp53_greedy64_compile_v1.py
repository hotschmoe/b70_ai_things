"""Prepare a standalone SDK52 native MAIN diagnostic, without executing it."""
import argparse,hashlib,json
from pathlib import Path
import prepare_original_mtp53_step_compile_v2 as component
HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-publisher-mtp53-greedy64-proposal-v1.json'
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
            drive.d.plan=ver.plan_sink();
            if(!strata::kernels::original_mtp53::original_greedy53(original53.get(),ver,o.tokens,int(o.max_new),win_pool_fn,win_pool_user) || drive.d.failed){
                // Failure is latched; avoid destructing target queues/state that may still be active.
                std::fflush(nullptr);std::_Exit(86);
            }
            return 0;
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
 target=overlay/'sycl/include/strata/kernels/original_mtp53_loaded_main_adapter.hpp';target.write_bytes((HERE/'original-mtp53/loaded_main_adapter_v5.hpp').read_bytes())
 (overlay/'sycl/include/strata/kernels/original_mtp53_draft_step.hpp').write_bytes((HERE/'original-mtp53/draft_step_v3.hpp').read_bytes())
 (overlay/'sycl/include/strata/kernels/original_mtp53_sustained_greedy.hpp').write_bytes((HERE/'original-mtp53/sustained_greedy_v1.hpp').read_bytes())
 generated=overlay/'sycl/src/program/original_mtp53_main_diagnostic.cpp';generated.parent.mkdir(parents=True,exist_ok=True);generated.write_text(transform(raw.decode('ascii')),encoding='ascii')
 argv=list(record['compile_commands'][0]['argv']);argv[argv.index('/overlay/sycl/src/kernels/original_mtp53_draft_step_loaded_fixture.cpp')]='/overlay/sycl/src/program/original_mtp53_main_diagnostic.cpp';argv[-1]='/out/original-mtp53-greedy64'
 record.update({'compile_commands':[{'name':'original-mtp53-greedy64','argv':argv}],'main_proposal_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'native_generate_original_sha256':hashlib.sha256(raw).hexdigest(),'native_generate_diagnostic_sha256':hashlib.sha256(generated.read_bytes()).hexdigest(),'real_MAIN_executed':False,'draft_numerical_qualified':False,'full64_equivalence_qualified':False})
 component.write(out/'compile-recipe.json',record);return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
