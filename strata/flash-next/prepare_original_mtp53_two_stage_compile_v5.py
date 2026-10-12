"""New source-only two-stage original-publisher diagnostic; root builds/runs only."""
import argparse,ast,hashlib,json
from pathlib import Path
import prepare_original_mtp53_step_compile_v2 as component
import prepare_original_mtp53_greedy64_compile_v3 as previous
HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-publisher-mtp53-two-stage-proposal-v5.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def previous_transform(source):
 early='    strata::core::MtpDrafter mtp;'
 late='        std::atomic<bool> stop_req{false};'
 if source.count(early)!=1 or source.count(late)!=1:raise ValueError('Exact native52 constructor/pre-stdin hooks required')
 reserve='''    std::unique_ptr<strata::kernels::original_mtp53::LoadedMainDraft53> originalPair;
    const char* originalPairMode=std::getenv("STRATA_ORIGINAL_MTP53_PAIR_MODE");
    if(originalPairMode){
        const bool controls=std::strcmp(originalPairMode,"controls")==0;
        if(!controls && std::strcmp(originalPairMode,"off")!=0 && std::strcmp(originalPairMode,"on")!=0){std::fprintf(stderr,"MTP53 pair unknown mode\\n");return 2;}
        const char* phase_timing=std::getenv("STRATA_ORIGINAL_MTP53_PHASE_TIMING");if(phase_timing && std::strcmp(phase_timing,"0")!=0){std::fprintf(stderr,"MTP53 pair phase timing must be OFF; old2048 row limit is not this scope\\n");return 2;}
        const char* path=std::getenv("STRATA_ORIGINAL_MTP53_MAIN_GGUF");const char* mirror=std::getenv("STRATA_VERIFY_MIRROR_STAGE_T1");
        if(!strata::core::original_mtp53::enabled() || !path || !*path || !o.serve || !native_pack || o.spec!=2 || !o.mtp.empty() || o.batch!=0 || o.pipeline_windows!=0 || stages.size()!=1 || !last_st || last_st->dev!=1 || last_st->lb!=24 || last_st->le!=48 || (split_at.size()!=1 || split_at[0]!=24) || g.n_layers!=48 || g.n_embd!=2560 || g.hc!=4 || o.max_context!=4096 || o.prefill_chunk!=512 || !o.no_prefill_borrow || o.prompt_cache!=0 || o.conversation_cache_mib!=0 || o.adapt_every!=0 || (mirror && std::strcmp(mirror,"1")==0) || !o.stop_eos || o.eos_ids!=std::vector<int64_t>{248044,248046}){std::fprintf(stderr,"MTP53 pair exact original/native24+24/spec2/context4096/pref512/noBorrow/cache0/mirrorT1off guards differ\\n");return 2;}
        if(controls ? (o.tokens!=std::vector<int64_t>{248045} || o.max_new!=64) : (o.tokens.size()!=2048 || o.max_new!=512)){std::fprintf(stderr,"MTP53 pair actual control/natural request shape differs\\n");return 2;}
        const strata::core::OnDevice on(last_st->dev);
        // BOTH execution arms reserve the same original owned tensors/state BEFORE cache sizing.
        originalPair.reset(new strata::kernels::original_mtp53::LoadedMainDraft53(path,dpct::get_in_order_queue(),last_st->dev,int(o.max_context)));
        std::fprintf(stderr,"MTP53 PAIR reserve_device=1 context4096 original_tensor_bytes=2775258112 both_execution_arms_resident=1\\n");
    }
'''
 hook='''
        if(originalPairMode){
            if(!stage_mirrors || !originalPair || !last_st){std::fprintf(stderr,"MTP53 pair owned mirrors/stage/draft missing\\n");std::fflush(nullptr);std::_Exit(86);}
            originalPair->bind_target(ver,last_st->ver);
            drive.d.plan=ver.plan_sink();
            const std::vector<strata::kernels::original_mtp53::TargetStage53> owners{{&ver,&ss,0,0,24},{&last_st->ver,&last_st->ss,1,24,48}};
            const bool controls=std::strcmp(originalPairMode,"controls")==0;
            const bool ok=controls?originalPair->exercise_pair(ver,owners,g,win_pool_fn,win_pool_user):strata::kernels::original_mtp53::original_pair_greedy53(*originalPair,std::strcmp(originalPairMode,"on")==0,ver,sp,last_st->sp,o.tokens,int(o.max_new),win_pool_fn,win_pool_user);
            if(!ok || drive.d.failed){std::fflush(nullptr);std::_Exit(86);}
'''
 # Reuse exact original shutdown, including BOTH context drains before graph/source/mirror release.
 tree=ast.parse(Path(previous.__file__).read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='transform');oldhook=next(n.value.value for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='hook'for t in n.targets))
 shutdown=oldhook[oldhook.index('            try {'):].replace('original53.reset();','originalPair.reset();')
 hook+=shutdown
 return '#include "strata/kernels/original_mtp53_pair_greedy.hpp"\n'+source.replace(early,reserve+early,1).replace(late,hook+late,1)
def fresh_reset_body(source):
 start='            auto public_reset_main=[&](std::string& e)->bool{'
 end='            };\n            int64_t reread_to'
 if source.count(start)!=1 or source.count(end)!=1:raise ValueError('Unique actual native52 public fresh reset required')
 body=source[source.index(start)+len(start):source.index(end,source.index(start))]
 cleanup='                live.clear();live_imgs.clear();checks.clear();live_ok=false;return true;'
 if body.count(cleanup)!=1:raise ValueError('Exact public host request bookkeeping tail required')
 # This hook precedes request bookkeeping: there is no live/checkpoint branch to clear.
 return body.replace(cleanup,'                return true;',1)

def reset_hook(source):
 return '''
            // Same actual52 public fresh reset for BOTH target owners, before any draft bind or target run.
            if(ss.layer_lo!=0 || ss.layer_hi!=24 || last_st->ss.layer_lo!=24 || last_st->ss.layer_hi!=48){std::fprintf(stderr,"MTP53 pair fresh reset carve mismatch\\n");std::fflush(nullptr);std::_Exit(86);}
            auto originalPairFreshReset=[&](std::string& e)->bool{'''+fresh_reset_body(source)+'''
            };
            std::string originalPairResetError;
            if(!originalPairFreshReset(originalPairResetError)){std::fprintf(stderr,"MTP53 pair fresh reset failed: %s\\n",originalPairResetError.c_str());std::fflush(nullptr);std::_Exit(86);}
            std::fprintf(stderr,"MTP53_PAIR_FRESH_RESET stages=2 devices=0,1 ranges=0:24,24:48 native52_public_reset=1 before_target_bind=1\\n");
'''

def transform(source):
 result=previous_transform(source);anchor='            originalPair->bind_target(ver,last_st->ver);'
 if result.count(anchor)!=1:raise ValueError('Unique owned pair target binding required')
 return result.replace(anchor,reset_hook(source)+anchor,1)

def prepare(root,receipt,out):
 plan=json.loads(PLAN.read_bytes())
 for name,want in plan['own_source_sha256'].items():
  if sha(HERE.parents[1]/name)!=want:raise ValueError('Exact pair source changed '+name)
 root=Path(root);raw=(root/'source/sycl/src/program/generate.cpp').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=plan['actual52_generate_sha256']:raise ValueError('Exact actual52 MAIN source required')
 out=Path(out);record=component.prepare(root,receipt,out);overlay=out/'overlay'
 for name,target in plan['new_overlay_sources'].items():
  p=overlay/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((HERE.parents[1]/name).read_bytes())
 generated=overlay/'sycl/src/program/original_mtp53_two_stage_diagnostic.cpp';generated.parent.mkdir(parents=True,exist_ok=True);generated.write_text(transform(raw.decode('ascii')),encoding='ascii')
 argv=list(record['compile_commands'][0]['argv']);argv[argv.index('/overlay/sycl/src/kernels/original_mtp53_draft_step_loaded_fixture.cpp')]='/overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp';argv[-1]='/out/original-mtp53-two-stage'
 record.update(compile_commands=[{'name':'original-mtp53-two-stage','argv':argv}],two_stage_source_plan_sha256=sha(PLAN),native_MAIN_original_sha256=hashlib.sha256(raw).hexdigest(),native_MAIN_two_stage_sha256=sha(generated),source_only=True,model_runtime_qualified=False,prefill_serial_REFERENCE_comparison_required=True)
 (out/'compile-recipe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
