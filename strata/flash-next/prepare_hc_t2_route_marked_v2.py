"""Fresh marked assembly over immutable paired HC recipe; no compile/model/GPU."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PLAN=HERE/'hc-t2-route-marked-source-plan-v2.json'
PARENT=Path('/mnt/vm_8tb/b70/build/hc-t2-paired-mtp-compile-inputs-v1')
PARENT_RECIPE_SHA='fa613f414ac1d6b96d8d576132537eb9d556db8cfafc7644346f01930ad2da2f'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def replace_one(text,old,new):
 if text.count(old)!=1:raise ValueError('Unique HC proof anchor changed '+old[:100])
 return text.replace(old,new,1)
def marked_verify(source):
 text='#include "strata/core/hc_t2_route_proof.hpp"\n'+source
 text=replace_one(text,'    const OnDevice on_device(device_);\n    if (!native_hc_policy(err))', '    hc_t2_route_proof::Record hc_record(this,T,device_,int(lb_),int(le_));\n    const OnDevice on_device(device_);\n    if (!native_hc_policy(err))')
 old='const bool pair=t2_hc && T==2';text=replace_one(text,old,'const bool eligible=T==2')
 old='                    parts_ && K>=HC && max_t_>=2;\n                if(pair){'
 text=replace_one(text,old,'                    parts_ && K>=HC && max_t_>=2;\n                const bool pair=t2_hc && eligible;hc_record.note(eligible,pair);\n                if(pair){')
 text=replace_one(text,'    stamp(g.n_layers, 1, 0);\n    return true;', '    stamp(g.n_layers, 1, 0);\n    hc_record.finish();\n    return true;')
 text=replace_one(text,'    const dpct::err0 ue = 0;   // no cudaGraphUpload on SYCL:', '    hc_t2_route_proof::seal(this,(const void*)exec_t,T);\n    const dpct::err0 ue = 0;   // no cudaGraphUpload on SYCL:')
 # Keep the EXACT existing selection expression and submit call; identify its chosen graph separately.
 selection='prefix30_prompt_ ? prefix30_prompt_exec_[ar_off_?1:0][T] : (batch_observe_ && batch_admission_ && T==1 ? (batch_phase_==3?batch_direct_exec_[ar_off_?1:0]:batch_phase_==2?batch_migration_exec_[ar_off_?1:0]:batch_admission_exec_[ar_off_?1:0]) : layer0_first_ && T==1 ? layer0_exec_[(ar_off_?1:0)+(fidelity_first_?2:0)] : (fidelity_first_ && T==1 ? fidelity_exec_[ar_off_?1:0] : (ar_off_ ? exec_nr_[T] : exec_[T])))'
 if 'ext_oneapi_graph(*('+selection+'))'not in text:raise ValueError('Exact submitted SYCL graph selection changed')
 text=replace_one(text,'    const uint64_t h36_ReplaySubmit_id=', '    const void* hc_graph=(const void*)('+selection+');\n    bool hc_submitted=false;\n    const uint64_t h36_ReplaySubmit_id=')
 text=replace_one(text,'    (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));\n    VDBG("launched\\n");', '    if(std::getenv("STRATA_VERIFY_EAGER")==nullptr)hc_submitted=hc_t2_route_proof::submit(this,hc_graph,T,pos0);\n    (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));\n    VDBG("launched\\n");')
 # Existing successful drain and normal error predicate precede this completion hook.
 text=replace_one(text,"    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)", "    hc_t2_route_proof::complete(this,hc_graph,hc_submitted);\n    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)")
 # Earlier-stage graph body succeeds before the head/late finish; publish ONLY on that successful path.
 text=replace_one(text,'        if(batch_observe_ && batch_snapshot_)batch_snapshot_->stamp(*cs);\n        return true;\n    }\n\n    // ---- the head, T columns, and the argmax of each', '        if(batch_observe_ && batch_snapshot_)batch_snapshot_->stamp(*cs);\n        hc_record.finish();\n        return true;\n    }\n\n    // ---- the head, T columns, and the argmax of each')
 return text
def prepare(output):
 import shutil
 plan=json.loads(PLAN.read_bytes())
 for name,want in plan['files'].items():
  if sha(HERE.parents[1]/name)!=want:raise ValueError('Exact marked source changed '+name)
 output=Path(output)
 if not output.is_absolute()or output.resolve()!=output or output.exists():raise ValueError('Fresh canonical marked output required')
 if sha(PARENT/'compile-recipe.json')!=PARENT_RECIPE_SHA:raise ValueError('Exact preserved parent recipe changed')
 recipe=json.loads((PARENT/'compile-recipe.json').read_bytes())
 for name,want in recipe['overlay_sha256'].items():
  p=PARENT/'overlay'/name
  if p.is_symlink()or not p.is_file()or sha(p)!=want:raise ValueError('Original parent overlay changed '+name)
 output.mkdir();shutil.copytree(PARENT/'overlay',output/'overlay');(output/'output').mkdir()
 verify=output/'overlay/sycl/src/core/verify.cpp';verify.write_text(marked_verify(verify.read_text()),encoding='ascii')
 header=output/'overlay/sycl/include/strata/core/hc_t2_route_proof.hpp';header.write_bytes((HERE/'hc_t2_route_proof_v1.hpp').read_bytes())
 main=output/'overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp';text=main.read_text();text='#include "strata/core/hc_t2_route_proof.hpp"\n'+text
 text=replace_one(text,'            if(!ok || drive.d.failed){std::fflush(nullptr);std::_Exit(86);}', '            if(!ok || drive.d.failed){std::fflush(nullptr);std::_Exit(86);}\n            strata::core::hc_t2_route_proof::close();')
 main.write_text(text,encoding='ascii');recipe['mounts']={'SDK_RO':recipe['actual52']['SDK'],'overlay_RO':str(output/'overlay'),'output_RW':str(output/'output')}
 recipe.update(schema=3,marked_source_plan_sha256=sha(PLAN),marked_parent_recipe_sha256=PARENT_RECIPE_SHA,native_MAIN_two_stage_sha256=sha(main),overlay_sha256={str(p.relative_to(output/'overlay')):sha(p)for p in sorted((output/'overlay').rglob('*'))if p.is_file()},host_route_proof=True,host_route_proof_flag='STRATA_HC_T2_ROUTE_PROOF=1',Verifier_public_private_layout_changed=False,GPU_waits_queries_copies_added=0,model_or_speed_qualified=False)
 (output/'compile-recipe.json').write_text(json.dumps(recipe,indent=2)+'\n',encoding='ascii');return recipe
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();print(json.dumps(prepare(args.output),ensure_ascii=True))
