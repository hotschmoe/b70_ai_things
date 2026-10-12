"""Root-only fresh paired MAIN/Verifier/HC object recipe; no compiler or model reads."""
import argparse,hashlib,json,shlex
from pathlib import Path
import prepare_original_mtp53_two_stage_compile_v6 as parent
import prepare_hc_t2_caller_v2 as candidate
HERE=Path(__file__).resolve().parent
PLAN=HERE/'hc-t2-paired-mtp-compile-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def modularize(record):
 old=record['compile_commands'][0]['argv'];source='/overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp'
 if old.count(source)!=1 or old.count('-Wl,--start-group')!=1:raise ValueError('Exact paired V6 MAIN link required')
 split=old.index('-Wl,--start-group');base=old[:split];base.remove(source)
 # Original monolithic argv has no output before the archive group.
 commands=[]
 for src,obj in [(source,'main.hc-t2.o'),('/overlay/sycl/src/core/verify.cpp','verify.hc-t2.o'),('/overlay/sycl/src/kernels/hc_native_projection.cpp','hc_projection.hc-t2.o'),('/overlay/sycl/src/kernels/hc_native_composition.cpp','hc_composition.hc-t2.o')]:
  commands.append({'name':obj,'argv':[*base,'-I/sdk/source/sycl/src/kernels/cuda','-c',src,'-o','/out/'+obj]})
 link=old[split:];link[link.index('/sdk/build/libstrata_engine.a')]='/out/libstrata_engine.without_verify.a';link[link.index('/sdk/build/libstrata_kernels.a')]='/out/libstrata_kernels.without_hc.a';link[-1]='/out/original-mtp53-two-stage-hc-t2'
 filters=[]
 for archive,members,target in [('libstrata_engine.a',['verify.cpp.o'],'libstrata_engine.without_verify.a'),('libstrata_kernels.a',['hc_native_projection.cpp.o','hc_native_composition.cpp.o'],'libstrata_kernels.without_hc.a')]:
  filters.append({'original':'/sdk/build/'+archive,'copy':'/out/'+target,'remove_members':members,'before_counts':[1]*len(members),'after_counts':[0]*len(members),'copy_command':['cp','/sdk/build/'+archive,'/out/'+target],'remove_command':['ar','d','/out/'+target,*members],'original_never_modified':True})
 return commands,[old[0],*shlex.split(record['production_link_flags']),*['/out/'+c['name']for c in commands],*link],filters

def prepare(root,receipt,out):
 plan=json.loads(PLAN.read_bytes())
 for name,want in plan['files'].items():
  if sha(HERE.parents[1]/name)!=want:raise ValueError('Paired HC compile source changed '+name)
 root=Path(root);out=Path(out);record=parent.prepare(root,receipt,out)
 # Both source constructors independently authenticate immutable original52 inputs.
 temp=out/'hc-source';candidate.prepare(temp)
 for name,want in json.loads(candidate.PLAN.read_bytes())['constructed_sha256'].items():
  src=temp/name
  if sha(src)!=want:raise ValueError('Exact paired HC overlay required')
  target=out/'overlay'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(src.read_bytes())
 # Ensure the original production Verifier uses the same precision/SYCL flags.
 ninja=(root/'build/build.ninja').read_text();needle='build CMakeFiles/strata_engine.dir/src/core/verify.cpp.o:'
 if ninja.count(needle)!=1:raise ValueError('Unique original Verifier compile rule required')
 block=ninja.split(needle,1)[1].split('\n\n',1)[0]
 flags=next(line.split(' = ',1)[1]for line in block.splitlines()if line.strip().startswith('FLAGS = '))
 if flags!=record['object_fields']['FLAGS']:raise ValueError('Verifier and HC production flags differ')
 commands,link,filters=modularize(record)
 record.update(schema=2,compile_commands=commands,link_argv=link,archive_filters=filters,hc_source_plan_sha256=sha(PLAN),paired_hc_proposal_sha256=sha(candidate.PLAN),overlay_sha256={str(p.relative_to(out/'overlay')):sha(p)for p in sorted((out/'overlay').rglob('*'))if p.is_file()},original_archive_sha256={a:sha(root/'build'/a)for a in ['libstrata_engine.a','libstrata_kernels.a']},scope='Default-OFF paired original MTP V6 target; no arithmetic/layout change; runtime and benefit unqualified',actual_compiled_receipt_ELF_required=True,expert_pair_reuse_required=False,runtime_env={'STRATA_EXPERT_PAIR_REUSE':'0','STRATA_VERIFY_HC_T2_REUSE':'0 or 1 only matched delta'},paired_model_qualified=False)
 (out/'compile-recipe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
