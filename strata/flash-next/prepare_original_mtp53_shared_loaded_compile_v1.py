"""Root-only one-pack/private-slot fixture recipe; no model or compiler execution."""
import argparse,hashlib,json
from pathlib import Path
import prepare_original_mtp53_two_stage_compile_v6 as parent
HERE=Path(__file__).resolve().parent;PLAN=HERE/'original-mtp53-shared-loaded-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(root,receipt,out):
 plan=json.loads(PLAN.read_bytes())
 for n,w in plan['files'].items():
  if sha(HERE.parents[1]/n)!=w:raise ValueError('Shared loaded source changed '+n)
 out=Path(out);r=parent.prepare(root,receipt,out)
 for source,target in plan['overlay_sources'].items():
  p=out/'overlay'/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((HERE.parents[1]/source).read_bytes())
 argv=list(r['compile_commands'][0]['argv']);old='/overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp';argv[argv.index(old)]='/overlay/sycl/src/kernels/original_mtp53_shared_slots_loaded_fixture.cpp';argv[-1]='/out/original-mtp53-shared-slots'
 r.update(shared_loaded_source_plan_sha256=sha(PLAN),compile_commands=[{'name':'original-mtp53-shared-slots','argv':argv}],shared_overlay_sha256={t:sha(out/'overlay'/t)for t in plan['overlay_sources'].values()},fixture_scope='Synthetic original34 one shared GPU1 pack/private slots; MAIN/model fidelity/runtime unqualified',actual_runtime_requires_fresh_original_whole_sha=True,MAIN_model_mount_required=False,compiler_executed=False)
 (out/'compile-recipe.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
