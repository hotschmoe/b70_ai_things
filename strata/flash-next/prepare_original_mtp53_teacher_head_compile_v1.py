"""Default-OFF teacher-only original-head omission; root builds/runs only."""
import argparse,hashlib,json
from pathlib import Path
import prepare_original_mtp53_two_stage_compile_v6 as parent
HERE=Path(__file__).resolve().parent;PLAN=HERE/'original-mtp53-teacher-head-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(root,receipt,out):
 plan=json.loads(PLAN.read_bytes())
 for n,w in plan['files'].items():
  if sha(HERE.parents[1]/n)!=w:raise ValueError('Teacher head source changed '+n)
 out=Path(out);r=parent.prepare(root,receipt,out)
 for source,target in plan['overlay_sources'].items():
  p=out/'overlay'/target;p.write_bytes((HERE.parents[1]/source).read_bytes())
 r.update(teacher_head_source_plan_sha256=sha(PLAN),teacher_head_overlay_sha256={t:sha(out/'overlay'/t)for t in plan['overlay_sources'].values()},only_matched_delta='STRATA_ORIGINAL_MTP53_TEACHER_HEAD_SKIP=0/1',first_proposal_expected_counters={'teacher_rows':2047,'OFF_calls':2048,'OFF_skips':0,'ON_calls':1,'ON_skips':2047,'committed_cells':2048},runtime_qualified=False,head_skip_only_in_teacher_prefill=True)
 (out/'compile-recipe.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
