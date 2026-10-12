"""Pure stage-local extension of frozen HC T2 V1, no compile/GPU/model execution."""
import argparse,difflib,hashlib,json
from pathlib import Path
import prepare_hc_t2_caller_v1 as original
HERE=Path(__file__).resolve().parent;PLAN=HERE/'hc-t2-caller-source-proposal-v2.json'
def construct():
 files=original.construct();p='sycl/src/core/verify.cpp';s=files[p]
 old='next_==nullptr && lb_==0 && le_==g.n_layers && ext_stream_==nullptr'
 new='lb_>=0 && lb_<le_ && le_<=g.n_layers && (lb_==0 || hand_in_) && (le_==g.n_layers || hand_out_) &&\n                    hits_.stage_mirror->lb==lb_ && hits_.stage_mirror->le==le_ && (le_==g.n_layers ? next_==nullptr : next_!=nullptr) && ext_stream_==nullptr'
 files[p]=original.once(s,old,new);return files

def prepare(out):
 plan=json.loads(PLAN.read_bytes())
 for p,h in plan['original_sha256'].items():
  if hashlib.sha256((original.SDK/p).read_bytes()).hexdigest()!=h:raise ValueError('Original52 HC source changed')
 for p,h in plan['own_source_sha256'].items():
  if hashlib.sha256((HERE.parents[1]/p).read_bytes()).hexdigest()!=h:raise ValueError('Stage-local HC candidate source changed')
 files=construct();out=Path(out)
 if not out.is_absolute()or out.resolve()!=out or out.exists():raise ValueError('New canonical source output required')
 out.mkdir()
 for p,s in files.items():
  if hashlib.sha256(s.encode('ascii')).hexdigest()!=plan['constructed_sha256'][p]:raise ValueError('Exact stage-local HC construction changed')
  target=out/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s,encoding='ascii')
 record={'source_proposal_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'copied_sources':plan['constructed_sha256'],'compile_executed':False,'GPU_executed':False,'model_payload_read':False,'paired_target_qualified':False}
 (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.output),ensure_ascii=True))
