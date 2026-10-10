#!/usr/bin/env python3
"""CPU metadata case port only; original prompt corpus and budgets unchanged."""
import argparse,copy,json
from pathlib import Path
from batch_numerical_proofs_v40 import read,write,sha,require,ROOT,PLAN,PLAN_SHA
HERE=Path(__file__).resolve().parent
def prepare(slots,paired_two_control=None):
 require(type(slots) is int and slots in (2,4,6),'Declared requested2/4/bounded6 scope required')
 old=HERE/('batch-numerical-case'+str(slots)+'-source35-v2.json');spec=copy.deepcopy(read(old));plan=read(ROOT/PLAN);require(sha(ROOT/PLAN)==PLAN_SHA,'Exact combined36+37 build recipe changed')
 require(spec['schema']==2 and spec['harness_generation']==7 and spec['source_lane']=='source35','Frozen V7 originalcorpus declaration required')
 spec.update(harness_generation=40,source_lane='source37',source_plan_sha256=PLAN_SHA,source_sha256=plan['expected_patched_source_sha256'],original_case={'path':str(old),'sha256':sha(old)},actual_current_source37_runtime_qualified=False)
 if paired_two_control is not None:
  require(slots>2,'Firstpaired2 doesnot borrow laterprogression control');spec['paired_two_control']=paired_two_control
 return spec
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--slots',type=int,choices=[2,4,6],required=True);p.add_argument('--paired-two-control',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New immutable case path required');write(a.output,prepare(a.slots,read(a.paired_two_control) if a.paired_two_control else None));print(json.dumps({'actual_GPU_run':False,'case_sha256':sha(a.output)}))
