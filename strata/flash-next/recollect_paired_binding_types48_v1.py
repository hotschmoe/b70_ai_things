#!/usr/bin/env python3
"""Root-only current CPU proof/type diagnosis using explicit pack48 gate ports.
No GPU or model inference. Actual pack byte reads are root-only execution.
"""
import argparse,json,hashlib
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
from operation_pack_hash_witness_v2 import for_prepared,require
from paired_control_pack48_v1 import finalized_binding

def differences(actual,saved,path='$'):
 rows=[]
 if type(actual)!=type(saved):rows.append({'path':path,'actual_type':type(actual).__name__,'saved_type':type(saved).__name__})
 if isinstance(actual,dict) and isinstance(saved,dict):
  # canonical first rejects duplicate JSON-converted keys/nonfinite values.
  canonical(actual);keys={next(iter(json.loads(json.dumps({k:0},allow_nan=False)))):k for k in actual}
  for converted,key in keys.items():
   if converted in saved:
    if type(key)is not str:rows.append({'path':path+'.'+converted,'actual_key_type':type(key).__name__,'saved_key_type':'str'})
    rows.extend(differences(actual[key],saved[converted],path+'.'+converted))
 elif isinstance(actual,(tuple,list)) and isinstance(saved,(tuple,list)) and len(actual)==len(saved):
  for index,(a,b) in enumerate(zip(actual,saved)):rows.extend(differences(a,b,path+'['+str(index)+']'))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New immutable type recollection required');source_path=Path(__file__).resolve().parent/'paired-native-binding-types48-source-plan-v1.json';source_raw=source_path.read_bytes();source=read_unique(source_path);repo=Path(__file__).resolve().parents[2]
 for name,digest in source['files'].items():require(hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,'Current type recollector source changed '+name)
 raw=a.plan.read_bytes();plan=read_unique(a.plan);epoch=for_prepared(Path(plan['prepared']),3600)
 actual=finalized_binding(plan['paired_two_control'],plan['engine_receipt_sha256'],pack_epoch=epoch);saved=plan['paired_two_control_binding'];native_equal=actual==saved;canonical_equal=canonical(actual)==canonical(saved);types=differences(actual,saved);epoch.seal_predevice();proof=epoch.finalize();require(a.plan.read_bytes()==raw,'Original failed plan changed during recollection')
 require(source_path.read_bytes()==source_raw,'Type recollector source plan changed')
 for name,digest in source['files'].items():require(hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,'Type recollector source changed after actual traversal '+name)
 result={'source_plan_sha256':hashlib.sha256(source_raw).hexdigest(),'recollector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_binding':source['files'],'schema':1,'passed':canonical_equal,'native_python_equal':native_equal,'typed_canonical_JSON_equal':canonical_equal,'actual_binding_canonical_sha256':hashlib.sha256(canonical(actual).encode()).hexdigest(),'saved_binding_canonical_sha256':hashlib.sha256(canonical(saved).encode()).hexdigest(),'native_type_differences':types,'plan_path':str(a.plan.resolve()),'plan_sha256':hashlib.sha256(raw).hexdigest(),'pack_operation_witness':proof,'original_parent_failure_rewritten':False,'actual_GPU_touch':False,'actual_model_inference':False,'full_math_quality_or_latency_qualified':False}
 with a.output.open('x') as handle:handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'typed_canonical_JSON_equal':canonical_equal,'native_python_equal':native_equal,'type_differences':len(types),'actual_GPU_touch':False}));return 0 if canonical_equal else 1
if __name__=='__main__':raise SystemExit(main())
