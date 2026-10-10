#!/usr/bin/env python3
"""CPU metadata admission for genuinely NEW source40 whole390 oracle/runtime."""
import argparse,json
from pathlib import Path
import c1_serve_controller_combined_v140_v1 as c
HERE=Path(__file__).resolve().parent
PLAN=HERE/'native-source-upload-plan-full-source40-v1.json'
PLAN_SHA='a9fb076ca12050b70b2c43120f97ad1e20f42772f76c8adca6d3eccf538dd067'
BUILDER=HERE/'build_source_upload_oracle_full.py'
RUNNER=HERE/'run_source_upload_oracle_full_v2.py'
def admission(engine,oracle=None):
 generation=c.combined_generation_gate(Path(engine));c.require(c.sha(PLAN)==PLAN_SHA,'Exact newsource40 upload proposal changed')
 plan=c.read(PLAN);c.require(plan['source_generation']==40 and plan['engine_source_plan']['sha256']==generation['plan_sha256'] and plan['runtime_environment'].get('STRATA_CRITICAL_PATH_TRACE')=='0' and plan['runtime_environment'].get('STRATA_FULL_CACHE_OBSERVER38')=='0' and plan['runtime_environment'].get('STRATA_QSA3_TARGET')=='0','Explicit source40/H36OFF whole390 recipe required')
 result={'source_generation':40,'combined_generation':generation,'upload_plan':str(PLAN),'upload_plan_sha256':PLAN_SHA,'actual_upload_qualified':False,'ordinary_weight_payload_readback_qualified':False,'actual_native_QSA_targets_qualified':False,'full_model_math_qualified':False}
 if oracle is not None:
  oracle=Path(oracle).resolve();r=c.read(oracle);root=oracle.parent
  c.require(r.get('passed') is True and r.get('build_rc')==0 and r.get('libraries_unchanged') is True and r.get('image')==c.BASE_IMAGE,'Actual fresh oracle compile prerequisite absent')
  c.require(r['engine_receipt_sha256']==generation['engine_receipt_sha256'] and Path(r['engine_receipt']).resolve()==Path(generation['engine_receipt']).resolve() and r['plan_sha256']==PLAN_SHA==c.sha(root/'plan.snapshot.json') and c.read(root/'plan.snapshot.json')==plan,'Oldsource35/oracleplan/SDK cannot transfer')
  c.require(c.sha(root/'oracle.cpp')==r['oracle_source_sha256']==plan['oracle_source_sha256'] and c.sha(root/'source-upload-oracle')==r['binary_sha256'],'Actual fresh source/oracle ELF identity changed')
  with (root/'source-upload-oracle').open('rb') as stream:c.require(stream.read(4)==b'\x7fELF','Actual compiled oracle ELF required')
  c.require(r['library_sha256'] and all(c.sha(Path(path))==digest for path,digest in r['library_sha256'].items()),'Actual source40 archive association changed')
  result['actual_fresh_oracle_compile']={'receipt':str(oracle),'receipt_sha256':c.sha(oracle),'binary_sha256':r['binary_sha256'],'current_library_sha256':r['library_sha256']}
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine-root',type=Path,required=True);p.add_argument('--oracle-receipt',type=Path);a=p.parse_args();print(json.dumps(admission(a.engine_root,a.oracle_receipt),indent=2))
