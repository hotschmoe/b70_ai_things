#!/usr/bin/env python3
"""CPU-only complete buffered original-shard identity after genuine combined20 upload."""
import argparse,time
from pathlib import Path
import qualify_serial_prefix_v6 as wrapper
ROOT=Path(__file__).resolve().parents[2]
ENGINE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T112318Z-4uag6j5x')
ENGINE_SHA='1647fdb0941d6e960bdbded793849b998de9de963eb5a6e6c687f18ea5547b00'

def validate_upload_binding(upload_path,oracle_path):
 wrapper.require(wrapper.sha(ENGINE/'receipt.json')==ENGINE_SHA,'Actual combined20 engine receipt changed')
 engine=wrapper.read(ENGINE/'receipt.json');wrapper.require(engine.get('build_rc')==0 and engine.get('plan_sha256')==wrapper.ENGINE_PLAN_SHA,'Combined20 build/source plan differs')
 upload=wrapper.read(upload_path);oracle=wrapper.read(oracle_path)
 wrapper.require(upload.get('passed') and upload.get('post_health_passed') and upload.get('finished_epoch',0)>0,'Genuine completed source upload/post-health required before current hash')
 wrapper.require(upload.get('oracle_schema')==2 and len(upload.get('cases',[]))==5,'Whole390 cases/lifecycle incomplete')
 wrapper.require(upload.get('oracle_receipt_sha256')==wrapper.sha(oracle_path),'Upload oracle receipt association differs')
 wrapper.require(oracle.get('passed') and oracle.get('engine_receipt_sha256')==ENGINE_SHA and Path(oracle.get('engine_receipt','')).resolve()==(ENGINE/'receipt.json').resolve(),'Upload oracle targets another engine generation/path')
 wrapper.require(wrapper.sha(Path(oracle['engine_receipt']))==oracle['engine_receipt_sha256'],'Oracle actual engine receipt changed')
 wrapper.require(upload.get('oracle_plan_sha256')==oracle.get('plan_sha256'),'Upload/oracle source plan association differs')
 return upload

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upload-lifecycle',type=Path,required=True);p.add_argument('--oracle-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 upload=validate_upload_binding(a.upload_lifecycle,a.oracle_receipt)
 lock_path=ROOT/'strata/flash-next/model-lock.json';lock=wrapper.read(lock_path);shards=[ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')]
 a.output.mkdir(parents=True,exist_ok=False)
 result=wrapper.full_buffered_identity(lock_path,lock,shards,a.output/'receipt.json',upload['finished_epoch'])
 result['scope']='Actual independent complete original four-shard buffered hashes/stat before-after after combined20 upload; no GPU/model launch/cache repair'
 result['source_upload_lifecycle_sha256']=wrapper.sha(a.upload_lifecycle);result['combined20_engine_receipt_sha256']=ENGINE_SHA
 result['oracle_receipt_sha256']=wrapper.sha(a.oracle_receipt);result['hash_helper_sha256']=wrapper.sha(Path(__file__))
 wrapper.write(a.output/'receipt.json',result)
 print('PASSED',result['passed'],'IDENTITY',a.output/'receipt.json')
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
