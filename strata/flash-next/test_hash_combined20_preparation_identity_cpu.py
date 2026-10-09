#!/usr/bin/env python3
"""Actual upload association and tiny buffered-reader boundaries; no GPU/full scan."""
import copy,hashlib,json,tempfile,time
from pathlib import Path
from unittest.mock import patch
import hash_combined20_preparation_identity as helper


def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Wrong identity association accepted')

def main():
 upload=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f07-20261009/source-upload-combined20-v1/receipt.json');oracle=Path('/mnt/vm_8tb/b70/build/strata-source-upload-oracle-full-tkbp0s5d/receipt.json')
 actual=helper.validate_upload_binding(upload,oracle);assert actual['passed']
 with tempfile.TemporaryDirectory(prefix='combined20-hash-helper-cpu-') as name:
  root=Path(name);oracle_old=root/'old-oracle.json';old=helper.wrapper.read(oracle);old.update(engine_receipt='/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp/receipt.json',engine_receipt_sha256=helper.wrapper.sha(Path(old['engine_receipt']).parents[0]/'receipt.json'))
  # A matching upload->oracle SHA must not conceal the old engine association.
  old['engine_receipt_sha256']=helper.wrapper.sha(Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp/receipt.json'));helper.wrapper.write(oracle_old,old)
  upload_old=root/'old-upload.json';reused=copy.deepcopy(actual);reused['oracle_receipt_sha256']=helper.wrapper.sha(oracle_old);helper.wrapper.write(upload_old,reused);reject(lambda:helper.validate_upload_binding(upload_old,oracle_old))
  changed=copy.deepcopy(actual);changed['oracle_receipt_sha256']='0'*64;wrong=root/'wrong.json';helper.wrapper.write(wrong,changed);reject(lambda:helper.validate_upload_binding(wrong,oracle))
  wrong_oracle=copy.deepcopy(helper.wrapper.read(oracle));wrong_oracle['engine_receipt']=str(root/'other-engine.json');helper.wrapper.write(oracle_old,wrong_oracle);changed=copy.deepcopy(actual);changed['oracle_receipt_sha256']=helper.wrapper.sha(oracle_old);helper.wrapper.write(wrong,changed);reject(lambda:helper.validate_upload_binding(wrong,oracle_old))
  lock_path=root/'lock.json';files=[];shards=[]
  for index in range(4):
   path=root/f'part{index}.gguf';raw=bytes([index])*8192;path.write_bytes(raw);shards.append(path);files.append({'path':f'UD-Q4_K_XL/part{index}.gguf','size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  lock={'files':files,'revision':'CPU_SYNTHETIC_ONLY'};helper.wrapper.write(lock_path,lock)
  result=helper.wrapper.full_buffered_identity(lock_path,lock,shards,root/'tiny.json',time.time()-1);assert result['passed'] and len(result['rows'])==4
  bad=copy.deepcopy(lock);bad['files'][2]['sha256']='0'*64;assert not helper.wrapper.full_buffered_identity(lock_path,bad,shards,root/'tiny-failure.json',time.time()-1)['passed']
 receipt={'CONFIG':'actual new20 upload/oracle/engine/sourceplan readonly binding plus tiny synthetic shards; no rescan/GPU','COMMAND':'python3 strata/flash-next/test_hash_combined20_preparation_identity_cpu.py','RESULT':{'actual_authentic_association':True,'matching_old18_upload_oracle_rejected':True,'wrong_oracle_sha_and_engine_path_rejected':True,'tiny_all4_buffered_and_hash_failure':True},'VERDICT':'PASS helper gates only; original genuine full-hash receipt preserved unchanged','source_sha256':helper.wrapper.sha(Path(helper.__file__)),'original_executed_helper_sha256':helper.wrapper.sha(Path(helper.__file__).with_name('hash_combined20_preparation_identity_original_v1.py'))}
 Path(helper.__file__).with_name('combined20-hash-helper-cpu-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS actual combined20 association; old18 rejected; tiny reader controls, no full scan')
if __name__=='__main__':main()
