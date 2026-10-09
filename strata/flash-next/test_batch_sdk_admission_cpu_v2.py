#!/usr/bin/env python3
"""Read actual source29 SDK identity only; deny every model/pack payload open."""
import builtins,io,json
from pathlib import Path
from batch_numerical_proofs_v2 import engine_binding,PLAN_SHA,sha
ENGINE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T185206Z-d5q32zc7')
EXPECTED='bd17b41b0074b816ab88d3804af78727480291f96d2510e5759b9788091b3c2d'

def main():
 assert sha(ENGINE/'receipt.json')==EXPECTED
 real_open=builtins.open;real_io=io.open;denied=[]
 def guard(fn):
  def guarded(path,*args,**kwargs):
   if not isinstance(path,int):
    name=str(Path(path).resolve())
    if '/b70/models/' in name or '/models/files/' in name or name.endswith('.gguf') or '/flashnext-native-source-pack-' in name:denied.append(name);raise AssertionError('Forbidden model/pack payload open in SDK-only admission test')
   return fn(path,*args,**kwargs)
  return guarded
 builtins.open=guard(real_open);io.open=guard(real_io)
 try:
  actual=engine_binding(ENGINE);assert actual['plan_sha256']==PLAN_SHA and len(actual['patched_source_sha256'])==60 and len(actual['binary_sha256'])==8
 finally:builtins.open=real_open;io.open=real_io
 assert not denied
 print(json.dumps({'passed':True,'actual_engine_receipt_sha256':EXPECTED,'actual_source_files':60,'actual_ELF_binaries':8,'model_or_pack_payload_opens':0,'scope':'Read-only actual SDK/source identity only; no genuine C1 prepare, model arithmetic, GPU, SDK compilation or serving'},ensure_ascii=True))
if __name__=='__main__':main()
