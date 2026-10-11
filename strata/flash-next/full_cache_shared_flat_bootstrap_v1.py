#!/usr/bin/env python3
"""ROOT no-device actual-image smoke: imports plus pinned tokenizer EOS only."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();sys.path.insert(0,'/src')
 from full_cache_flat_producer_import_v1 import roster
 rows=roster('/controller')
 for name in rows:importlib.import_module(name)
 from serve import server
 from c1_trace_contract import pinned_eos
 cfg=json.loads(a.config.read_bytes());eos=pinned_eos(cfg)
 result={'flat_imports_complete':True,'controller_mount':'/controller','controller_modules':{n:r['sha256']for n,r in rows.items()},'server_source_sha256':hashlib.sha256(Path(server.__file__).read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),'pinned_eos_ids':eos,'native_engine_started':False,'model_payload_mounted':False,'GPU_devices_mounted':False}
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
