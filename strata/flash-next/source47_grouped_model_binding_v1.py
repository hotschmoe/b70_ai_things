"""Actual fresh SDK47 bytes; no SDK43 runtime or model qualification transfer."""
import hashlib,json
from pathlib import Path
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T223956Z-0b555kgw')
RECEIPT_SHA='31a2227f167dac12c326c22faa7c550940b7e8fb3f892bc1a63ea9e0647cc4d7'
HERE=Path(__file__).resolve().parent
PLAN=HERE/'native47-grouped-f16-usm-engine-build-proposal-v1.json'
PLAN_SHA='e01caf2bfbdeda5c22c4d67b666431b744c2eb5b2e2540cdd0fcce2baf48c6c2'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 path=Path(path);require(path.resolve()==path and path.is_file()and not path.is_symlink(),'Canonical current SDK file required');before=path.stat();h=hashlib.sha256()
 with path.open('rb')as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'SDK file changed during hash');return h.hexdigest()
def binding():
 require(sha(PLAN)==PLAN_SHA and sha(SDK/'receipt.json')==RECEIPT_SHA,'Exact actual47 plan/receipt required');plan=json.loads(PLAN.read_bytes());receipt=json.loads((SDK/'receipt.json').read_bytes());require(type(receipt['build_rc'])is int and receipt['build_rc']==0 and receipt['devices_exposed']is False and receipt['external_source_unchanged']is True and receipt['plan_snapshot_unchanged']is True and receipt['plan_sha256']==PLAN_SHA,'Fresh actual47 build/isolation proof required')
 require(len(plan['expected_patched_source_sha256'])==72 and len(receipt['binary_sha256'])==8,'All72 source/all8 fresh ABI required')
 for name,want in plan['expected_patched_source_sha256'].items():require(sha(SDK/'source'/name)==want,'Actual47 patched source changed '+name)
 for name,want in receipt['binary_sha256'].items():require(Path(name).parent==SDK/'build'and sha(name)==want,'Actual47 binary changed '+name)
 require(sha(PLAN)==PLAN_SHA and sha(SDK/'receipt.json')==RECEIPT_SHA,'Closing SDK47 proof changed')
 return {'source_generation':47,'SDK':str(SDK),'receipt_sha256':RECEIPT_SHA,'plan_sha256':PLAN_SHA,'patched_source_files':72,'all_fresh_ABI_binaries':8,'executable_sha256':receipt['binary_sha256'][str(SDK/'build/strata')],'old43_runtime_proof_transferred':False,'model_qualified':False}
