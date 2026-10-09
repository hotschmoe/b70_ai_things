#!/usr/bin/env python3
"""Reconstruct five source files from immutable base + reviewable patch, CPU only."""
import argparse,hashlib,json,shutil,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();plan=json.loads((HERE/'solo-migration-observer-source-draft-v1.json').read_bytes());base=a.base or Path(plan['base_source']);out=a.output or Path(tempfile.mkdtemp(prefix='strata-observer26-reconstructed-',dir='/mnt/vm_8tb/b70/build'))
 if a.output:out.mkdir(parents=True,exist_ok=False)
 for name,want in plan['base_source_sha256'].items():
  assert sha(base/name)==want,'Frozen base differs '+name;target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/name,target)
 patch=ROOT/plan['patch'];assert sha(patch)==plan['patch_sha256'];subprocess.run(['git','apply','--check',str(patch)],cwd=out,check=True);subprocess.run(['git','apply',str(patch)],cwd=out,check=True)
 actual={n:sha(out/n) for n in plan['expected_source_sha256']};assert actual==plan['expected_source_sha256']
 receipt={'schema':1,'passed':True,'mode':'CPU_SOURCE_RECONSTRUCTION_ONLY','patch_sha256':plan['patch_sha256'],'base':str(base),'output':str(out),'expected_source_sha256':actual,'SDK_or_GPU_or_model_execution':False};(out/'reconstruction.json').write_text(json.dumps(receipt,indent=2)+'\n');print(out)
if __name__=='__main__':main()
