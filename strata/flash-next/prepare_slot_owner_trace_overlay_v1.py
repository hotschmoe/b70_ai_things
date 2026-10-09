#!/usr/bin/env python3
"""CPU-only immutable base/patch reconstruction; no model/GPU/SDK invocation."""
import argparse,hashlib,json,shutil,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=json.loads((HERE/'slot-owner-trace-source-draft-v1.json').read_bytes());base=a.base or Path(r['base']);out=a.output or Path(tempfile.mkdtemp(prefix='strata-slot29-reconstructed-',dir='/mnt/vm_8tb/b70/build'))
 if a.output:out.mkdir(parents=True,exist_ok=False)
 for n,v in r['base_source_sha256'].items():assert sha(base/n)==v;p=out/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
 patch=ROOT/r['patch'];assert sha(patch)==r['patch_sha256'];subprocess.run(['git','apply','--check',str(patch)],cwd=out,check=True);subprocess.run(['git','apply',str(patch)],cwd=out,check=True)
 assert {n:sha(out/n) for n in r['expected_source_sha256']}==r['expected_source_sha256'];print(out)
if __name__=='__main__':main()
