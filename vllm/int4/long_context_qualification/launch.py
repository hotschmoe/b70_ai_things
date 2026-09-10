#!/usr/bin/env python3
"""Read-only prerequisites before delegating to the existing leased arm runner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
REPO=Path(__file__).resolve().parents[3]

def validate(root):
    pre=json.loads((root/'prerequisite.json').read_text());run=Path(pre['run'])
    assert (run/'WORKLOADS_PASSED').is_file() and Path(pre['lifecycle']).read_text().strip()=='0','100K predecessor not qualified'
    assert not (root/'run').exists(),'refuse output reuse'
    manifest=json.loads((root/'manifest.json').read_text())
    for rel,digest in manifest['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
    for path,digest in manifest['external'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args();validate(a.directory)
    if not a.check_only:raise SystemExit(subprocess.call(['python3',str(REPO/'vllm/cache800k/run_arm.py'),str(a.directory/'plan.json')]))
