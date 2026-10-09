#!/usr/bin/env python3
"""CPU source compilation only; parent owns future leased health controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
HERE=Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=a.output/'control.cpp';source.write_bytes((HERE/'matched_sycl_health_control.cpp').read_bytes())
    command=['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
        '-v',str(a.output.resolve())+':/work',IMAGE,
        'icpx -fsycl -std=c++17 -O2 -fp-model=precise /work/control.cpp -o /work/matched-sycl-health']
    with (a.output/'build.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipt={'schema':1,'image':IMAGE,'devices_exposed':False,'source_sha256':sha(source),'command':command,'build_rc':r.returncode,'passed':r.returncode==0,'gpu_executed':False}
    if receipt['passed']:receipt['binary_sha256']=sha(a.output/'matched-sycl-health')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':receipt['passed'],'receipt':str(a.output/'receipt.json')}))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
