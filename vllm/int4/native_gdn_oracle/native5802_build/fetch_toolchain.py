#!/usr/bin/env python3
"""Download/extract pinned compiler packages only; never install on the host."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
packages = json.loads(Path(__file__).with_name('toolchain-packages.json').read_text())
(a.out / 'debs').mkdir(parents=True, exist_ok=True)
(a.out / 'root').mkdir(exist_ok=True)
for name, package in packages.items():
    target = a.out / 'debs' / Path(package['Filename']).name
    if not target.exists():
        urllib.request.urlretrieve('https://apt.repos.intel.com/oneapi/' + package['Filename'], target)
    with target.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == package['SHA256'], name
    subprocess.run(['dpkg-deb', '-x', str(target), str(a.out / 'root')], check=True)
print('Pinned packages SHA256 verified and extracted; no host installation.')
