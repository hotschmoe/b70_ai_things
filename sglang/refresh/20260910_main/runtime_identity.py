#!/usr/bin/env python3
"""CPU-only identity collection; never queries XPU runtime devices."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

site = Path('/opt/venv/lib/python3.12/site-packages')
packages = {}
for name in ['torch', 'triton-xpu', 'intel-sycl-rt', 'intel-ccl-lib-rt']:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
files = [site / 'torch/lib/libtorch_xpu.so', site / 'torch/lib/libc10_xpu.so',
         Path('/opt/venv/lib/libccl.so.1.0')]
files += list(Path('/usr/lib/x86_64-linux-gnu').glob('libze_intel_gpu.so.*'))
digests = {}
for f in files:
    if f.is_file():
        with f.open('rb') as handle:
            digests[str(f)] = hashlib.file_digest(handle, 'sha256').hexdigest()
result = dict(held_packages=packages, held_files=digests,
              driver_packages=subprocess.check_output(
                  ['dpkg-query', '-W', 'libze-intel-gpu1', 'libze1'], text=True),
              kernel=subprocess.check_output(['uname', '-r'], text=True).strip())
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
