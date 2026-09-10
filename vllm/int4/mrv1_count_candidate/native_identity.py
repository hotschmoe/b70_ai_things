"""CPU-only native identity; no imports of Torch/SGLang, no device queries."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

files={}
for root in (Path('/opt/venv/lib'),Path('/usr/lib/x86_64-linux-gnu'),Path('/opt/intel')):
    if not root.exists():continue
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        try:
            with path.open('rb') as stream:
                magic=stream.read(8)
                if not (magic.startswith(b'\x7fELF') or magic==b'!<arch>\n'):continue
                stream.seek(0)
                files[str(path)]={'bytes':path.stat().st_size,
                                  'sha256':hashlib.file_digest(stream,'sha256').hexdigest()}
        except PermissionError:
            raise RuntimeError('native identity unreadable: '+str(path))
assert files,'empty native identity'
packages={}
for name in ('torch','triton-xpu','sglang','sglang-kernel-xpu','torch-memory-saver',
             'intel-sycl-rt','intel-ccl-lib-rt'):
    try:packages[name]=metadata.version(name)
    except metadata.PackageNotFoundError:packages[name]=None
print(json.dumps(dict(packages=packages,native_files=files),indent=2,sort_keys=True))
