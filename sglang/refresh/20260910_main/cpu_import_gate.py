#!/usr/bin/env python3
"""Check installed native imports without initializing an XPU device."""
import ctypes
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess

assert not Path('/dev/dri').exists(), 'CPU-only check requires hidden devices'
import torch
import sgl_kernel
import sglang
import torch_memory_saver

assert metadata.version('sglang-kernel-xpu') == '0.2.0'
assert metadata.version('sglang') == '0.5.19.dev0+g2f7393f0d'
assert hasattr(torch.ops.sgl_kernel, 'gdn_attention')
assert torch._C._dispatch_has_kernel_for_dispatch_key('sgl_kernel::gdn_attention', 'XPU')
for name in ['aten::_convert_weight_to_int4pack', 'aten::_weight_int4pack_mm_with_scales_and_zeros']:
    assert torch._C._dispatch_has_kernel_for_dispatch_key(name, 'XPU'), name
try:
    metadata.distribution('sgl-kernel')
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError('Old sgl-kernel distribution remains installed')
if os.environ.get('B70_BUILD_FMHA') == 'OFF':
    assert not hasattr(torch.ops.sgl_kernel, 'fwd')
    importlib.import_module('sgl_kernel.flash_attn')
rust_modules = [
    'sglang.srt.rust_extensions._grpc',
    'sglang.srt.rust_extensions._multimodal',
    'sglang.srt.rust_extensions._server',
    'sglang.srt.mem_cache.rust_tree_core.mem_cache',
]
rust_imports = {name: importlib.import_module(name).__file__ for name in rust_modules}
root = Path(sgl_kernel.__file__).parent
native = {}
for dist_name in ['sglang-kernel-xpu', 'torch_memory_saver', 'sglang']:
    dist = metadata.distribution(dist_name)
    for rel in dist.files or []:
        path = Path(dist.locate_file(rel))
        if path.suffix == '.so' and path.is_file():
            with path.open('rb') as handle:
                native[str(path)] = hashlib.file_digest(handle, 'sha256').hexdigest()
memory_saver_native = [p for p in native if 'torch_memory_saver' in p]
assert len(memory_saver_native) == 1
memory_saver_library = ctypes.CDLL(memory_saver_native[0])
assert getattr(memory_saver_library, 'tms_get_current_tag')
loaded_kernel_native = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/sgl_kernel/' in line and '.so' in line})
assert set(loaded_kernel_native).issubset(set(native)), 'Loaded untracked kernel native library'
recorded_kernel_native = {Path(p).resolve() for p in native if Path(p).parent.resolve() == root.resolve()}
assert {p.resolve() for p in root.glob('*.so')} == recorded_kernel_native, 'Untracked native payload remains in sgl_kernel'
result = {'imports': {m.__name__: m.__file__ for m in [torch, sgl_kernel, sglang, torch_memory_saver]},
          'native_hashes': native, 'native_count': len(native), 'rust_imports': rust_imports,
          'loaded_kernel_native': loaded_kernel_native,
          'fmha_built': hasattr(torch.ops.sgl_kernel, 'fwd'),
          'device_exposed': False}
Path('/out/cpu-import-gate.json').write_text(json.dumps(result, indent=2) + '\n')
check = subprocess.run(['python3', '-m', 'pip', 'check'], capture_output=True, text=True)
Path('/out/pip-check.txt').write_text(check.stdout + check.stderr)
Path('/out/pip-check.rc').write_text(str(check.returncode) + '\n')
print('CPU native package import/API gate PASS; dependency report requires review.')
