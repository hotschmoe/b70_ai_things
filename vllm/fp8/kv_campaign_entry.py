#!/usr/bin/env python3
"""API server entry for an explicit, inherited calibration hook."""
import sys
import hashlib
import json
from pathlib import Path
import sysconfig
import vllm

package = Path(vllm.__file__).resolve().parent
expected = Path(sysconfig.get_path('purelib')) / 'vllm'
if package != expected:
    raise RuntimeError('campaign must load the installed vLLM package: ' + str(package))
print('B70_PACKAGE_IDENTITY ' + json.dumps({'package': str(package),
      'xpu_ops_sha256': hashlib.sha256((package / '_xpu_ops.py').read_bytes()).hexdigest()}), flush=True)
from vllm.entrypoints.cli.main import main

if __name__ == '__main__':
    sys.argv.insert(1, 'serve')
    main()
