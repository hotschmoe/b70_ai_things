#!/usr/bin/env python3
"""Check native/recipe evidence, then delegate one lease acquisition to server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    pre = json.loads((a.root/'prerequisites.json').read_text())
    plan = json.loads((a.root/'plan.json').read_text())
    for file, digest in pre['files'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest, file
    assert not Path(plan['out']).exists()
    assert os.environ.get('B70_XPU_GDN_PREFIX_CONV_COPY', '0') == '0'
    assert Path(pre['pair_pass']).is_file()
    assert json.loads(Path(pre['pair_inspect']).read_text())[0]['Id'] == pre['image']
    for key in ['numeric_outcome', 'numeric_tp2_local_outcome']:
        value = json.loads(Path(pre[key]).read_text())
        assert value['image'] == pre['image'] and value['passed'] is True and value['numeric_passed'] is True
    assert '--leased' not in plan['server']
    assert plan['server'][plan['server'].index('--tensor-parallel-size')+1] == '2'
    assert plan['server'][plan['server'].index('--p2p')+1] == '0'
    if not a.run:
        print('CPU prerequisites pass; TP1 model review and GPU allocation still required. No GPU launch.')
        return
    # run_arm is unleased; its existing server acquires both leases exactly once.
    os.execv(sys.executable, [sys.executable, *pre['launch'][1:]])


if __name__ == '__main__':
    main()
