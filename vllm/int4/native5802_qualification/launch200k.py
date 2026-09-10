"""Read-only frozen/prerequisite gates, then existing leased lifecycle."""
import argparse
import json
from pathlib import Path
import subprocess
from prerequisite import validate,digest


def check(root):
    root=Path(root)
    pre=json.loads((root/'prerequisite.json').read_text())
    validate(pre['receipt'],pre['sha256'])
    if (root/'run').exists():raise ValueError('Output already exists')
    hashes=json.loads((root/'frozen.json').read_text())
    for path,sha in hashes.items():
        if digest(path)!=sha:raise ValueError(('Frozen file changed',path))
    for name in ('plan.json','prerequisite.json'):
        if str((root/name).resolve()) not in hashes:raise ValueError('Missing frozen dependency')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();check(a.root)
    if not a.check_only:
        repo=Path(__file__).resolve().parents[3]
        raise SystemExit(subprocess.call(['python3',str(repo/'vllm/cache800k/run_arm.py'),str(a.root/'plan.json')]))
