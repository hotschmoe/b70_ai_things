#!/usr/bin/env python3
"""Print a prepared feature plan; explicit --run checks prerequisites then execs."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def validate(plan):
    assert Path(plan['pair_preflight_pass']).is_file()
    numerical = json.loads(Path(plan['read_policy_numeric_outcome']).read_text())
    assert numerical['passed'] is True and numerical['all_nine_retained'] is True
    assert numerical['image'] == plan['image']
    for root in plan['prior_viability_outputs']:
        root = Path(root)
        assert (root/'exit.rc').read_text().strip() == '0'
        assert (root/'job.rc').read_text().strip() == '0'
        result = json.loads((root/'viability/OUTCOME.json').read_text())
        assert result['passed'] is True and result['requests'] == 26
    for path,digest in plan['frozen_files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    assert not Path(plan['output']).exists(), 'fresh output/cache required'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan',type=Path)
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.run:
        print(json.dumps(plan,indent=2))
        return 0
    validate(plan)
    os.execvp(plan['command'][0],plan['command'])


if __name__ == '__main__':
    raise SystemExit(main())
