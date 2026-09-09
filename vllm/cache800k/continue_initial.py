#!/usr/bin/env python3
"""Bounded continuation of the September 9 initial calibration/backend gates."""
import argparse
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]


def decision(root):
    if (root / 'exit.rc').read_text().strip() != '0':
        raise RuntimeError('previous lifecycle unhealthy; stop for review')
    if (root / 'WORKLOADS_PASSED').exists():
        return 'freeze'
    results = json.loads((root / 'arm-results.json').read_text())
    if results == {'01-quality': 0, '01b-guides': 124}:
        return 'retry_timeout'
    raise RuntimeError('previous correctness gate failed; stop for review')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    status = root / 'continuation-status.json'
    def report(stage, **kw):
        status.write_text(json.dumps(dict(stage=stage, **kw), indent=2) + '\n')
        print(stage, flush=True)
    def run_plan(path):
        subprocess.run(['python3', str(REPO / 'vllm/cache800k/run_arm.py'), str(path)], check=True)
    try:
        report('waiting-for-f0')
        cal = root / 'f0-calibration'
        deadline = time.monotonic() + 8 * 3600
        while not (cal / 'exit.rc').exists():
            if time.monotonic() > deadline or (root / 'STOP_CONTINUATION').exists():
                raise RuntimeError('continuation wait stopped or expired')
            time.sleep(5)
        action = decision(cal)
        if (root / 'STOP_CONTINUATION').exists():
            raise RuntimeError('continuation stopped')
        if action == 'retry_timeout':
            report('retrying-incomplete-guide-with-larger-timeout')
            run_plan(root / 'f0b-calibration.plan.json')
            cal = root / 'f0b-calibration'
        report('freezing-scales')
        scales = root / 'int4-mtp0-kv-scales.json'
        subprocess.run(['python3', str(REPO / 'vllm/cache800k/freeze_scales.py'),
                        '--calibration', str(cal), '--preservation', str(root / 'preservation'),
                        '--out', str(scales)], check=True)
        for name in ('f1-fp8', 's0-sglang-fp16'):
            if (root / 'STOP_CONTINUATION').exists():
                raise RuntimeError('continuation stopped')
            report('running-' + name)
            run_plan(root / (name + '.plan.json'))
        report('initial-gates-complete-review-required', promoted=False)
    except Exception as exc:
        report('stopped-for-review', error=ascii(exc), promoted=False)
        raise


if __name__ == '__main__':
    main()
