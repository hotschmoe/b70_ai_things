#!/usr/bin/env python3
"""Run a frozen cache campaign arm; the existing server owns GPU leases."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('plan', type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    out = Path(plan['out'])
    if out.exists():
        raise RuntimeError('refusing to overwrite an arm')
    args.plan.with_suffix('.pid').write_text(str(os.getpid()) + '\n')
    log = args.plan.with_suffix('.log').open('w')
    server = subprocess.Popen(plan['server'], stdout=log, stderr=subprocess.STDOUT)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
        if out.exists():
            (out / 'STOP').touch()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    results = {}
    try:
        deadline = time.monotonic() + 1500
        while not (out / 'READY').exists():
            if stopped or server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('startup did not complete')
            time.sleep(1)
        (out / 'arm-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        for job in plan['jobs']:
            if stopped:
                raise RuntimeError('arm interrupted')
            name = job['name']
            dest = out / 'jobs' / (name + '.json')
            temp = dest.with_suffix('.tmp')
            temp.write_text(json.dumps({k: v for k, v in job.items() if k != 'name'}) + '\n')
            temp.replace(dest)
            deadline = time.monotonic() + job.get('timeout', 1800) + 60
            while not dest.with_suffix('.done').exists():
                if stopped or server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('job did not complete: ' + name)
                time.sleep(1)
            rc = int(dest.with_suffix('.done').read_text())
            results[name] = rc
            (out / 'arm-results.json').write_text(json.dumps(results, indent=2) + '\n')
            if rc:
                raise RuntimeError('failed job: ' + name + ' rc=' + str(rc))
        (out / 'WORKLOADS_PASSED').touch()
    finally:
        stop()
        rc = server.wait()
        log.close()
        args.plan.with_suffix('.lifecycle-rc').write_text(str(rc) + '\n')
    if rc:
        raise RuntimeError('lifecycle failed: ' + str(rc))


if __name__ == '__main__':
    main()
