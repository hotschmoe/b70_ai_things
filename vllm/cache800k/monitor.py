#!/usr/bin/env python3
"""One non-overlapping scheduled Codex campaign review; no direct GPU work."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/mnt/vm_8tb/b70/results/cache800k_20260909')


def main():
    out = ROOT / 'monitor'
    out.mkdir(exist_ok=True)
    with (out / 'review.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        report = out / (stamp + '.md')
        status = out / 'status.json'
        status.write_text(json.dumps(dict(started=stamp, state='running')) + '\n')
        with (REPO / 'vllm/cache800k/monitor_prompt.md').open() as prompt, \
                (out / (stamp + '.log')).open('w') as log:
            rc = subprocess.run([
                str(Path.home() / '.local/bin/codex'), 'exec',
                '-C', str(REPO), '-s', 'danger-full-access',
                '-c', 'approval_policy="never"', '--color', 'never',
                '-o', str(report), '-'], stdin=prompt, stdout=log,
                stderr=subprocess.STDOUT).returncode
        if report.exists() and rc == 0:
            temp = out / 'latest.tmp'
            temp.write_bytes(report.read_bytes())
            temp.replace(out / 'latest.md')
        status.write_text(json.dumps(dict(started=stamp, state='complete' if rc == 0 else 'failed',
                                         rc=rc, report=str(report))) + '\n')
        return rc


if __name__ == '__main__':
    raise SystemExit(main())
