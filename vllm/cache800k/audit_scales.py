#!/usr/bin/env python3
"""Bounded held-out clipping diagnostic; never use this arm for timing claims."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server-root', required=True, type=Path)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    flag = args.server_root / 'AUDIT'
    flag.touch()
    try:
        rc = subprocess.call(['python3', str(Path(__file__).resolve().parents[1] / 'fp8/kv_campaign_probe.py'),
                              '--model', args.model, '--out', str(args.out), '--concurrency', '4'])
    finally:
        flag.unlink(missing_ok=True)
    records = [json.loads(p.read_text()) for p in args.server_root.glob('kv-audit-*.json')]
    if not records:
        raise RuntimeError('no held-out scale observations')
    totals = {key: sum(r[key] for rows in records for r in rows.values())
              for key in ('k_values', 'v_values', 'k_clipped', 'v_clipped')}
    totals['note'] = 'Periodic snapshots, not all final steps. Instrumented correctness diagnostic, not speed.'
    (args.out / 'clipping.json').write_text(json.dumps(totals, indent=2) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
