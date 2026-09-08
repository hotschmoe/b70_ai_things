#!/usr/bin/env python3
"""Preserve raw symbol-loop evidence that code sanitization can hide."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect_samples(path):
    failures = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        content = row.get('solution', row.get('completion', ''))
        matches = list(re.finditer(r'([!?])\1{63,}', content))
        if matches:
            failures.append({'task_id': row['task_id'],
                             'sha256': hashlib.sha256(content.encode()).hexdigest(),
                             'runs': [{'offset': m.start(), 'symbol': m[1], 'length': len(m[0])}
                                      for m in matches]})
    return failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = {'samples': str(args.samples), 'symbol_loops': inspect_samples(args.samples)}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
