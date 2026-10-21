"""Reject apt simulation before any existing runtime package can change."""
from pathlib import Path
import re

before=dict(line.split('\t',1) for line in Path('/opt/d1-base-packages.tsv').read_text().splitlines())
existing={name.split(':')[0] for name in before}
plan=Path('/opt/d1-gdb-apt-plan.txt').read_text()
for line in plan.splitlines():
    if line.startswith('Remv '):
        raise SystemExit('apt proposes package removal: '+line)
    if line.startswith('Inst '):
        name=line.split()[1].split(':')[0]
        if name in existing or re.search(r'^Inst \S+ \[',line):
            raise SystemExit('apt proposes existing package change: '+line)
summary=re.search(r'(\d+) upgraded, (\d+) newly installed, (\d+) to remove',plan)
if not summary or summary.group(1)!='0' or summary.group(3)!='0':
    raise SystemExit('apt plan lacks zero-upgrade/zero-removal proof')
print('APT_PLAN_PASS: no existing package is changed')
