"""CPU image-build check: no existing serving package may change."""
import json
from pathlib import Path
import subprocess

def parse(text):
    return dict(line.split('\t',1) for line in text.splitlines())

before=parse(Path('/opt/d1-base-packages.tsv').read_text())
after=parse(subprocess.check_output(['dpkg-query','-W','-f','${binary:Package}\t${Version}\n'],text=True))
changed={name:dict(before=version,after=after.get(name)) for name,version in before.items() if after.get(name)!=version}
added={name:version for name,version in after.items() if name not in before}
Path('/opt/d1-gdb-package-delta.json').write_text(json.dumps(dict(changed=changed,added=added),indent=2)+'\n')
if changed:
    raise SystemExit('GDB installation changed existing packages; diagnostic image rejected')
print(json.dumps(dict(existing_packages_unchanged=len(before),added=added)))
