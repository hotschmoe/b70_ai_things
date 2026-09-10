"""Freeze prepared200K inputs only after a completed reviewed100K receipt."""
import argparse
import json
from pathlib import Path
from prerequisite import validate,digest


def freeze(root,receipt,receipt_sha):
    root=Path(root)
    validate(receipt,receipt_sha)
    if any((root/n).exists() for n in ('plan.json','run','frozen.json','prerequisite.json')):
        raise ValueError('Refuse existing frozen plan/output')
    template=json.loads((root/'template-inputs.json').read_text())
    for p,sha in template.items():
        if digest(p)!=sha:raise ValueError(('Template changed',p))
    plan=json.loads((root/'plan.template.json').read_text())
    plan['notes']=plan['notes'].replace('TEMPLATE_NOT_RUN','FROZEN_NOT_RUN')
    (root/'prerequisite.json').write_text(json.dumps({'receipt':str(Path(receipt).resolve()),'sha256':receipt_sha},indent=2)+'\n')
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    files=dict(template)
    for p in (root/'plan.json',root/'prerequisite.json',Path(receipt)):
        files[str(p.resolve())]=digest(p)
    (root/'frozen.json').write_text(json.dumps(files,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--receipt',required=True);p.add_argument('--receipt-sha256',required=True)
    a=p.parse_args();freeze(a.root,a.receipt,a.receipt_sha256)
