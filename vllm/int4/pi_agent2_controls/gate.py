#!/usr/bin/env python3
"""Require exact resolved geometry and preserve all Pi semantic flags."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_replay_quality import audit

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--startup',action='store_true');a=p.parse_args()
 result={}
 try:
  sizes=[int(n) for n in re.findall(r'Setting attention block size to (\d+) tokens', (a.root/'server.log').read_text())]
  assert sizes and set(sizes)=={1600},'resolved block must be1600'
  result['resolved_blocks']=sizes
  if not a.startup:
   reviewed=audit(a.root/'replay');result['review']=reviewed
   assert reviewed['requests']==3
   assert all(not r['bang'] and not r['error'] and not r['flags'] for r in reviewed['rows']),'Pi quality failure'
   targets=[r for r in reviewed['rows'] if '-target-' in r['name']]
   assert len(targets)==2 and all(r['usage']['prompt_tokens_details']['cached_tokens']>0 for r in targets),'missing target cache reuse'
  result['passed']=True
 except Exception as exc:result.update(passed=False,error=ascii(exc))
 (a.root/('geometry-gate.json' if a.startup else 'strict-pi-gate.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
