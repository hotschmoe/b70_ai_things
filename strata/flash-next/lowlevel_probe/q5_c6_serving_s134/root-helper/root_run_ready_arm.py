#!/usr/bin/env python3
"""Root-only finite S134 phases; always attempt normal owned cleanup once."""
from pathlib import Path
import argparse,json,subprocess,time
R=Path(__file__).resolve().parent.parent
REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
p=argparse.ArgumentParser();p.add_argument('stage',choices=['134']);p.add_argument('arm',choices=['0','1']);a=p.parse_args()
assert a.stage=='134'
D=R/'q5-head-c6-runtime-ones-s134-serving-prep'
out=D/('mode'+a.arm);assert(out/'container-ready.json').is_file()
result=out/'root-phases.json';assert not result.exists(),'No phase retry'
records=[];stop=None;exc=None
try:
 for phase in ['warm','timed','functional']:
  cmd=['bin/cpu-run']+(['--quiet']if phase in ['warm','timed']else[])+['python3',str(D/'harness-prep/client.py'),a.arm,phase]
  t=time.monotonic()
  with(out/('root-'+phase+'.log')).open('x')as f:
   r=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
  records.append({'phase':phase,'command':cmd,'exit':r.returncode,'elapsed_s':time.monotonic()-t})
  result.write_text(json.dumps({'phases':records,'cleanup':'PENDING'},indent=2)+'\n',encoding='ascii')
  print(json.dumps(records[-1]),flush=True)
  if r.returncode:break
except Exception as e:exc=repr(e)
finally:
 cmd=['bin/cpu-run','python3',str(D/'harness-prep/stop_and_health.py'),a.arm]
 t=time.monotonic()
 with(out/'root-stop-health.log').open('x')as f:
  r=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
 stop={'command':cmd,'exit':r.returncode,'elapsed_s':time.monotonic()-t}
 final={'phases':records,'exception':exc,'cleanup':stop,'no_retry':True}
 result.write_text(json.dumps(final,indent=2)+'\n',encoding='ascii')
 print(json.dumps({'cleanup':stop,'exception':exc}),flush=True)
raise SystemExit(0 if exc is None and len(records)==3 and all(x['exit']==0 for x in records)and stop['exit']==0 else 2)
