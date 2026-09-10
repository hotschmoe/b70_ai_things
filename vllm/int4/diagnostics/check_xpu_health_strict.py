"""CPU-only finite-result and owned-container regression for the strict probe."""
import json, os, pathlib, subprocess, sys, tempfile
ROOT=pathlib.Path(__file__).resolve().parent
candidate=ROOT/'xpu_health_strict.sh'
source=candidate.read_text(); probe=source.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
rows=[]
with tempfile.TemporaryDirectory() as tmp:
 d=pathlib.Path(tmp)
 (d/'torch.py').write_text('''import os, math
class Value:
 def __init__(self, finite): self.finite=finite
 def matmul(self, other): return self
 def all(self): return self
 def item(self): return self.finite
class Xpu:
 def device_count(self): return 1
 def synchronize(self): pass
xpu=Xpu()
def randn(a,b,device): return Value(float(os.environ.get('BAD_VALUE','nan')) if os.environ.get('BAD_SHAPE') == str(a) else 1.0)
def isfinite(t): return Value(math.isfinite(t.finite))
''')
 for bad,value,expected in [('', 'nan',0),('2048','nan',1),('2048','inf',1),('16','nan',1),('16','inf',1)]:
  r=subprocess.run([sys.executable,'-c',probe],env=dict(os.environ,PYTHONPATH=tmp,BAD_SHAPE=bad,BAD_VALUE=value),capture_output=True,text=True)
  assert r.returncode==expected,(bad,r)
  assert ('HEALTH_OK True' in r.stdout)==(expected==0)
  rows.append(dict(case='fake-torch-'+(bad or 'finite')+'-'+value,rc=r.returncode,stdout=r.stdout.strip()))
 docker=d/'docker'
 docker.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys, time
p=pathlib.Path(os.environ['MOCK_STATE']); a=sys.argv[1:]
with open(os.environ['MOCK_TRACE'],'a') as f: f.write(json.dumps(a)+'\\n')
if a[0]=='run':
 assert '--name' in a and '--label' in a
 p.write_text(a[a.index('--name')+1])
 if os.environ.get('MOCK_HANG')=='1': time.sleep(60)
 print(os.environ.get('MOCK_SENTINEL','HEALTH_OK True'))
 raise SystemExit(int(os.environ.get('MOCK_RC','0')))
if a[0]=='ps':
 if p.exists(): print('owned-id')
 raise SystemExit(0)
if a[:2]==['rm','-f']:
 assert a[2:]==['owned-id']
 p.unlink(missing_ok=True)
 raise SystemExit(0)
raise SystemExit(99)
''')
 docker.chmod(0o755)
 for label,sentinel,status,hang,expected in [('true','HEALTH_OK True',0,False,0),('false','HEALTH_OK False',0,False,1),('nonzero','HEALTH_OK True',7,False,1),('conflicting','HEALTH_OK True\nHEALTH_FAIL bad',0,False,1),('true-plus-false','HEALTH_OK True\nHEALTH_OK False',0,False,1),('infra','',125,False,2),('timeout','',0,True,1)]:
  trace=d/(label+'.trace');state=d/(label+'.state')
  env=dict(os.environ,PATH=tmp+os.pathsep+os.environ['PATH'],MOCK_STATE=str(state),MOCK_TRACE=str(trace),MOCK_SENTINEL=sentinel,MOCK_RC=str(status),MOCK_HANG=str(int(hang)))
  r=subprocess.run(['bash',str(candidate),'--card','1','--img','mock','--timeout','0.1' if hang else '2'],env=env,capture_output=True,text=True,timeout=20)
  assert r.returncode==expected,(label,r.returncode,r.stderr)
  assert not state.exists(),label
  events=[json.loads(x) for x in trace.read_text().splitlines()]
  assert any(x[:2]==['rm','-f'] for x in events)
  assert events[-1][0]=='ps'
  rows.append(dict(case='mock-docker-'+label,rc=r.returncode,verified_removed=True))
# Raw investigation retains the original detailed CPU result artifact.
print('PASS 5 fake-torch finite checks and 7 mock-Docker sentinel/cleanup cases; no GPU use')
