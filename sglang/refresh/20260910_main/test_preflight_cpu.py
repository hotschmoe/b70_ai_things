import importlib.util
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
p=pathlib.Path(__file__).resolve().with_name('preflight.py')
spec=importlib.util.spec_from_file_location('preflight',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
results=[]
with tempfile.TemporaryDirectory() as td:
 root=pathlib.Path(td);m.ROOT=root;m.DOCKER='/mock/docker'
 # Exact owner label is required before any removal.
 commands=[]
 replies=[NS(returncode=0,stdout='abc\n',stderr=''),
          NS(returncode=0,stdout='[{"Config":{"Labels":{"b70.preflight.owner":"mine"}},"Name":"/owned","Image":"sha256:test"}]',stderr=''),
          NS(returncode=0,stdout='',stderr=''), NS(returncode=0,stdout='',stderr='')]
 def cap(args,timeout=30):
  commands.append(args);return replies.pop(0)
 m.capture=cap
 assert m.cleanup_owned('mine',root/'clean.json')
 assert ['rm','-f','abc']==commands[2][1:]
 results.append('owned removal then absence verification')
 replies[:]=[NS(returncode=0,stdout='abc\n',stderr=''),NS(returncode=0,stdout='[{"Config":{"Labels":{"b70.preflight.owner":"other"}}}]',stderr='')]
 commands.clear()
 assert not m.cleanup_owned('mine',root/'wrong-owner.json')
 assert not any('rm' in c for c in commands)
 results.append('ownership mismatch forbids deletion')
 # Timeout is CPU-only; simulated cleanup cannot invoke Docker or GPU code.
 cleaned=[]
 m.cleanup_owned=lambda owner,out: cleaned.append(owner) or True
 rc=m.run([sys.executable,'-c','import time; time.sleep(60)'],root/'timeout.log',0.05,owned=True)
 assert rc==124 and len(cleaned)==1
 results.append('outer timeout tears down launcher before cleanup')
 # A transient cleanup failure is not allowed to return until absence is proved.
 answers=iter([False,False,True]);checks=[]
 m.cleanup_owned=lambda owner,out: checks.append(1) or next(answers)
 m.time.sleep=lambda seconds: None
 rc=m.run([sys.executable,'-c','pass'],root/'blocked.log',10,owned=True)
 assert rc==1 and len(checks)==3 and (root/'CLEANUP_RESOLVED').exists()
 assert not m.CLEANUP_PENDING
 results.append('cleanup uncertainty retains control until verified and marks failure')
print('\n'.join('PASS '+x for x in results))
