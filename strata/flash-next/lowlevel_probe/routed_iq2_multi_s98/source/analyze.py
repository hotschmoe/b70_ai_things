#!/usr/bin/env python3
"""Small exact inventory/byte gate; future run only, never opens a model."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,sys
p=Path(sys.argv[1]);leaf=p/'leaf'
records=[json.loads(l)for l in(p/'leaf.log').read_text().splitlines()if l.startswith('{')]
c=Counter(x.get('type')for x in records)
assert c=={'weight_identity':2,'route':3,'checked_snapshot':88,'correctness_pass':3,'timing_sample':16},dict(c)
assert json.loads((leaf/'result.json').read_text())['status']=='PASS_PRIMITIVE_ONLY'
expected={}
for layout in ['active','singleton','zero']:
 for b in ['a','b']:
  files=[leaf/layout/f'eager-{arm}-{b}.bin'for arm in['public','1','2']]
  data=[f.read_bytes()for f in files];assert all(len(x)==163840 for x in data);assert data[0]==data[1]==data[2]
  for f,v in zip(files,data):expected[str(f.relative_to(leaf))]={'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}
 if layout!='zero':assert (leaf/layout/'eager-public-a.bin').read_bytes()!=(leaf/layout/'eager-public-b.bin').read_bytes()
 else:assert (leaf/layout/'eager-public-a.bin').read_bytes()==(leaf/layout/'eager-public-b.bin').read_bytes()
for b in ['a','b']:
 for stem,size in [(f'input-{b}',61440),(f'canonical-{b}-q8',17792)]:
  f=leaf/f'{stem}.bin';v=f.read_bytes();assert len(v)==size;expected[f.name]={'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}
assert {str(f.relative_to(leaf))for f in leaf.rglob('*.bin')}==set(expected)
dots={str(f.relative_to(leaf))for f in leaf.rglob('*.dot')}
assert dots=={f'{layout}/{kind}-{arm}.dot'for layout in ['active','singleton','zero']for kind in (['single']if layout=='zero'else['single','batch20'])for arm in ['original','multi']}
samples=[x for x in records if x.get('type')=='timing_sample']
assert {(x['layout'],x['round'],x['position'],x['arm'])for x in samples}=={(layout,r,pos,2-pos if r%2 else 1+pos)for layout in [0,2]for r in range(4)for pos in range(2)}
for x in samples:
 assert x['calls']==20 and x['own_full_bytes']is True
 for k in ['enqueue_us','completed_us','barrier_span_us']:assert math.isfinite(x[k])and x[k]>=0
out={'status':'PASS_PRIMITIVE_ONLY','binary_files':expected,'DOT_files':sorted(dots),'timing_samples':samples,'speed_qualified':False,'serving_qualified':False,'limits':['Warm repeated two-expert GU microfixture, not full EP/serving or streaming-cache proof.','Barrier spans are interval observations, not isolated device service.','DOT captures kernel metadata; actual binding must pass the independent pre-run host/device audit.']}
with(p/'analysis.json').open('x',encoding='ascii')as f:json.dump(out,f,indent=2);f.write('\n')
