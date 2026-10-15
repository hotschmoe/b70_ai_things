#!/usr/bin/env python3
"""S80 host closure only; no reference computation, GPU or tolerance policy."""
from pathlib import Path
import json,hashlib,statistics,sys
p=Path(sys.argv[1]);leaf=p/'leaf'
rows=[json.loads(s) for s in (p/'leaf.log').read_text().splitlines() if s.startswith('{')]
assert json.loads((leaf/'result.json').read_text())['status']=='PASS_FULL_GU_TARGET_AND_EAGER_PIPELINE_COHERENCE_ONLY'
a=[x for x in rows if x['type']=='GU_accuracy'];assert len(a)==4
assert {(x['arm'],x['changed']) for x in a}=={(m,c) for m in ('scalar','grouped') for c in (0,1)}
assert all(x['values']==2237440 and x['failures']==0 for x in a)
c=[x for x in rows if x['type']=='correctness_complete'];assert len(c)==1
assert c[0]['GPU_CPU_GU_bytes']==104857600 and c[0]['GPU_down_crossalgorithm_bytes']==52428800
assert c[0]['own_eager_repeat_changed_restore'] and c[0]['same_GU_SiLU_producer'] and not c[0]['ideal_H_down_accuracy_claim']
t=[x for x in rows if x['type']=='timing'];assert len(t)==8
pairs=[]
for i in range(4):
 z=sorted((x for x in t if x['pair']==i),key=lambda x:x['position'])
 assert [x['arm'] for x in z]==(['scalar','grouped'] if i in (0,3) else ['grouped','scalar'])
 assert all(x['metadata_included'] and not x['timed_stage_copies'] and x['own_final_down_full_bytes'] for x in z)
 m={x['arm']:x for x in z};s=m['scalar']['completed_wall_ms'];g=m['grouped']['completed_wall_ms']
 assert 0<s and 0<g
 pairs.append({'pair':i,'scalar_ms':s,'grouped_ms':g,'delta_ms':g-s,'change_percent':100*(g/s-1)})
controls=[x['scalar_ms'] for x in pairs];candidates=[x['grouped_ms'] for x in pairs]
benefit=statistics.median([-x['delta_ms'] for x in pairs]);spread=max(controls)-min(controls)
shapes={'gu.f32':8949760,'h.fp16':2237440,'down.f32':17899520}
files={}
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
for arm in ('scalar','grouped'):
 for inp in ('a','b'):
  for suffix,n in shapes.items():
   f=leaf/f'{arm}-{inp}-{suffix}';assert f.stat().st_size==n
   files[f.name]={'bytes':n,'sha256':sha(f)}
 final=leaf/f'timed-final-{arm}-down.f32';assert final.stat().st_size==17899520
 assert sha(final)==files[f'{arm}-a-down.f32']['sha256']
 files[final.name]={'bytes':final.stat().st_size,'sha256':sha(final)}
for n,z in [('actual-gu-planes.fp16',104857600),('actual-down-planes.fp16',52428800)]:
 f=leaf/n;assert f.stat().st_size==z;files[n]={'bytes':z,'sha256':sha(f)}
assert len(files)==16 and {x.name for x in leaf.iterdir()}==set(files)|{'result.json'}
ident=json.loads((p/'identity.json').read_text())
assert files['actual-gu-planes.fp16']['sha256']==ident['reference_files']['gu-planes.fp16']['sha256']
for suffix,old in [('gu.f32','reference_GU.f32'),('h.fp16','reference_H.fp16'),('down.f32','reference_down.f32')]:
 assert files[f'scalar-a-{suffix}']['sha256']==ident['fixture_files'][old]['sha256']
d={'status':'PASS_FULL_GU_TARGET_OWN_EAGER_PIPELINE_AND_HOST_ARTIFACTS','GU_accuracy':a,'correctness':c[0],
 'pairs':pairs,'scalar_median_ms':statistics.median(controls),'grouped_median_ms':statistics.median(candidates),
 'median_paired_change_percent':statistics.median([x['change_percent'] for x in pairs]),
 'control_range_ms':spread,'median_paired_benefit_ms':benefit,
 'descriptive_prescreen_pass':all(x['delta_ms']<0 for x in pairs) and benefit>spread,
 'raw_timing_samples':t,'artifacts':files,'old_S42_strict_status':'FAILED_UNCHANGED',
 'limits':'Four paired observations nested in one process; full-wave micro only, no ideal H/down accuracy or serving/model claim. No GPU graphs attempted because actual production path is eager.',
 'serving_qualified':False,'speed_qualified':False}
with(p/'comparison.json').open('x',encoding='ascii')as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in d.items() if k not in ('artifacts','raw_timing_samples')},ensure_ascii=True))
