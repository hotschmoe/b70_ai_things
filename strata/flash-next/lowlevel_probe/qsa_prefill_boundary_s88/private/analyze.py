#!/usr/bin/env python3
"""Full valid-cell files and exact strict gates, descriptive four paired waves."""
from pathlib import Path
import json,hashlib,statistics,sys
out=Path(sys.argv[1]);leaf=out/'leaf';r=json.loads((leaf/'result.json').read_text());assert r['status']=='PASS' and r['cases']==1 and r['data_files']==8 and r['timed_samples']==8 and r['graphs']==0
rows=[json.loads(x)for x in(out/'leaf.log').read_text().splitlines()if x.startswith('{')]
cases={x['case']:x for x in rows if x['type']=='case'};assert len(cases)==1 and set(cases)=={'near16k'}
assert len([x for x in rows if x['type']=='identity' and x['pci']=='0000:0b:00.0' and x['visible_devices']==1])==1
assert len([x for x in rows if x['type']=='correctness' and x['status']=='PASS'])==1
files={};times={}
def record(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for tag,c in cases.items():
 nq=c['nq'];last=c['last_nkv'];assert nq==256 and last==16384 and last//4+1==4097;assert c['nkv_mod4_mask']==15 and c['stride']==50002 and c['pool_rows']==50001
 ns=sum((last-nq+i+1)//4+1 for i in range(nq))*4;ni=sum(min(last-nq+i+1,2051)for i in range(nq))*4
 for state in ['initial','changed']:
  for suffix,size in [('scores.f32',ns),('ids.i32',ni)]:
   a=leaf/tag/('public-'+state+'.'+suffix);b=leaf/tag/('candidate-'+state+'.'+suffix)
   ra,rb=record(a),record(b);assert ra==rb and ra['bytes']==size
   files[str(a.relative_to(leaf))]=ra;files[str(b.relative_to(leaf))]=rb
 assert files[tag+'/public-initial.scores.f32']['sha256']!=files[tag+'/public-changed.scores.f32']['sha256']
 t=[x for x in rows if x['type']=='timing' and x['case']==tag];assert len(t)==8
 pairs=[]
 for i,order in enumerate(['AB','BA','BA','AB']):
  z=[x for x in t if x['pair']==i];assert len(z)==2 and {x['slot']for x in z}=={0,1}
  z.sort(key=lambda x:x['slot']);assert [x['arm']for x in z]==(['public','candidate']if order=='AB'else['candidate','public'])
  assert all(x['order']==order and x['full_scorer_plus_topk'] and x['full_bytes_checked'] and x['complete_us']>0 for x in z)
  v={x['arm']:x['complete_us']for x in z};pairs.append({'pair':i,**v,'delta_percent':100*(v['candidate']/v['public']-1)})
 times[tag]={'raw_pairs':pairs,'median_paired_delta_percent':statistics.median(x['delta_percent']for x in pairs),'scope':'one process, four descriptive paired scorer+unchanged-topk waves; no significance/serving or isolated-kernel claim'}
assert len(files)==8 and len(list(leaf.rglob('*.f32')))+len(list(leaf.rglob('*.i32')))==8
result={'status':'PASS_PRIMITIVE_ONLY','files':files,'data_bytes':sum(x['bytes']for x in files.values()),'timing':times,'graph_required':False,'serving_qualified':False,'speed_qualified':False}
with(out/'comparison.json').open('x',encoding='ascii')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'data_bytes':result['data_bytes'],'files':8}))
