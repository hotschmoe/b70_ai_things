from pathlib import Path
import hashlib,json,sys
p=Path(sys.argv[1]);rows=[json.loads(x)for x in(p/'leaf.stdout').read_text().splitlines()if x.startswith('{')]
strict=[r for r in rows if r.get('type')=='strict'];times=[r for r in rows if r.get('type')=='timing'];final=[r for r in rows if r.get('status')]
assert len(strict)==36 and len(final)==1 and final[0]['status']=='PASS_PRIVATE_QSA_THREE_STEPS'
assert {(r['start'],r['keepA'],r['keepB'],r['phase'])for r in strict}=={(s,k,4-k,a)for s in[0,2,3,7]for k in[1,2,3]for a in[0,1,2]}
assert all(r['fields']==96 and all(r[x]is True for x in['complete_bytes','guards','readonly','eager_captured'])for r in strict)
assert len(times)==8 and[r['arm']for r in times]==[0,1,1,0,1,0,0,1]
assert all(r['completed_host_chain_us']>0 and r['groups']==24 and r['restores']==24 and r['appends']==(24 if r['arm']else 72)for r in times)
files=sorted((p/'leaf').glob('case*-phase*.bin'));assert len(files)==36
snap=[]
for f in files:
 assert f.stat().st_size==159840
 with f.open('rb')as s:h=hashlib.file_digest(s,'sha256').hexdigest()
 snap.append({'name':f.name,'bytes':f.stat().st_size,'sha256':h})
result={'status':'PASS_PRIVATE_STRICT_ONLY','strict':final[0],'snapshots':snap,'timings':times,'timing_scope':'instrument-free completed HOST captured chain:24 tail D2D restores +72/24 public append calls,2owners*12layers,not isolatedkernel/serving','serving_or_speed_qualified':False}
(p/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
print(json.dumps({'strict_records':len(strict),'timings':len(times),'snapshots':len(files),'status':result['status']}))
