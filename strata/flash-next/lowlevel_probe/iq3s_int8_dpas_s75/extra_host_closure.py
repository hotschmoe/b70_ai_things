from pathlib import Path
import hashlib,json,sys
E=Path(__file__).resolve().parent
P=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq3s-int8-dpas-s75-prep-v2')
mode,out=sys.argv[1],Path(sys.argv[2])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pins=json.loads((E/'harness-pins.json').read_text())
actual={n:sha(Path(n)) for n in pins}
assert actual==pins,'source/host-harness hash mismatch'
report={'status':'PASS_HOST_HARNESS_PINS','mode':mode,'pins':actual,'GPU_work':False}
if mode=='post':
 assert json.loads((out/'extra-host-pre.json').read_text())['pins']==actual,'pre/post host-harness hashes differ'
 comparison=json.loads((out/'comparison.json').read_text())
 assert comparison['status']=='PASS_ALL_BYTE_GUARD_IMMUTABLE_DOT_BINDING_GATES'
 results={}
 for tag in ['gate','out']:
  d=out/'leaf'/tag
  for final,reference in [('final.bin','eager-public-a.bin'),('final-q8.bin','canonical-a-q8.bin')]:
   a,b=(d/final).read_bytes(),(d/reference).read_bytes()
   assert a==b,(tag,final,'full final saved bytes differ')
   results[tag+'/'+final]={'bytes':len(a),'sha256':hashlib.sha256(a).hexdigest(),'equal_to':reference}
 files=sorted((out/'leaf').glob('*/*.bin'))
 assert len(files)==92,('unexpected binary artifact count',len(files))
 report.update(status='PASS_FINAL_SAVED_BYTES_AND_HOST_HARNESS_PINS',final_artifacts=results,all_binary_files=len(files),all_binary_bytes=sum(f.stat().st_size for f in files))
else:assert mode=='pre'
with(out/('extra-host-'+mode+'.json')).open('x',encoding='ascii') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'mode':mode}))
