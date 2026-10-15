#!/usr/bin/env python3
"""All-or-nothing S91 gates; one process, three seeds, balanced eager timings."""
from pathlib import Path
import json,math,re,statistics,sys
out=Path(sys.argv[1]);result={'status':'FAIL_OR_INCOMPLETE_NO_TIMING_SUMMARY','timing':None}
CASES=[(1,1,64,4,0),(1,63,64,4,0),(1,64,64,4,0),(1,65,128,4,0),(1,65,128,4,1),(1,2051,2112,4,0),(4,2051,2112,4,0),(32,2051,2112,4,0),(32,2051,2112,512,0),(32,1999,2112,4,1),(12,11,64,4,0),(32,0,128,4,0)]
KEYS={(seed,*c)for seed in [7,31,109]for c in CASES}
NUM=r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
PASS=re.compile(r'PASS seed=(\d+) nq=(\d+) width=(\d+) cap=(\d+) page=(\d+) masked=(\d+) scratch=exact output=exact oracle_max=('+NUM+r') base_us=('+NUM+r') candidate_us=('+NUM+r')')
TIMING=re.compile(r'TIMING seed=(\d+) nq=(\d+) width=(\d+) cap=(\d+) page=(\d+) masked=(\d+) round=(\d+) arm=([01]) host_completed_us=('+NUM+r')')
def need(v,msg):
 if not v:raise ValueError(msg)
def stats(x):return {'count':len(x),'median':statistics.median(x),'minimum':min(x),'maximum':max(x)}
try:
 need((out/'leaf.exit').read_text().strip()=='0','leaf failed')
 need((out/'post-health.exit').read_text().strip()=='0','posthealth failed')
 need((out/'post-identity.exit').read_text().strip()=='0','post source/binary identity failed')
 text=(out/'leaf.log').read_text();err=(out/'leaf.stderr').read_text();need('FAIL'not in text and 'FAIL'not in err,'failure marker')
 lines=text.splitlines();need(lines.count('ALL 36 CASES PASS')==1 and lines[-1]=='ALL 36 CASES PASS','final ALL marker missing/duplicated/not last')
 device_rows=[x for x in lines if x.startswith('device=')];need(len(device_rows)==1 and device_rows[0].startswith('device=Intel'),'actual SYCL Intel device-name record')
 passes={};timings={};sequence=[]
 for line in lines:
  if line.startswith('PASS'):
   m=PASS.fullmatch(line);need(m is not None,'malformed PASS');key=tuple(map(int,m.groups()[:6]));need(key in KEYS and key not in passes,'unexpected/duplicate PASS key')
   vals=list(map(float,m.groups()[6:]));need(all(math.isfinite(x)and x>=0 for x in vals),'invalid PASS numeric');need(vals[0]<=2e-5,'existing FP64 sanity failed')
   passes[key]={'oracle_max':vals[0],'leaf_upper_median_base_us':vals[1],'leaf_upper_median_candidate_us':vals[2]};sequence.append(('PASS',key))
  elif line.startswith('TIMING'):
   m=TIMING.fullmatch(line);need(m is not None,'malformed TIMING');key=tuple(map(int,m.groups()[:6]));rnd,arm=map(int,m.groups()[6:8]);us=float(m[9]);need(key in KEYS and rnd in range(20)and math.isfinite(us)and us>0,'invalid timing key/value')
   ident=(key,rnd,arm);need(ident not in timings,'duplicate timing');timings[ident]=us;sequence.append(('TIMING',key,rnd,arm))
  else:need(line.startswith('device=')or line=='ALL 36 CASES PASS','unexpected stdout')
 need(set(passes)==KEYS and len(passes)==36,'require all36 PASS keys')
 need(len(timings)==1440 and set(timings)=={(k,r,a)for k in KEYS for r in range(20)for a in range(2)},'require all1440 raw timings')
 expected=[]
 for seed in [7,31,109]:
  for c in CASES:
   key=(seed,*c)
   for rnd in range(20):
    for slot in range(2):expected.append(('TIMING',key,rnd,slot^(rnd&1)))
   expected.append(('PASS',key))
 need(sequence==expected,'case/pass/order mismatch')
 # All finite cases/records/gates established BEFORE constructing any timing summary.
 per_seed=[];per_shape=[]
 for c in CASES:
  nested=[]
  for seed in [7,31,109]:
   key=(seed,*c);pairs=[];twins=[]
   for rnd in range(20):
    a,b=timings[(key,rnd,0)],timings[(key,rnd,1)];pairs.append({'round':rnd,'order':'AB'if rnd%2==0 else'BA','base_us':a,'candidate_us':b,'delta_percent':100*(b/a-1)})
   for t in range(10):
    z=pairs[2*t:2*t+2];a=statistics.mean(x['base_us']for x in z);b=statistics.mean(x['candidate_us']for x in z);twins.append({'twin':t,'rounds':[2*t,2*t+1],'order':'AB_then_BA','base_mean_us':a,'candidate_mean_us':b,'delta_percent':100*(b/a-1)})
   row={'seed':seed,'nq':c[0],'width':c[1],'cap':c[2],'page':c[3],'masked':c[4],'raw_rounds':pairs,'twinned_rounds':twins,'base_us':stats([x['base_us']for x in pairs]),'candidate_us':stats([x['candidate_us']for x in pairs]),'paired_delta_percent':stats([x['delta_percent']for x in pairs]),'twinned_delta_percent':stats([x['delta_percent']for x in twins]),'AB_delta_percent':stats([x['delta_percent']for x in pairs if x['order']=='AB']),'BA_delta_percent':stats([x['delta_percent']for x in pairs if x['order']=='BA']),'negative_pairs':sum(x['delta_percent']<0 for x in pairs),**passes[key]}
   per_seed.append(row);nested.append(row['paired_delta_percent']['median'])
  per_shape.append({'nq':c[0],'width':c[1],'cap':c[2],'page':c[3],'masked':c[4],'three_seed_median_paired_delta_percent':stats(nested),'scope':'3 seeds nested inside ONE process; not independent process replication'})
 result={'status':'PASS_STRICT_36_CASES_PRIMITIVE_ONLY','SYCL_device_name':device_rows[0][7:],'PASS_cases':36,'ALL_marker':True,'raw_timing_records':1440,'timing':{'per_seed_case':per_seed,'per_shape_nested_seeds':per_shape},'geometry':'INT8/G12/head_dim256/24queryheads/2KVheads/CHUNK64; cap2112/width2051 is33chunk surrogate, not cap2051','limits':['Host completed chunk+merge spans include enqueue/wait, not isolated GPU service.','4 warm and20 measured alternating rounds per arm/case; all36 gates required, no best-only selection.','3 seeds/12cases nested in one process; no population, serving gain, physical bandwidth, or causal claim.'],'graphs':0,'serving_qualified':False,'speed_qualified':False}
except Exception as e:
 result['error']=type(e).__name__+': '+str(e)
finally:
 with(out/'comparison.json').open('x',encoding='ascii')as f:json.dump(result,f,indent=2,ensure_ascii=True);f.write('\n');f.flush()
print(json.dumps({'status':result['status'],'timing_summary_present':result['timing']is not None}))
sys.exit(0 if result['status']=='PASS_STRICT_36_CASES_PRIMITIVE_ONLY'else 2)
