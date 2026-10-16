from pathlib import Path
import argparse,json,math,re,statistics
p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();out=a.out
lines=(out/'native.log').read_text().splitlines()
configs=[x for x in lines if x.startswith('CONFIG S112_PRIVATE ')]
assert len(configs)==1 and 'kernel=GreedyTop1S112 leader_acquire=1 device='in configs[0]
assert 'Intel'in configs[0]and'B70'in configs[0],'Actual leaf device is not the expected Intel B70 name'
results=[(i,re.fullmatch(r'RESULT n=(\d+) eager_and_capture=PASS owner_routes=4 cases=(\d+)',line))for i,line in enumerate(lines)if line.startswith('RESULT ')]
assert len(results)==2 and all(m for i,m in results)
assert [(int(m[1]),int(m[2]))for i,m in results]==[(4097,24),(248320,48)]
verdict=[x for x in lines if x.startswith('VERDICT ')]
assert verdict==['VERDICT PASS_STRICT_PRIVATE_ID_DOWNSTREAM_BYTES_GUARDS_COUNTERS_CAPTURE cases=48 speed_qualified=0']
rows=[]
for i,line in enumerate(lines):
 if not line.startswith('TIMING '):continue
 m=re.fullmatch(r'TIMING n=(\d+) repeat=(\d+) route=(reference|candidate) two_owner_chain_host_us=([0-9.]+)',line);assert m,line
 assert i>results[-1][0],'Timing occurred before all48 strict cases'
 n,repeat,route,us=int(m[1]),int(m[2]),m[3],float(m[4]);assert n==248320 and math.isfinite(us)and us>0
 rows.append({'n':n,'repeat':repeat,'route':route,'two_owner_chain_host_us':us})
assert len(rows)==8 and [(r['repeat'],r['route'])for r in rows]==[(j,k)for j in [0,1]for k in ['reference','candidate','candidate','reference']]
rounds=[]
for j in [0,1]:
 r={k:statistics.mean(x['two_owner_chain_host_us']for x in rows if x['repeat']==j and x['route']==k)for k in ['reference','candidate']};rounds.append({'repeat':j,'means_us':r,'candidate_change_pct':100*(r['candidate']/r['reference']-1)})
result={'status':'PASS_ALL48_STRICT_AND_FINITE_TIMING_DIAGNOSTIC_ONLY','actual_config':configs[0],'strict_cases':48,'timing_shape':248320,'timing4097':False,'raw_timing_rows':rows,'two_nested_ABBA_round_means':rounds,'scope':'One process, two ABBA rounds; host completed two-owner greedy/probability/select chain. Not isolated argmax GPU service, population, early mapped publication or serving qualification.','speed_qualified':False}
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii');print(json.dumps(result))
