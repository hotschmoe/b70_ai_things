"""Parse actual finite S116 coverage/order before summarizing any timing."""
from pathlib import Path
import json,math,re,statistics,sys
P=Path(__file__).resolve().parent;R=P.parent;O=Path(sys.argv[1]);S=R/'hc-producer-s116-oracle-prep'
lines=(O/'native.log').read_text().splitlines();config=[x for x in lines if x.startswith('CONFIG ')]
assert config==['CONFIG S116_PRIVATE sixHCclasses normalC16 sliced=1 MORE=0 ILP=0 GRF=default nsplit=1 pci=0000:44:00.0']
events=[]
for i,line in enumerate(lines):
 if line.startswith('{'):events.append((i,json.loads(line)))
allowed={'S39_anchor','strict_case','synthetic_reduce_actual_raw_part','synthetic_candidate_preUP_lo','DIRECT_CONTROL_UP_ONLY_raw_input','DIRECT_CONTROL_UP_ONLY_lo_NOT_candidate_proof','strict_complete_before_timing','timing'}
assert all(x.get('type')in allowed for _,x in events)
expected_cases=[(t,a,j)for t in[1,2,6]for a in[0,1]for j in[0,1]];cases=[(i,x)for i,x in events if x['type']=='strict_case']
assert len(cases)==12
assert[(x['T'],x['apply'],x['inject'])for _,x in cases]==expected_cases
for _,x in cases:assert x=={'type':'strict_case','T':x['T'],'apply':x['apply'],'inject':x['inject'],'states':'A-B-A','eager_and_captured':True,'raw_parts_final_bytes_guards_readonly':True,'PASS':True}
anchors=[(i,x)for i,x in events if x['type']=='S39_anchor'];anchor_spec=json.loads((S/'s39-anchors.json').read_text())
assert len(anchors)==len({x['name']for _,x in anchors})==36 and {x['name']for _,x in anchors}==set(anchor_spec['anchors'])
for _,x in anchors:assert x=={'type':'S39_anchor','name':x['name'],'bytes':747776,'PASS':True}
previous=-1
for(case_line,_),(t,a,j)in zip(cases,expected_cases):
 stem=f'T{t}-a{a}-i{j}';want=[stem+'-direct-a.bin']if t==1 else[stem+'-'+suffix+'.bin'for suffix in['direct-a','graph-a','direct-b','graph-b']]
 assert[x['name']for i,x in anchors if previous<i<case_line]==want
 previous=case_line
bitlabels=['synthetic_reduce_actual_raw_part','synthetic_candidate_preUP_lo','DIRECT_CONTROL_UP_ONLY_raw_input','DIRECT_CONTROL_UP_ONLY_lo_NOT_candidate_proof'];bits={k:[(i,x)for i,x in events if x['type']==k]for k in bitlabels}
assert[len(bits[k])for k in bitlabels]==[3,3,1,1]
for group in bits.values():
 for _,x in group:assert set(x)=={'type','bits'}and len(x['bits'])==14 and all(re.fullmatch('[0-9a-f]{8}',v)for v in x['bits'])
for label in bitlabels[:2]:assert bits[label][0][1]['bits']==bits[label][2][1]['bits'],'printed restored diagnostic bits'
assert bits[bitlabels[0]][0][1]['bits']!=bits[bitlabels[0]][1][1]['bits'],'printed changed raw partial bits'
assert bits[bitlabels[2]][0][1]['bits']==['00000000','80000000','00000001','80000001','007fffff','807fffff','00800000','80800000','3f800000','bf800000','7f800000','ff800000','7fc12345','ffc54321']
completes=[(i,x)for i,x in events if x['type']=='strict_complete_before_timing'];assert len(completes)==1;strict_line,complete=completes[0]
assert complete=={'type':'strict_complete_before_timing','base_cases':12,'ordinary_eager_route_calls':72,'ordinary_captured_route_calls':72,'historical_unique_anchors':36,'synthetic_route_calls':12,'direct_control_only_probes':1,'HC_counters':'NONE_NO_RESET_LOGIC','PASS':True}
assert cases[-1][0]<min(i for i,_ in bits[bitlabels[0]])and all(i<strict_line for k in bitlabels for i,_ in bits[k])and max(i for i,_ in anchors)<strict_line
assert[(x['type'])for i,x in events if cases[-1][0]<i<strict_line]==[q for _ in range(3)for q in bitlabels[:2]]+bitlabels[2:]
timings=[(i,x)for i,x in events if x['type']=='timing'];assert len(timings)==8 and[(x['round'],x['arm'])for _,x in timings]==[(r,a)for r in[0,1]for a in[0,1,1,0]]
assert all(i>strict_line for i,_ in timings)and timings[-1][0]==events[-1][0]
for _,x in timings:assert set(x)=={'type','round','arm','whole_reads','graph_submissions','completed_host_us'}and x['whole_reads']==x['graph_submissions']==1 and isinstance(x['completed_host_us'],(float,int))and math.isfinite(x['completed_host_us'])and x['completed_host_us']>0
verdict=json.loads((O/'oracle/verdict.json').read_text());assert verdict=={'status':'PASS_PRIVATE_STRICT_FIXTURES_ONLY','base_cases':12,'anchors':36,'timing':True,'production_unchanged':True}
rounds=[]
for r in[0,1]:
 means={str(a):statistics.mean(x['completed_host_us']for _,x in timings if x['round']==r and x['arm']==a)for a in[0,1]};rounds.append({'round':r,'arm_means_us':means,'candidate_change_pct':100*(means['1']/means['0']-1)})
result={'status':'PASS_EXACT_S116_STRICT_COVERAGE_THEN_8_TIMINGS_PRIVATE_ONLY','actual_config':config[0],'base_case_order':expected_cases,'counts':complete,'historical_anchor_names':[x['name']for _,x in anchors],'printed_diagnostic_bits':{k:[x for _,x in v]for k,v in bits.items()},'raw_timing_rows':[x for _,x in timings],'two_nested_ABBA_round_means':rounds,'arm_median_four_completed_host_us':{str(a):statistics.median(x['completed_host_us']for _,x in timings if x['arm']==a)for a in[0,1]},'scope':'One process on physical card1, seeded private HC fixture, two nested ABBA rounds; completed full four-kernel graph read including host submission/wait. Not isolated UP GPU service, device cycles, population replication or serving qualification. Direct control-only bits are not candidate cross-context proof.','speed_qualified':False,'serving_qualified':False}
(O/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii');print(json.dumps(result))
