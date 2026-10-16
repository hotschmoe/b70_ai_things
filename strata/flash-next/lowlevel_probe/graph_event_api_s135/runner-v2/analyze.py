"""Tiny exact-output/DOT oracle only. No trace parser, timings or actual API count claims."""
from pathlib import Path
import hashlib,json,sys
out=Path(sys.argv[1]);mode=int(sys.argv[2])if len(sys.argv)==3 else None
if mode is None:
 results=[json.loads((out/f'mode{m}/closure.json').read_text())for m in [0,1]]
 assert all(r['status']=='PASS_ONE_MODE_TWO_QUEUE_ORDERING_ONLY'for r in results)
 for phase in range(3):assert (out/f'mode0/leaf/phase-{phase}.bin').read_bytes()==(out/f'mode1/leaf/phase-{phase}.bin').read_bytes()
 v={'status':'PASS_TWO_QUEUE_GRAPH_BARRIER_ORDERING_ONLY','both_modes_exact':True,'fresh_processes':2,'same_4_kernel_operations_per_iteration':True,'A_B_A':True,'all_8_iteration_intermediates_checked_per_phase':True,'actual_LevelZero_counts_measured':False,'overlap_speed_MTP_qualified':False}
 (out/'comparison.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii');print(json.dumps(v));sys.exit(0)
assert mode in[0,1];p=out/f'mode{mode}';leaf=p/'leaf';r=json.loads((leaf/'result.json').read_text());assert r['status']=='PASS_TWO_QUEUE_GRAPH_BARRIER_ORDERING_ONLY'and r['mode']==mode
for k,v in {'graphs':3,'graph_kernel_nodes':[1,1,2],'phases':3,'depth_per_phase':8,'graph_submits_source':72,'kernel_operations_source':96,'barrier_submits_source':45 if mode else 0,'dependency_host_waits_source':0 if mode else 72,'common_setup_fills':6,'common_full_readbacks':3,'bytes_per_readback':32100}.items():assert r[k]==v,(k,r[k])
assert r['same_context_device_inorder']and r['mapped_A_B_A']and not r['actual_LevelZero_counts_measured']and not r['overlap_measured']and not r['speed_qualified']and not r['MTP_numeric_qualified']
rows=[json.loads(s)for s in(p/'leaf.log').read_text().splitlines()if s.startswith('{')];assert len(rows)==3 and[r['phase']for r in rows]==[0,1,2]
assert all(row['strict']and row['readonly']and row['guards']and row['iterations_checked']==8 and row['intermediates_full_bytes']==32100 for row in rows)
for name,count,classes in [('pre-a.dot',1,['PreA']),('pre-b.dot',1,['PreB']),('tail.dot',2,['TailData','TailAdvance'])]:
 s=(leaf/name).read_text();assert s.count('TYPE = CGExecKernel')==count and all(t in s for t in classes),(name,'unexpected capture topology')
b=[(leaf/f'phase-{i}.bin').read_bytes()for i in range(3)];assert all(len(x)==32100 for x in b)and b[0]==b[2]and b[0]!=b[1]
v={'status':'PASS_ONE_MODE_TWO_QUEUE_ORDERING_ONLY','mode':mode,'leaf':r,'snapshots':[{'phase':i,'bytes':len(x),'sha256':hashlib.sha256(x).hexdigest()}for i,x in enumerate(b)],'capture_kernel_nodes_observed':[1,1,2],'runtime_submission_counts':'not measured; source application counts only','no_speed_claim':True}
(p/'closure.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii');print(json.dumps(v))
