#!/usr/bin/env python3
"""HOST ONLY: all four ABBA numerical/DOT gates precede timing acceptance."""
from pathlib import Path
import hashlib,json,math,re,statistics,sys
P=Path(__file__).resolve().parent
root=Path(sys.argv[1]); size=747776
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={f'T{t}-a{a}-i{i}-{phase}.bin' for t in [1,2,6] for a in [0,1] for i in [0,1] for phase in ['direct-a','graph-a','direct-b','graph-b']}
expected|={f'T{t}-a1-i1-{phase}.bin' for t in [1,2,6] for phase in ['batch20-direct-a','batch20-graph-a','batch20-direct-b','batch20-graph-b','timed-r0','timed-r1','timed-r2']}
assert len(expected)==69
anchors=json.loads((P/'s39-anchors.json').read_text());assert len(anchors)==36

def dot(path,mode,calls):
 s=path.read_text();nodes=re.findall(r'^"([^"\n]+)" \[.*?TYPE = CGExecKernel.*?NAME = ([^\\\n]+)\\n',s,re.M)
 edges=re.findall(r'"([^"\n]+)" -> "([^"\n]+)"',s)
 assert len(nodes)==s.count('TYPE = CGExecKernel')==4*calls,(path,'kernel count')
 names=dict(nodes);assert len(names)==len(nodes);pred={x:[] for x in names};succ={x:[] for x in names}
 assert len(edges)==len(set(edges))==len(nodes)-1,(path,'edge count')
 for a,b in edges:assert a in names and b in names;succ[a].append(b);pred[b].append(a)
 assert all(len(pred[x])<=1 and len(succ[x])<=1 for x in names)
 heads=[x for x in names if not pred[x]];assert len(heads)==1
 order=[];n=heads[0]
 while n not in order:
  order.append(n)
  if not succ[n]:break
  n=succ[n][0]
 assert len(order)==len(nodes),(path,'complete chain')
 for j,n in enumerate(order):
  name=names[n];role=j%4
  if role==0:assert 'gr_norm_split_k' in name
  elif mode==0:
   wanted=['','gr_down_sliced_k','gr_down_reduce_k','gr_up_multi_cols'][role]
   assert wanted in name and 'grf256' not in name and '_more' not in name,(path,name)
   if role==3:assert 'ILi16EE' in name
  else:
   wanted=['','gr_down_sliced_more','gr_down_reduce_more','gr_up_multi_more'][role]
   assert wanted in name and 'dpct_kernel_scalarILi0EE' in name,(path,name)
 return {'sha256':sha(path),'kernels':len(nodes),'edges':len(edges),'directed_chain':True,'classes_in_order':list(dict.fromkeys(names[n] for n in order)),
         'binding_scope':'Captured kernel classes/order only. STD_Layout addresses are not decoded resource pointers; DOT has no actual NDRange geometry or runtime dispatch proof.'}

processes=[];data={}
for label,mode in [('a1',0),('b1',1),('b2',1),('a2',0)]:
 d=root/label;events=[json.loads(x) for x in (d/'oracle.log').read_text().splitlines() if x.startswith('{')]
 end=[x for x in events if x.get('status')=='PASS_ONE_PROCESS_CROSS_MORE_PENDING'];assert len(end)==1
 assert end[0]=={'status':'PASS_ONE_PROCESS_CROSS_MORE_PENDING','more':mode,'iters':20,'cases':12,'timing_rounds':9,'device_pci':'0000:44:00.0'}
 snapshots=[x for x in events if x.get('type')=='snapshot'];assert len(snapshots)==78
 assert all(x['bytes']==size and x['guards'] and x['finites'] and x['tails'] and x['inputs_readonly'] and x['weights_readonly'] for x in snapshots)
 done=[x for x in events if x.get('type')=='correctness_complete'];assert len(done)==1 and done[0]['cases']==12 and done[0]['own_batch20_cases']==3 and done[0]['timing_not_started']
 timings=[x for x in events if x.get('type')=='timing'];assert len(timings)==9
 assert events.index(done[0])<min(events.index(x) for x in timings)
 keys=[(x['T'],x['round']) for x in timings];assert len(set(keys))==9 and set(keys)=={(t,r) for t in [1,2,6] for r in range(3)}
 for x in timings:
  assert x['apply']==x['inject']==x['graph_submissions']==1 and x['calls']==20
  assert all(math.isfinite(x[k]) and x[k]>0 for k in ['enqueue_us','complete_us'])
  assert x['device_barrier_span_us'] is None or math.isfinite(x['device_barrier_span_us']) and x['device_barrier_span_us']>=0
 files=d/'snapshots';assert {f.name for f in files.glob('*.bin')}==expected
 hashes={}
 for n in sorted(expected):
  f=files/n;assert f.stat().st_size==size;hashes[n]=sha(f)
  if n in anchors:assert hashes[n]==anchors[n],(label,n,'frozen S39 anchor mismatch')
 data[label]=hashes
 dots={}
 for t in [1,2,6]:
  for a in [0,1]:
   for i in [0,1]:
    n=f'T{t}-a{a}-i{i}-single.dot';dots[n]=dot(files/n,mode,1)
  n=f'T{t}-a1-i1-batch20.dot';dots[n]=dot(files/n,mode,20)
 assert len(list(files.glob('*.dot')))==len(dots)==15
 # Every timed reset starts from the same canonical state and executes20 calls.
 for t in [1,2,6]:
  base=hashes[f'T{t}-a1-i1-batch20-direct-a.bin']
  assert all(hashes[f'T{t}-a1-i1-{n}.bin']==base for n in ['batch20-graph-a','timed-r0','timed-r1','timed-r2'])
 processes.append({'label':label,'mode':mode,'summary':end[0],'snapshots':hashes,'DOT':dots,'timings_raw':timings})
assert all(data[k]==data['a1'] for k in ['b1','b2','a2']),'cross-process full-byte mismatch; timing invalid'
# Descriptive only AFTER all four whole-state and capture-class gates.
summary={}
for t in [1,2,6]:
 vals={p['label']:[x['complete_us']/20 for x in p['timings_raw'] if x['T']==t] for p in processes}
 med={k:statistics.median(v) for k,v in vals.items()}
 a=statistics.median([med['a1'],med['a2']]);b=statistics.median([med['b1'],med['b2']])
 summary[str(t)]={'host_completed_us_per_read_rounds':vals,'process_medians_us':med,'pooled_nested_median_us':[a,b],'candidate_change_percent':100*(b/a-1),
  'limits':'Two fresh processes per mode, three nested graph-batch samples each; not six independent trials, per-call tail latency, device service or serving proof.'}
result={'status':'PASS_ALL_FOUR_PROCESS_NUMERICAL_CAPTURE_GATES','snapshot_pairs':69,'snapshot_bytes_each':size,'s39_anchors':36,'processes':processes,'timings':summary,'speed_qualified':False,'serving_qualified':False,
 'geometry_provenance':'Source launcher: norm4T, down80->480, reduceceil(324T/128), UP160->320, local256 except reduce128; compiler metadata/source binding not actual runtime trace.',
 'lifecycle_gate':'Root still must validate pinned binary/source/archive/header identity, exact env/ABBA commands, physicalcard1, pre/post health, createdCID+NAME removal and wrapper exits.'}
with (root/'comparison.json').open('x',encoding='ascii') as f:json.dump(result,f,indent=2);f.write('\n')
print('PASS all69 snapshots across4processes; 36S39 anchors and15DOT/process; timing descriptive only')
