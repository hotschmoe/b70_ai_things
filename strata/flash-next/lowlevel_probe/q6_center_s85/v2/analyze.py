from pathlib import Path
import json,hashlib,math,re,statistics,sys
p=Path(sys.argv[1]);leaf=p/'leaf';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def need(b,m):
 if not b:raise ValueError(m)
def graph(f,count,arm):
 s=f.read_text();nodes=re.findall(r'^"([^"\n]+)" \[.*?TYPE = CGExecKernel.*?NAME = ([^\\\n]+)\\n',s,re.M);ids={n for n,k in nodes};edges=re.findall(r'"([^"\n]+)" -> "([^"\n]+)"',s)
 need(s.count('TYPE = CGExecKernel')==len(nodes)==len(ids)==count,'DOT nodes');need(len(edges)==len(set(edges))==count-1,'DOT edges')
 pred={n:[]for n in ids};succ={n:[]for n in ids}
 for a,b in edges:need(a in ids and b in ids,'DOT edge binding');succ[a].append(b);pred[b].append(a)
 need(all(len(pred[n])<=1 and len(succ[n])<=1 for n in ids),'DOT branching');roots=[n for n in ids if not pred[n]];need(len(roots)==1,'DOT root');seen=set();n=roots[0]
 while n not in seen:
  seen.add(n)
  if not succ[n]:break
  n=succ[n][0]
 need(seen==ids,'DOT chain');blocks=dict(re.findall(r'^"([^"\n]+)" \[(.*?)\];$',s,re.M));resources=[]
 for n,k in nodes:
  need(('native_mmvq_q6k_wide_a2'in k and'ILi6EE'in k)if arm=='public'else('PrivateC6ILb0EE'if arm=='original'else'PrivateC6ILb1EE')in k,'DOT kernel')
  v={x for x in re.findall(r'Type: Pointer Ptr: [^()\\\n]+\((0x[0-9a-fA-F]+)\)',blocks[n])if int(x,16)};need(len(v)==3,'DOT resources');resources.append(v)
 need(all(v==resources[0]for v in resources),'DOT per-node resources');return {'sha256':sha(f),'nodes':count,'kernel_names':sorted({k for n,k in nodes}),'resources':sorted(resources[0])}
try:
 need(json.loads((leaf/'result.json').read_text())['status']=='PASS','leaf');rows=[json.loads(l)for l in(p/'leaf.log').read_text().splitlines()if l.startswith('{')];identity=[x for x in rows if x['type']=='identity'];need(len(identity)==1 and identity[0]['pci']=='0000:0b:00.0'and identity[0]['public_type']==14,'identity')
 correct=[x for x in rows if x['type']=='correctness_pass'];need(len(correct)==2 and{x['shape']for x in correct}=={'out','qkv'},'correctness');first=next(i for i,x in enumerate(rows)if x['type']=='timing_sample');need(all(rows.index(x)<first for x in correct),'timing before correctness');result={}
 for tag,N,M in [('out',6144,2560),('qkv',2560,10240)]:
  d=leaf/tag;active=6*M*4;qb=6*(N//32)*36;phases=set();hashes={}
  si=[x for x in rows if x['type']=='shape_identity'and x['shape']==tag];need(len(si)==1 and(si[0]['N'],si[0]['M'],si[0]['C'],si[0]['weight_bytes'])==(N,M,6,12902400 if tag=='out'else 21504000),'shape identity')
  for c in 'ab':
   base=(d/f'eager-public-{c}.bin').read_bytes();qbase=(d/f'canonical-{c}-q8.bin').read_bytes();need(len(base)==active and len(qbase)==qb,'sizes')
   stems=[f'{kind}-{arm}-{c}'for kind,arms in [('eager',['public','original','candidate']),('repeat',['original','candidate']),('graph',['public','original','candidate']),('graph-repeat',['public','original','candidate']),('batch20',['original','candidate'])]for arm in arms];phases.update(stems)
   if c=='a':
    stems +=[f'restore-{kind}-{arm}'for kind,arms in [('eager',['public','original','candidate']),('graph',['public','original','candidate']),('batch20',['original','candidate'])]for arm in arms]+['final'];phases.update(stems)
   for st in stems:
    f=d/(st+'.bin');need(f.read_bytes()==base and(d/(st+'-q8.bin')).read_bytes()==qbase,'full output/Q8 '+tag+'/'+st);hashes[st]=sha(f)
   need((d/f'input-{c}.bin').stat().st_size==6*N*4,'input size')
  need((d/'eager-public-a.bin').read_bytes()!=(d/'eager-public-b.bin').read_bytes(),'changed output');need((d/'canonical-a-q8.bin').read_bytes()!=(d/'canonical-b-q8.bin').read_bytes(),'changed Q8')
  gs={f'{kind}-{arm}':graph(d/f'{kind}-{arm}.dot',num,arm)for kind,arms,num in [('single',['public','original','candidate'],1),('batch20',['original','candidate'],20)]for arm in arms};need(all(g['resources']==gs['single-public']['resources']for g in gs.values()),'DOT shared resources')
  phases|={f'warm-{i}-{a}'for i in range(3)for a in ('original','candidate')};phases|={f'timed-r{i}-s{j}-'+(['original','candidate','candidate','original']if i%2==0 else['candidate','original','original','candidate'])[j]for i in range(8)for j in range(4)}
  snaps=[x for x in rows if x['type']=='checked_snapshot'and x['shape']==tag];need(len(snaps)==len(phases)==73 and{x['phase']for x in snaps}==phases,'snapshot phases');need(all(all(x[k]is True for k in ('guards','tail','finite','input_readonly','weight_readonly','Q8_readonly'))for x in snaps),'snapshot guards')
  ts=[x for x in rows if x['type']=='timing_sample'and x['shape']==tag];ps=[x for x in rows if x['type']=='paired_mean'and x['shape']==tag];need(len(ts)==32 and len(ps)==8,'timing counts');pairs=[]
  for i in range(8):
   z=[x for x in ts if x['round']==i];order=['original','candidate','candidate','original']if i%2==0 else['candidate','original','original','candidate'];need([x['slot']for x in z]==list(range(4))and[x['arm']for x in z]==order,'timing order')
   need(all(x['captured_projection_calls']==20 and x['quantization_in_timed_graph']is False and x['checked_full_bytes']is True and math.isfinite(x['enqueue_us'])and x['enqueue_us']>=0 and all(math.isfinite(x[k])and x[k]>0 for k in ('complete_us','barrier_span_us'))for x in z),'timing work')
   v={a:statistics.mean(x['complete_us']/20 for x in z if x['arm']==a)for a in ('original','candidate')};pair=next(x for x in ps if x['round']==i);need(all(abs(pair[a+'_complete_us_per_call']-v[a])<2e-6 for a in v),'mean recomputation');pairs.append(dict(round=i,original_us=v['original'],candidate_us=v['candidate'],change_percent=100*(v['candidate']/v['original']-1)))
  result[tag]={'snapshots':73,'hashes':hashes,'DOTs':gs,'completed_wall_pairs':pairs,'median_pair_change_percent':statistics.median(x['change_percent']for x in pairs),'raw_samples':ts}
 v={'status':'PASS_FULL_BYTES_READONLY_GUARDS_DOT','shapes':result,'limits':'8 balanced round means nested in one process; 20 projection calls/graph, canonical saved Q8 ready outside graphs; completed host wall primary, barrier span includes boundaries; no population, kernel cause or serving claim','serving_qualified':False,'speed_qualified':False}
 with(p/'comparison.json').open('x',encoding='ascii')as f:json.dump(v,f,indent=2);f.write('\n')
 print(json.dumps({'status':v['status'],'timings':{k:z['median_pair_change_percent']for k,z in result.items()}}))
except Exception as e:print('S85 analysis failure: '+str(e),file=sys.stderr);sys.exit(5 if isinstance(e,ValueError)else 2)
