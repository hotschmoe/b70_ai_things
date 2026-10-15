#!/usr/bin/env python3
"""Full-byte/DOT closure before reporting S65 descriptive paired timings."""
from pathlib import Path
import hashlib,json,math,re,statistics,sys
out=Path(sys.argv[1]);leaf=out/'leaf'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def graph(f,count,arm):
 s=f.read_text()
 nodes=re.findall(r'^"([^"\n]+)" \[.*?TYPE = CGExecKernel.*?NAME = ([^\\\n]+)\\n',s,re.M)
 edges=re.findall(r'"([^"\n]+)" -> "([^"\n]+)"',s)
 need(s.count('TYPE = CGExecKernel')==len(nodes)==count,f'{f} node count')
 ids={n for n,k in nodes};need(len(ids)==count,f'{f} unique nodes')
 need(len(edges)==count-1 and len(set(edges))==len(edges),f'{f} edges')
 if count>1:
  pred={n:[] for n in ids};succ={n:[] for n in ids}
  for a,b in edges:need(a in ids and b in ids,f'{f} edge binding');succ[a].append(b);pred[b].append(a)
  need(all(len(pred[n])<=1 and len(succ[n])<=1 for n in ids),f'{f} branching')
  roots=[n for n in ids if not pred[n]];need(len(roots)==1,f'{f} root')
  seen=set();n=roots[0]
  while n not in seen:
   seen.add(n)
   if not succ[n]:break
   n=succ[n][0]
  need(seen==ids,f'{f} whole dependency chain')
 for _,name in nodes:
  need('native_quantize_q8_1' not in name,f'{f} unexpected quantization')
  if arm=='public':need('mmvq_multi_kernel_3404b2' in name and 'ILi21EE' in name and 'ILi6EE' in name,f'{f} public IQ3S/C6 binding')
  else:need(('PrivateC6ILb0EE' if arm=='original' else 'PrivateC6ILb1EE') in name,f'{f} private arm binding')
 # Compiler capture order can vary, but these graphs must reference the same
 # three actual USM resources (weights, canonical Q8, output) for this shape.
 pointers=re.findall(r'Type: Pointer Ptr: [^()\\\n]+\((0x[0-9a-fA-F]+)\)',s)
 need(len(pointers)==count*3,f'{f} pointer count')
 resource=set(pointers[:3]);need(len(resource)==3,f'{f} distinct resources')
 for i in range(count):need(set(pointers[3*i:3*i+3])==resource,f'{f} per-node resources')
 return {'sha256':sha(f),'nodes':count,'edges':len(edges),'whole_chain':True,'kernel_names':sorted({k for n,k in nodes}),'resources':sorted(resource)}
try:
 r=json.loads((leaf/'result.json').read_text());need(r['status']=='PASS' and r['shapes']==2 and r['public_original_bit_anchor'] is True and r['candidate_full_bytes'] is True,'leaf result')
 rows=[json.loads(l) for l in(out/'leaf.log').read_text().splitlines() if l.startswith('{')]
 ids=[x for x in rows if x['type']=='identity'];need(len(ids)==1 and ids[0]['pci']=='0000:0b:00.0' and ids[0]['visible_devices']==1,'card identity')
 shapeids=[x for x in rows if x['type']=='shape_identity'];need(len(shapeids)==2,'shape identities')
 correct=[x for x in rows if x['type']=='correctness_pass'];need({x['shape'] for x in correct}=={'gate','out'} and len(correct)==2,'correctness closures')
 firsttime=next(i for i,x in enumerate(rows) if x['type']=='timing_sample')
 need(all(rows.index(x)<firsttime for x in correct),'all shapes correct before timing')
 evidence={};summaries={}
 for tag,N,M in [('gate',2560,6144),('out',6144,2560)]:
  d=leaf/tag;active=6*M*4;qbytes=6*(N//32)*36;hashes={}
  ident=next(x for x in shapeids if x['shape']==tag)
  need((ident['N'],ident['M'],ident['C'],ident['weight_bytes'])==(N,M,6,6758400),tag+' shape')
  for changed in ['a','b']:
   base=(d/f'eager-public-{changed}.bin').read_bytes();qbase=(d/f'canonical-{changed}-q8.bin').read_bytes()
   need(len(base)==active and len(qbase)==qbytes,tag+' sizes')
   names=[f'eager-public-{changed}']+[f'{kind}-{arm}-{changed}' for kind,arms in [('eager',['original','candidate']),('repeat',['original','candidate']),('graph',['public','original','candidate']),('batch20',['original','candidate'])] for arm in arms]
   for stem in names:
    f=d/(stem+'.bin');need(f.read_bytes()==base,f'{tag}/{stem} full bytes');need((d/(stem+'-q8.bin')).read_bytes()==qbase,f'{tag}/{stem} Q8')
    hashes[stem]=sha(f)
   need((d/f'input-{changed}.bin').stat().st_size==6*N*4,tag+' input size')
  need((d/'eager-public-a.bin').read_bytes()!=(d/'eager-public-b.bin').read_bytes(),tag+' changed output')
  need((d/'canonical-a-q8.bin').read_bytes()!=(d/'canonical-b-q8.bin').read_bytes(),tag+' changed Q8')
  graphs={}
  for kind,arms,count in [('single',['public','original','candidate'],1),('batch20',['original','candidate'],20)]:
   for arm in arms:graphs[kind+'-'+arm]=graph(d/(kind+'-'+arm+'.dot'),count,arm)
  resources=graphs['single-public']['resources']
  need(all(g['resources']==resources for g in graphs.values()),tag+' public/private actual resource binding')
  phases={f'{kind}-{arm}-{c}' for c in ['a','b'] for kind,arms in [('eager',['public','original','candidate']),('repeat',['original','candidate']),('graph',['public','original','candidate']),('batch20',['original','candidate'])] for arm in arms}
  phases|={f'warm-{i}-{arm}' for i in range(3) for arm in ['original','candidate']}
  phases|={f'timed-r{i}-s{j}-'+(['original','candidate','candidate','original'] if i%2==0 else ['candidate','original','original','candidate'])[j] for i in range(8) for j in range(4)}
  phases.add('final')
  snaps=[x for x in rows if x['type']=='checked_snapshot' and x['shape']==tag]
  need(len(snaps)==len(phases)==59 and {x['phase'] for x in snaps}==phases,tag+' all warm/timed/own snapshots')
  need(all(all(x.get(k) is True for k in ['guards','tail','finite','input_readonly','weight_readonly','Q8_readonly']) for x in snaps),tag+' structural snapshots')
  timing=[x for x in rows if x['type']=='timing_sample' and x['shape']==tag];pairs=[x for x in rows if x['type']=='paired_mean' and x['shape']==tag]
  need(len(timing)==32 and len(pairs)==8,tag+' timing count')
  descriptive=[]
  for i in range(8):
   rr=[x for x in timing if x['round']==i];need([x['slot'] for x in rr]==[0,1,2,3],tag+' slots')
   expected=['original','candidate','candidate','original'] if i%2==0 else ['candidate','original','original','candidate']
   need([x['arm'] for x in rr]==expected,tag+' balanced order')
   for x in rr:
    need(x['captured_projection_calls']==20 and x['quantization_in_timed_graph'] is False and x['checked_full_bytes'] is True,tag+' timed work')
    need(all(math.isfinite(x[k]) and x[k]>0 for k in ['complete_us','barrier_span_us']) and math.isfinite(x['enqueue_us']) and x['enqueue_us']>=0,tag+' finite timing')
   actual={arm:statistics.mean(x['barrier_span_us']/20 for x in rr if x['arm']==arm) for arm in ['original','candidate']}
   pair=next(x for x in pairs if x['round']==i)
   for arm in actual:need(abs(pair[arm+'_span_us_per_call']-actual[arm])<2e-6,tag+' pair recomputation')
   descriptive.append({'round':i,'order':'OCCO' if i%2==0 else 'COOC','original_span_us_per_call':actual['original'],'candidate_span_us_per_call':actual['candidate'],'candidate_change_percent':100*(actual['candidate']/actual['original']-1)})
  summaries[tag]={'paired_means':descriptive,'median_pair_change_percent':statistics.median(x['candidate_change_percent'] for x in descriptive),'limits':'8 nested balanced round means in one process, not independent process samples or significance; projection only, canonical Q8 already ready; not serving'}
  evidence[tag]={'identity':ident,'output_hashes':hashes,'DOT_inventory':graphs,'snapshots':snaps,'raw_timing_samples':timing,'reported_paired_means':pairs}
 v={'status':'PASS_ALL_BYTE_GUARD_IMMUTABLE_DOT_BINDING_GATES','identity':ids[0],'evidence':evidence,'timings':summaries,'speed_qualified':False,'serving_qualified':False}
 with(out/'comparison.json').open('x',encoding='ascii') as f:json.dump(v,f,indent=2);f.write('\n')
 print(json.dumps({'status':v['status'],'timings':summaries}));sys.exit(0)
except Exception as e:
 print('S65 analysis failure: '+str(e),file=sys.stderr);sys.exit(5 if isinstance(e,ValueError) else 2)
