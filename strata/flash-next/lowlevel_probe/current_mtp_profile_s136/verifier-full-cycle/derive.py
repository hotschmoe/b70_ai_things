#!/usr/bin/env python3
"""Only retained S136 5151-event timeline; no raw trace, cross-clock join or idle inference."""
import collections, hashlib, json, pathlib, time
R=pathlib.Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014')
P=R/'current-mtp-profile-s136/run/analysis/gpu/cycles.json'
t0=time.monotonic(); raw=P.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='4783c04c69341d8eb55270443e94a473c0953f215b50171a8b395a45203f8e48'
d=json.loads(raw); c=next(x for x in d['cycles'] if x.get('bounded_GPU_event_timeline'))
e=c['bounded_GPU_event_timeline']; S=c['trace_start_us']; E=c['trace_end_us']-S
assert c['index']==3 and len(e)==5151 and {x['pid'] for x in e}=={d['primary_pid']}
by={str(x['id']):x for x in e}; assert len(by)==len(e)
def start(i): return by[i]['trace_ts_us']-S
def end(i): return start(i)+by[i]['duration_us']
def family(n):
 if n=='zeCommandListAppendEventReset': return 'event_reset'
 if '62080; 1; 1' in n and 'WideQ5K' in n: return 'vocabulary_head'
 if 'native_gu_port' in n: return 'routed_GU'
 if 'native_down_port' in n: return 'routed_down'
 if any(t in n for t in ('gr_up_multi_cols','gr_down_sliced_k','gr_down_reduce_k','gr_norm_split_k')): return 'HC'
 if 'native_mmvq_q6k' in n: return 'Q6_dense'
 if 'WideQ4K' in n: return 'Q4_dense'
 if 'mmvq' in n: return 'other_MMVQ'
 if 'ep_wait' in n: return 'EP_wait'
 if 'gdn_' in n: return 'GDN'
 if 'qsa_' in n or 'Qsa' in n: return 'QSA'
 return 'other'
def calc(a,b):
 buckets=collections.defaultdict(collections.Counter); count=collections.Counter(); service=collections.Counter(); names=collections.defaultdict(lambda:[0,0.0]); events=[]
 for x in e:
  u=max(a,x['trace_ts_us']-S); v=min(b,x['trace_ts_us']-S+x['duration_us'])
  if v<=u: continue
  f=family(x['name']); count[f]+=1; service[f]+=v-u; names[x['name']][0]+=1; names[x['name']][1]+=v-u
  buckets[u][f]+=1; buckets[v][f]-=1; events.append((u,v,x))
 active=collections.Counter(); last=a; union=0.0; exclusive=collections.Counter(); multi=0.0; gaps=[]
 for t in sorted(set(buckets)|{a,b}):
  fs={f for f,n in active.items() if n>0}; dt=t-last
  if dt>0:
   if fs: union+=dt
   else: gaps.append((last,t))
   if len(fs)==1: exclusive[next(iter(fs))]+=dt
   elif len(fs)>1: multi+=dt
  active.update(buckets[t]); last=t
 assert not any(active.values())
 gap_records=[]
 # Only a bounded 8 gaps get adjacent identities; sweep itself is O(n log n).
 for u,v in sorted(gaps,key=lambda z:z[1]-z[0],reverse=True)[:8]:
  before=max((z for z in events if z[1]<=u),key=lambda z:z[1],default=None)
  after=min((z for z in events if z[0]>=v),key=lambda z:z[0],default=None)
  brief=lambda z:None if z is None else {'id':z[2]['id'],'name':z[2]['name'],'start_cycle_us':z[0],'end_cycle_us':z[1]}
  gap_records.append({'start_cycle_us':u,'end_cycle_us':v,'duration_us':v-u,'before':brief(before),'after':brief(after)})
 return {'start_cycle_us':a,'end_cycle_us':b,'span_us':b-a,'overlap_clipped_event_count':len(events),'family_counts':dict(count),'family_service_us':dict(service),'recorded_union_us':union,'uncovered_us':b-a-union,'one_family_only_coverage_us':dict(exclusive),'multiple_family_coverage_us':multi,'top8_uncovered_intervals':gap_records,'top15_operation_rows':[{'name':n,'count':v[0],'service_us':v[1]}for n,v in sorted(names.items(),key=lambda z:-z[1][1])[:15]]}
sel0=max(end('140990'),end('140991')); sel1=max(end('141201'),end('141202'))
segments=[('LEFT_previous_C6_end_to_first_C2',0,start('140985')),('first_C2_head',start('140985'),end('140985')),('first_pair_tail',end('140985'),sel0),('j1_headfree_source_compatible',sel0,start('141196')),('second_C2_head',start('141196'),end('141196')),('second_pair_tail',end('141196'),sel1),('verifier_source_compatible',sel1,start('143765')),('following_C6_head',start('143765'),E)]
out={'status':'BOUNDED_SAVED_TIMELINE_DERIVATION_ONLY','input':{'path':str(P),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'cycle':{'index':3,'primary_pid':d['primary_pid'],'trace_start_us':S,'trace_end_us':c['trace_end_us'],'duration_us':E,'events':len(e),'no_boundary_crossers':all(x['trace_ts_us']>=S and x['trace_ts_us']+x['duration_us']<=c['trace_end_us']for x in e)},'anchors':{i:by[i]for i in ('140985','140990','140991','141196','141201','141202','143765')},'segments':{n:calc(a,b)for n,a,b in segments},'limits':['Source-compatible intervals are not proved owner/tensor bindings.','All service/coverage uses one retained primary-device trace clock; no CPU/peer join.','Uncovered intervals are absent recorded events, not physical idle or host cause.','Float timestamps have approximately 0.25us precision; relative timestamp plus duration avoids large-float addition loss. Last 0.084us head suffix is rounding.','Original whole analyzer was interrupted; this bounded derivation does not retroactively pass global coverage.','Event reset origin/overhead and profiler perturbation are unresolved.','Family service includes overlap; multiple-family coverage is kept separate, never allocated as critical-path savings.']}
assert abs(sum(z['span_us']for z in out['segments'].values())-E)<1e-8
out['elapsed_s']=time.monotonic()-t0
pathlib.Path(__file__).with_name('facts.json').write_text(json.dumps(out,indent=2)+'\n',encoding='ascii')
print(json.dumps({'status':out['status'],'elapsed_s':out['elapsed_s'],'cycle_us':E,'segments':[{'name':n,'span_us':z['span_us'],'union_us':z['recorded_union_us'],'uncovered_us':z['uncovered_us']}for n,z in out['segments'].items()]}))
