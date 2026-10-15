import hashlib,json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'run-clean'
closure=json.loads((R/'closure.json').read_text())
assert closure['oracle_exit']==closure['pre_health_exit']==closure['post_health_exit']==0
assert closure['container_removed'] and closure['snapshot_file_count']==78 and closure['external_profiler']=='none'
events=[json.loads(l) for l in (R/'oracle.log').read_text().splitlines() if l.startswith('{')]
assert events[-1]['status']=='PASS_S48_STRICT_PROBE_ONLY' and events[-1]['strict_snapshot_pairs']==36
snapshots=[e for e in events if e.get('type')=='snapshot'];timings=[e for e in events if e.get('type')=='paired_warm_timing']
assert len(snapshots)==78 and len(timings)==6
old=P.parent/'grf-oracle-s39-prep/run-profiled'
recorded={x['file']:x for x in json.loads((old/'comparison.json').read_text())['snapshots']}
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
anchors=[];pairs=[]
for e in snapshots:
 assert e['guards'] and e['finites'] and e['tails']
 if e['phase'].startswith('warm-') or e['variant']!=0:continue
 stem=f"T{e['T']}-a{e['apply']}-i{e['inject']}"
 baseline=R/'snapshots'/f"{stem}-mode0-{e['phase']}.bin"
 candidate=R/'snapshots'/f"{stem}-mode1-{e['phase']}.bin"
 legacy=old/'mode0'/f"{stem}-{e['phase']}.bin"
 a=baseline.read_bytes();assert a==candidate.read_bytes() and a==legacy.read_bytes()
 assert len(a)==e['bytes']==recorded[legacy.name]['bytes']
 h=hashlib.sha256(a).hexdigest();assert h==recorded[legacy.name]['sha256']
 anchors.append({'phase':legacy.name,'bytes':len(a),'sha256':h})
assert len(anchors)==36
for i in range(3):
 rows=sorted([e for e in timings if e['round']==i],key=lambda e:e['position'])
 assert [e['variant'] for e in rows]==([1,0] if i==1 else [0,1])
 assert all(e['calls']==32 and e['warmup_calls_per_variant']==8 and not e['begin_waited_before_host_t0'] for e in rows)
 a,b=[next(e for e in rows if e['variant']==v) for v in (0,1)]
 assert all((e['T'],e['apply'],e['inject'])==(6,1,1) for e in rows)
 assert (R/f'snapshots/T6-a1-i1-mode0-warm-r{i}.bin').read_bytes()==(R/f'snapshots/T6-a1-i1-mode1-warm-r{i}.bin').read_bytes()
 pairs.append(100*(b['complete_us']/a['complete_us']-1))
metrics={}
for v in (0,1):
 rows=sorted([e for e in timings if e['variant']==v],key=lambda e:e['round'])
 metrics[str(v)]={}
 for key in ('enqueue_us','after_enqueue_to_drained_us','complete_us','same_queue_barrier_span_us'):
  values=[e[key]/e['calls'] for e in rows]
  metrics[str(v)][key]={'batch_mean_us_per_call':values,'median':statistics.median(values),'min':min(values),'max':max(values)}
a=metrics['0']['complete_us'];b=metrics['1']['complete_us']
delta=b['median']-a['median'];pct=100*delta/a['median']
spread=max(a['max']-a['min'],b['max']-b['min'])
same_sign=all(x<0 for x in pairs) or all(x>0 for x in pairs)
signal=same_sign and abs(delta)>spread and abs(pct)>=2
report={'stage':'S48','status':'STRICT_MICROPROBE_AND_S39_BASELINE_ANCHOR_PASS','numerical':{'cases':12,'strict_snapshot_pairs':36,'strict_pair_bytes':sum(x['bytes'] for x in anchors),'S39_recorded_sha_anchors':anchors,'timed_final_pairs':3,'snapshot_files':78},'metrics':metrics,'paired_complete_percent_change':pairs,'ratio_of_complete_medians_percent_change':pct,'max_same_variant_complete_spread_us_per_call':spread,'declared_screen':{'all3_same_sign':same_sign,'median_difference_exceeds_max_spread':abs(delta)>spread,'absolute_change_at_least2percent':abs(pct)>=2,'passes_signal':signal},'host_submit_fraction_of_eventspan':{v:metrics[v]['enqueue_us']['median']/metrics[v]['same_queue_barrier_span_us']['median'] for v in ('0','1')},'limits':['Synthetic BF16 fixture/onecard/T6A1I1 direct warmtiming only','Three32callbatchmeans, not independentkernelp95','Eventspan includes hostgaps/barriers; no isolatedservice/physicaltraffic/servingclaim','Baseline anchor checks preserved S39 perfileSHA/bytes, not its oldprofiledtimings'],'closure':closure}
(P/'s48-analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='ascii')
print(json.dumps({k:report[k] for k in ('status','paired_complete_percent_change','ratio_of_complete_medians_percent_change','declared_screen','host_submit_fraction_of_eventspan')}))
print(json.dumps(metrics))
