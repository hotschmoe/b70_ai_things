#!/usr/bin/env python3
"""Summarize recorded activity while all eight C8 streams overlap."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics

BASE=Path('/mnt/vm_8tb/b70/results/conc-0216')
OUT=Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/concurrency-0216/ledger/b7')
TRACE=BASE/'b7-trace/strata-dd-0216.453.json'


def main():
    src=BASE/'b7-profiled-c8'
    m=json.loads((src/'clock-marks.json').read_text())
    requests=json.loads((src/'repeat-0.json').read_text())
    start=max(r['common_clock']['first'] for r in requests)*1e9
    end=min(r['common_clock']['end'] for r in requests)*1e9
    if end <= start: raise RuntimeError('All eight client content intervals do not overlap')
    def tr(mark):
        return (mark['real_before_ns']+mark['real_after_ns'])/2-mark['boot_ns']+mark['raw_ns']
    slope=(tr(m[1])-tr(m[0]))/(m[1]['monotonic_ns']-m[0]['monotonic_ns'])
    def convert(monotonic_ns):
        return (tr(m[0])+slope*(monotonic_ns-m[0]['monotonic_ns']))/1000
    lo,hi=convert(start),convert(end)
    with TRACE.open('rb') as f:
        f.seek(max(0,TRACE.stat().st_size-32));closed=re.sub(rb'\s+',b'',f.read()).endswith(b']}')
    groups=defaultdict(list); names={}; threads={}; malformed=0; count=0; heads=[]
    with TRACE.open('rb',buffering=1024*1024) as f:
        for line in f:
            if not line.startswith(b'{"ph"'): continue
            if b'"ph": "M"' in line:
                try:
                    e=json.loads(line.rstrip(b',\r\n'))
                    if e.get('name')=='process_name': names[e['pid']]=e['args']['name']
                    if e.get('name')=='thread_name': threads[(e['pid'],e['tid'])]=e['args']['name']
                except (ValueError,KeyError): malformed+=1
            elif b'"ph": "X"' in line and (b'"cat": "gpu_op"' in line or b'"cat": "cpu_op"' in line):
                try:
                    e=json.loads(line.rstrip(b',\r\n'));ts=e['ts'];dur=e['dur']
                    if lo<=ts and ts+dur<=hi:
                        groups[(e['cat'],e['pid'],e['tid'],e['name'])].append(dur);count+=1
                        if e['cat']=='gpu_op' and e['pid']==4294952799 and 'KregQ5K' in e['name'] and ('{31040;' in e['name'] or '{62080;' in e['name']):
                            match=re.search(r'KregQ5K, dpct_kernel_scalar<(\d+)>',e['name'])
                            heads.append({'end_us':ts+dur,'rows':int(match.group(1)) if match else None})
                except (ValueError,KeyError,TypeError): malformed+=1
    rows=[]
    for (kind,pid,tid,op),values in groups.items():
        values.sort(); i=(len(values)-1)*.95; k=int(i)
        rows.append({'kind':'gpu' if kind=='gpu_op' else 'host','process':names.get(pid,str(pid)),
                     'engine_or_thread':threads.get((pid,tid),str(tid)),'operation':op,'count':len(values),
                     'total_ms':sum(values)/1000,'mean_us':statistics.mean(values),'median_us':statistics.median(values),
                     'p95_us':values[k]+(values[min(k+1,len(values)-1)]-values[k])*(i-k),
                     'min_us':values[0],'max_us':values[-1]})
    rows.sort(key=lambda r:(r['kind'],-r['total_ms']))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'operations.csv').open('w',newline='',encoding='ascii') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    heads.sort(key=lambda x:x['end_us'])
    gaps=[(b['end_us']-a['end_us'])/1000 for a,b in zip(heads,heads[1:])]
    def percentile(v,p):
        v=sorted(v); x=(len(v)-1)*p; i=int(x); return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
    window={'head_events':len(heads),'head_rows':dict(__import__('collections').Counter(str(x['rows']) for x in heads)), 'head_to_head_ms_p50':percentile(gaps,.5) if gaps else None,'head_to_head_ms_p95':percentile(gaps,.95) if gaps else None, 'note':'Profiled intervals between consecutive full-vocabulary verifier heads; includes commit, drafts, waits and capture/admission if present.'}
    out={'window_diagnostic':window,'trace':str(TRACE),'source_closed':closed,'start_us':lo,'end_us':hi,'complete_events':count,
         'clock_mapping':'Linear interpolation of realtime+RAW-BOOTTIME versus MONOTONIC, bounded by recorded begin/end samples; not proof of cross-device causality.',
         'scope':'All eight client content intervals overlap; the eight-row engine may schedule fewer than eight streams in each verifier window.',
         'malformed_event_lines':malformed,'operations':rows}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
    print(json.dumps({k:v for k,v in out.items() if k!='operations'},ensure_ascii=True))


if __name__=='__main__': main()
