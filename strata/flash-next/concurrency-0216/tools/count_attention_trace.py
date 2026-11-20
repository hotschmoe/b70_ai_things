#!/usr/bin/env python3
"""Count relevant C2 verifier/attention kernels in a bounded diagnostic trace."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--markers', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    marks = json.loads(a.markers.read_text())
    for m in marks:
        m['trace_us'] = ((m['real_before_ns'] + m['real_after_ns']) / 2000 + (m['raw_ns'] - m['boot_ns']) / 1000)
    def t(label): return next(m['trace_us'] for m in marks if m['label']==label)
    lo,hi=max(t('b2_first_0'),t('b2_first_1')),t('b2_end')
    groups = defaultdict(list)
    invalid = 0
    with a.trace.open('rb') as f:
        f.seek(max(0,a.trace.stat().st_size-32))
        closed = re.sub(rb'\s+',b'',f.read()).endswith(b']}')
    with a.trace.open('rb',buffering=1024*1024) as f:
        for line in f:
            if not any(k in line for k in (b'attn_rows',b'QsaVectorCodes',b'resolve_kernel',b'copy_kernel_4bb253',b'KregQ5K')):
                continue
            try: event=json.loads(line.rstrip(b',\r\n'))
            except ValueError:
                invalid+=1;continue
            if event.get('ph')!='X' or event.get('cat')!='gpu_op' or event.get('pid')!=4294952799: continue
            ts,dur=event['ts'],event['dur']
            if lo<=ts and ts+dur<=hi:groups[event['name']].append(dur)
    a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,ds in groups.items():
        rows.append(dict(operation=name,count=len(ds),total_ms=sum(ds)/1000,mean_us=sum(ds)/len(ds)))
    rows.sort(key=lambda r:-r['total_ms'])
    with (a.out/'attention-kernels.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['operation','count','total_ms','mean_us']);w.writeheader();w.writerows(rows)
    summary=dict(source_closed=closed,invalid_event_lines=invalid,start_us=lo,end_us=hi,
                 operations=rows,note='Profiled client-bounded interval; events may overlap or include edge work. Not an unprofiled speed measure.')
    (a.out/'attention-kernels.json').write_text(json.dumps(summary,indent=2,ensure_ascii=True)+'\n')
    print(json.dumps(dict(source_closed=closed,invalid_event_lines=invalid,operation_count=len(rows))))
    if not closed or invalid or not rows:raise RuntimeError('Trace did not close or contains no selected operations')


if __name__=='__main__':main()
