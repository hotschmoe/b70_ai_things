#!/usr/bin/env python3
"""CPU-only fresh correction for default unitrace epoch-BOOTTIME+RAW timestamps."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
samples=[]
for _ in range(64):
 r0=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
 b=time.clock_gettime_ns(time.CLOCK_BOOTTIME)
 r1=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
 samples.append({'raw_before_ns':r0,'boottime_ns':b,'raw_after_ns':r1,'bracket_ns':r1-r0,'shift_us':(b-(r0+r1)/2)/1000})
best=min(samples,key=lambda x:x['bracket_ns'])
v={'time_ns':time.time_ns(),'trace_to_client_shift_us':best['shift_us'],'selected':best,'samples':samples,'formula':'epoch_client_us = raw_trace_ts_us + (BOOTTIME - MONOTONIC_RAW)/1000','source':'Installed pti-gpu/tools/unitrace/src/unitimer.h default anchor; do not use if UNITRACE_SystemTime or clock configuration differs.','scope':'Fresh host clock correction only; no inter-card GPU clock calibration.'}
a.output.write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
print(json.dumps({'output':str(a.output),'shift_us':v['trace_to_client_shift_us'],'bracket_ns':best['bracket_ns']}))
