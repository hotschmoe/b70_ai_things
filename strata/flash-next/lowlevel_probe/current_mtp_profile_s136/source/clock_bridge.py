#!/usr/bin/env python3
"""Fresh host clock brackets, not proof of the C++ steady_clock clock ID."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=[]
for _ in range(64):
 m0=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
 raw=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
 boot=time.clock_gettime_ns(time.CLOCK_BOOTTIME)
 real=time.clock_gettime_ns(time.CLOCK_REALTIME)
 m1=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
 s.append(dict(monotonic_before_ns=m0,raw_ns=raw,boottime_ns=boot,realtime_ns=real,monotonic_after_ns=m1,bracket_ns=m1-m0))
b=min(s,key=lambda x:x['bracket_ns']);mid=(b['monotonic_before_ns']+b['monotonic_after_ns'])/2
v=dict(status='HOST_CLOCK_BRACKETS_ONLY',selected=b,samples=s,offset_from_monotonic_mid_ns={k:b[k]-mid for k in ['raw_ns','boottime_ns','realtime_ns']},uncertainty_half_bracket_ns=b['bracket_ns']/2,native_steady_to_MONOTONIC_identity='UNVERIFIED',native_to_trace_join_allowed=False,scope='Same-host fresh samples. C++ steady_clock implementation and loaded-library provenance not established by Python clock samples; preserve CPU steady spans/ordinal roles separately. No cross-card GPU join.')
with a.output.open('x',encoding='ascii')as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps({'status':v['status'],'bracket_ns':b['bracket_ns'],'native_to_trace_join_allowed':False}))
