"""Read-only host /proc sampling, started only immediately before dockerstop."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

def read(p):
    try:return p.read_text()
    except OSError as e:return 'READ_ERROR:'+type(e).__name__

def stat_fields(text):
    try:
        fields=text[text.rindex(')')+2:].split()
        return {'state':fields[0],'starttime':int(fields[19]),'utime':int(fields[11]),
                'stime':int(fields[12]),'exit_code_waitstatus':int(fields[49])}
    except Exception:return {'parse_error':True}

def watch(pid,out,seconds=45,armed=None):
    root=Path('/proc')/str(pid);started=time.monotonic();initial=read(root/'stat')
    starttime=stat_fields(initial).get('starttime');last={};count=0;gone=False
    with out.open('a',encoding='ascii')as f:
        while time.monotonic()-started<seconds:
            text=read(root/'stat');leader=stat_fields(text)
            if text.startswith('READ_ERROR:')or leader.get('starttime')!=starttime:
                gone=True;break
            threads=[]
            for t in sorted((root/'task').glob('*')):
                threads.append({'tid':int(t.name),'stat':read(t/'stat'),'wchan':read(t/'wchan')})
            last={'elapsed_s':time.monotonic()-started,'wall_ns':time.time_ns(),'host_pid':pid,
                  'leader':leader,'leader_stat':text,'status':read(root/'status'),'threads':threads}
            f.write(json.dumps(last,ensure_ascii=True)+'\n');f.flush();count+=1
            if armed and not armed.exists():armed.write_text(json.dumps({'host_pid':pid,'first_sample_saved':True}))
            if leader.get('state')=='Z':break
            time.sleep(.5)
    result={'host_pid':pid,'sample_count':count,'proc_disappeared':gone,'last_leader':last.get('leader'),
            'elapsed_s':time.monotonic()-started,'scope':'read-only stat/wchan; missing process does not prove exit code'}
    out.with_suffix('.summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
    if armed and not armed.exists():armed.write_text(json.dumps({'host_pid':pid,'first_sample_saved':False}))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--container');p.add_argument('--pid',type=int)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--armed',type=Path);p.add_argument('--seconds',type=int,default=45)
    a=p.parse_args();assert 1<=a.seconds<=45
    pid=a.pid
    if pid is None:
        r=subprocess.run(['docker','top',a.container,'-eo','pid,comm,args'],capture_output=True,text=True,timeout=5)
        a.out.with_suffix('.docker-top.txt').write_text(r.stdout+r.stderr,encoding='ascii',errors='backslashreplace')
        candidates=[int(l.split()[0])for l in r.stdout.splitlines()[1:]if 'EngineCor'in l]
        if len(candidates)!=1:
            if a.armed:a.armed.write_text(json.dumps({'host_pid':None,'error':'EngineCore hostPID lookup failed'}))
            raise SystemExit(2)
        pid=candidates[0]
    print(json.dumps(watch(pid,a.out,a.seconds,a.armed)),flush=True)
