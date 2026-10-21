"""Bounded post-READY HTTP phases; children own their CPUquiet locks."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

R=Path('/mnt/vm_8tb/github/b70_ai_things')
H=R/'vllm/flashnext/harness'
ALIAS='qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4'

def run_phases(out,base,budget=240,commands=None):
    started=time.monotonic();deadline=started+budget
    status={'budget_s':budget,'scope':'postREADY only; elapsed bound is not stall proof',
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phases':{},'overall_bound_expired':False}
    def save():
        tmp=out/'clients-status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n',encoding='ascii');tmp.replace(out/'clients-status.json')
    def progress(event,**kw):
        row=dict(event=event,elapsed_s=time.monotonic()-started,**kw)
        with (out/'client-progress.jsonl').open('a',encoding='ascii')as f:f.write(json.dumps(row)+'\n')
    env=dict(os.environ,F1_BASE=base,F1_MODEL='hotschmoe-dd',F1_ALIAS=ALIAS,F1_OUT_DIR=str(out),
             F1_OUT=str(out/'probe.json'),F1_LENGTHS='1024',F1_REPEAT='1',F1_COHERENCE_LEN='1024',
             F1_DECODE='0',F1_NEEDLE_LENS='',F1_TIMEOUT='180',B70_AGENT='D1')
    quiet=[str(R/'bin/cpu-run'),'--quiet']
    if commands is None:
        commands=[('gate',quiet+[sys.executable,str(H/'promo_gate.py'),'--base-url',base,'--profile','flashnext',
            '--expect-alias',ALIAS,'--timeout','180','--out',str(out/'gate.json')]),
          ('bench',quiet+[sys.executable,str(H/'bench_latency.py'),'--base-url',base,'--model','hotschmoe-dd',
            '--alias',ALIAS,'--decode-tokens','64','--prefill','1024','--timeout','180','--tag','d1-routeq8f',
            '--out',str(out/'bench.json')]),
          ('probe',quiet+[sys.executable,str(H/'f1_probe.py')]),
          # This existing callback takes its OWN quiet lock; never nest one.
          ('canonical',['bash',str(H/'d1_lmx_callback.sh')])]
    stopped=False
    def interrupted(signum,frame):
        nonlocal stopped
        stopped=True
    previous={s:signal.signal(s,interrupted)for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for name,command in commands:
            begin=time.monotonic()
            if stopped or begin>=deadline:status['overall_bound_expired']=begin>=deadline;break
            phase={'command':command,'started_elapsed_s':begin-started,'rc':None,'deadline_expired':False}
            status['phases'][name]=phase;save();progress('phase_start',phase=name)
            with (out/(name+'-client.log')).open('wb')as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                last=0
                while child.poll()is None:
                    now=time.monotonic()
                    if stopped or now>=deadline:
                        phase['deadline_expired']=now>=deadline
                        status['overall_bound_expired']=now>=deadline
                        progress('bound_or_interrupt',phase=name,bound_expired=now>=deadline,
                                 interpretation='overall time bound or user interrupt; not liveness diagnosis')
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                        break
                    if now-last>=5:
                        # Host-only output growth; no GPU access/per-op logging.
                        files=[out/'gate.json',out/'bench.json',out/'probe.json',out/'lmx/measure.log']
                        counters={p.name:p.stat().st_size for p in files if p.exists()}
                        progress('phase_active',phase=name,phase_elapsed_s=now-begin,
                                 output_bytes=counters,child_pid=child.pid,
                                 canonical_samples_saved=sum(1 for _ in (out/'lmx').glob('**/sample.json')))
                        last=now
                    time.sleep(.2)
                phase['rc']=child.returncode
            phase['finished_elapsed_s']=time.monotonic()-started
            progress('phase_return',phase=name,rc=phase['rc']);save()
            if stopped or status['overall_bound_expired']:break
        status['completed_all_phases']=len(status['phases'])==len(commands) and all(
            p['rc']is not None for p in status['phases'].values()) and not stopped and not status['overall_bound_expired']
        status['finished_elapsed_s']=time.monotonic()-started
        status['interrupted']=stopped
        save()
        if status['overall_bound_expired']:return 124
        if stopped:return 143
        return 0 if status['completed_all_phases'] and all(p['rc']==0 for p in status['phases'].values()) else 1
    finally:
        for s,h in previous.items():signal.signal(s,h)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--base',required=True)
    a=p.parse_args();raise SystemExit(run_phases(a.out,a.base))
