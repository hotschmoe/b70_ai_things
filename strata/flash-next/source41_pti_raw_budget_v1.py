"""Periodic raw apparent-byte budget; overshoot is observed, never hard-capped."""
import stat,threading,time
from pathlib import Path
TOTAL=1<<30;INTERVAL=1

def require(value,message):
    if not value:raise ValueError(message)

def measure(root):
    root=Path(root);require(root.is_dir()and not root.is_symlink(),'Original owned raw root required');rows=[]
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'Profiler raw symlink refused')
        try:info=path.stat()
        except FileNotFoundError:continue
        if stat.S_ISDIR(info.st_mode):continue
        require(stat.S_ISREG(info.st_mode),'Profiler raw special file refused');rows.append({'path':str(path.relative_to(root)),'bytes':info.st_size})
    return {'epoch':time.time(),'monotonic_ns':time.monotonic_ns(),'bytes':sum(row['bytes']for row in rows),'files':rows}

class Watcher:
    def __init__(self,root,interrupt,cap=TOTAL,interval=INTERVAL):
        require(type(cap)is int and 0<cap<=TOTAL and type(interval)in(int,float)and 0<interval<=INTERVAL,'Bounded explicit raw watcher required')
        self.root=Path(root);self.interrupt=interrupt;self.cap=cap;self.interval=interval;self.stop=threading.Event();self.thread=None;self.rows=[];self.errors=[];self.latched=False;self.peak=0;self.overshoot=0
    def sample(self):
        row=measure(self.root);self.rows.append(row);self.peak=max(self.peak,row['bytes']);self.overshoot=max(self.overshoot,max(0,row['bytes']-self.cap))
        require(len(self.rows)<=21600,'Six-hour bounded watcher roster exceeded')
        if row['bytes']>self.cap:raise ValueError('Total profiler raw apparent-byte budget exceeded')
    def start(self):
        self.sample();self.thread=threading.Thread(target=self.run,name='source41-pti-raw-budget',daemon=False);self.thread.start()
    def run(self):
        try:
            while not self.stop.wait(self.interval):self.sample()
        except BaseException as exc:
            self.errors.append(type(exc).__name__+': '+str(exc));self.latched=True
            try:self.interrupt()
            except BaseException as failure:self.errors.append('notify: '+str(failure))
    def close(self):
        self.stop.set()
        if self.thread:
            while self.thread.is_alive():
                try:self.thread.join(.2)
                except BaseException as exc:self.errors.append('join: '+str(exc));self.latched=True
        try:self.sample()
        except BaseException as exc:self.errors.append('final: '+str(exc));self.latched=True
        return {'passed':not self.latched and not self.errors,'errors':self.errors,'samples':self.rows,'peak_observed_bytes':self.peak,'observed_overshoot_bytes':self.overshoot,'cap_bytes':self.cap,'interval_seconds':self.interval,'thread_retired':self.thread is None or not self.thread.is_alive(),'measurement':'regular raw-file apparent bytes; periodic samples can miss larger transient overshoot; not a hard storage quota'}
