"""Small exact source41 CLI artifact contract; no inference implementation."""
import hashlib,json,math,re,threading,time
from pathlib import Path
from source41_model_cli_v2 import require
from serial37_canonical_json_v3 import canonical

def epoch(value):
    require(type(value)in(int,float)and math.isfinite(value)and value>0,'Finite typed actual epoch required');return value

def requests(rows,expected):
    require(type(rows)is list and len(rows)==len(expected)==4,'Exactly four original requests required')
    views=[]
    for row,want in zip(rows,expected):
        require(type(row['ordinal'])is int and row['ordinal']==want['ordinal'],'Actual ordinal differs');raw=row['raw']
        require(canonical(raw['ids'])==canonical(want['ids'])and type(raw['fresh'])is int and raw['fresh']==1 and raw['pin']is None and raw['cancel_requested']is None and raw['stop_sent']is False,'Actual fresh absent-PIN/noSTOP input differs')
        require(raw['command']=='GEN %d fresh=1 ckpt=1 temperature=0 logprobs=20 '%want['max_new']+' '.join(map(str,want['ids'])),'Exact actual native input command differs')
        lines=raw['stdout'];require(type(lines)is list and lines and all(type(x)is str for x in lines),'Original native stdout rows required')
        require(not any(x.startswith(('ERR','FATAL'))for x in lines),'Native emitted error')
        ids=[]
        for line in lines:
            if line.startswith('T '):
                fields=line.split();require(len(fields)==2 and fields[1].isdigit(),'Actual T shape differs');ids.append(int(fields[1]))
        done=lines[-1].split();require(len(done)>=6 and done[0]=='DONE' and done[1].isdigit()and done[2].isdigit(),'Actual DONE terminal missing')
        require(int(done[1])==len(ids)==len(raw['output_ids'])and canonical(ids)==canonical(raw['output_ids'])and 0<len(ids)<=want['max_new']and int(done[2])==len(want['ids'])and done[5]in('stop','length'),'Actual terminal/count/input/budget differs')
        require(all(math.isfinite(float(x))and float(x)>=0 for x in done[3:5]),'Native host intervals invalid')
        require(len(raw['stdout_seconds'])==len(lines)and all(type(x)in(int,float)and math.isfinite(x)and x>=0 for x in raw['stdout_seconds'])and raw['stdout_seconds']==sorted(raw['stdout_seconds']),'Original monotonic receive intervals differ')
        views.append({'ids':want['ids'],'output_ids':ids,'done_signature':done[:3]+done[5:]})
    return views

def equivalence(left,right):
    require(canonical(left)==canonical(right),'Actual source40/source41OFF outputs/finish differ')
    for arm in(left,right):require(canonical(arm[0])==canonical(arm[3])and canonical(arm[1])==canonical(arm[2]),'Actual repeated output/finish differs')
    return {'OFF_equivalence_passed':True,'scope':'four bounded fresh native CLI requests only; no API/fullmath/serving authority'}

class Guardian:
    """Only existing readonly three-page checks; failures latch and notify owner."""
    def __init__(self,check,interrupt,interval=2):
        require(type(interval)in(int,float)and 0<interval<=5,'Bounded guardian interval required')
        self.check=check;self.interrupt=interrupt;self.interval=interval;self.stop=threading.Event();self.rows=[];self.errors=[];self.thread=None
    def start(self):
        self.rows.append(self.check());self.thread=threading.Thread(target=self.run,name='source41-page-guardian',daemon=False);self.thread.start()
    def run(self):
        try:
            while not self.stop.wait(self.interval):
                require(len(self.rows)<4096,'Bounded guardian sample quota reached');self.rows.append(self.check())
        except BaseException as exc:
            self.errors.append(type(exc).__name__+': '+str(exc))
            try:self.interrupt()
            except BaseException as failure:self.errors.append('notify: '+str(failure))
    def close(self):
        self.stop.set()
        if self.thread is not None:self.thread.join()
        value={'passed':not self.errors,'errors':list(self.errors),'samples':self.rows,'thread_retired':self.thread is None or not self.thread.is_alive(),'finished_epoch':time.time(),'readonly_cache_population_effect_declared':True}
        require(value['thread_retired'],'Actual guardian must retire');return value
