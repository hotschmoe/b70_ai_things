#!/usr/bin/env python3
"""ROOT CPU-only bounded page observation; no model identity or causal claim."""
import argparse,hashlib,json,mmap,os,signal,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ALLOWED_READERS={'admit_postboot_original_qsa_reference_v4.py'}
PAGE=4096
CANCEL=threading.Event()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def stat5(path):
 s=path.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def write(path,row):path.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n',encoding='ascii')
def page(path,offset,direct=False):
 before=stat5(path);fd=os.open(path,os.O_RDONLY|(os.O_DIRECT if direct else 0))
 try:
  if direct:
   with mmap.mmap(-1,PAGE)as buf:
    view=memoryview(buf)
    try:n=os.preadv(fd,[view],offset);raw=bytes(view[:n])
    finally:view.release()
  else:raw=os.pread(fd,PAGE,offset)
 finally:os.close(fd)
 require(len(raw)==PAGE and stat5(path)==before,'Page length or non-atime source identity changed')
 return raw,before
def dontneed(path,offset):
 fd=os.open(path,os.O_RDONLY)
 try:os.posix_fadvise(fd,offset,PAGE,os.POSIX_FADV_DONTNEED)
 finally:os.close(fd)
def phases(path):
 if path is None:return []
 require(path.is_file()and not path.is_symlink()and path.stat().st_size<=65536,'Bounded regular phase recipe required')
 rows=json.loads(path.read_text());require(type(rows)is list and len(rows)<=8,'Bounded explicit phase list required')
 for row in rows:
  require(type(row)is dict and set(row)=={'name','argv','script_sha256','timeout'}and type(row['name'])is str and row['name'].isascii()and row['name'].replace('_','').isalnum(),'Exact named phase recipe required')
  argv=row['argv'];require(type(argv)is list and len(argv)>=2 and all(type(x)is str for x in argv)and argv[0]=='/usr/bin/python3','Absolute root Python reader required')
  script=Path(argv[1]);require(script.parent==HERE and script.name in ALLOWED_READERS and not script.is_symlink()and digest(script.read_bytes())==row['script_sha256'],'Only exact pinned readonly reference reader phase supported')
  require(type(row['timeout'])is int and 1<=row['timeout']<=7200,'Bounded CPU phase timeout required')
 require(len({r['name']for r in rows})==len(rows),'Unique phase names required');return rows
def run_phase(row,out,deadline):
 write(out/(row['name']+'.command.json'),row);start=time.time();proc=None;threads=[];errors=[];streams={};timed_out=False
 def pump(role,stream):
  total=0;ended=False
  try:
   with (out/(row['name']+'.'+role)).open('xb')as f:
    while True:
     raw=stream.read(65536)
     if not raw:ended=True;break
     total+=len(raw)
     if total<=8<<20:f.write(raw)
     elif not errors:errors.append('Phase output quota exceeded')
  except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
  finally:stream.close();streams[role]={'bytes':total,'EOF':ended,'finished_epoch':time.time()}
 try:
  proc=subprocess.Popen(row['argv'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  for role,stream in [('stdout',proc.stdout),('stderr',proc.stderr)]:
   t=threading.Thread(target=pump,args=(role,stream),daemon=True);threads.append(t);t.start()
  try:
   until=min(deadline,time.monotonic()+row['timeout'])
   while proc.poll()is None:
    if CANCEL.is_set()or time.monotonic()>=until:raise subprocess.TimeoutExpired(row['argv'],row['timeout'])
    try:proc.wait(timeout=.2)
    except subprocess.TimeoutExpired:pass
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
 except BaseException as exc:
  errors.append(type(exc).__name__+': '+str(exc))
  if proc is not None and proc.poll()is None:
   os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
 finally:
  for t in threads:t.join(5)
  if proc is not None and any(t.is_alive()for t in threads):
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   for t in threads:t.join(5)
 result={'name':row['name'],'command':row,'source_sha256':digest(Path(__file__).read_bytes()),'started_epoch':start,'finished_epoch':time.time(),'producer_pid':os.getpid(),'child_pid':proc.pid if proc else None,'return_code':proc.returncode if proc else None,'timeout':timed_out,'errors':errors,'streams':streams,'reader_threads_retired':all(not t.is_alive()for t in threads),'GPU_workload_launched':False}
 result['passed']=proc is not None and proc.returncode==0 and not timed_out and not errors and result['reader_threads_retired']and len(streams)==2 and all(x['EOF']for x in streams.values())
 write(out/(row['name']+'.receipt.json'),result);require(result['reader_threads_retired'],'CPU phase descendants retain stdout; failed receipt preserved');return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--path',type=Path,required=True);p.add_argument('--page-offset',type=int,required=True);p.add_argument('--expected-page-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--duration',type=float,default=60);p.add_argument('--interval',type=float,default=1);p.add_argument('--phase-file',type=Path);p.add_argument('--single-page-dontneed-after-preservation',action='store_true');a=p.parse_args()
 for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,lambda *_:CANCEL.set())
 require(a.path.is_file()and not a.path.is_symlink()and a.path.resolve()==a.path and a.page_offset>=0 and a.page_offset%PAGE==0,'Exact regular source and aligned page offset required');require(len(a.expected_page_sha256)==64 and all(c in'0123456789abcdef'for c in a.expected_page_sha256),'Explicit expected page hash required');require(.1<=a.interval<=60 and 1<=a.duration<=7200 and a.duration/a.interval<=10000,'Bounded watcher duration/interval required');require(not a.output.exists(),'Exclusive new evidence directory required');rows=phases(a.phase_file);a.output.mkdir(parents=True);(a.output/'probe.snapshot.py').write_bytes(Path(__file__).read_bytes());start=time.time();stop=threading.Event();status=[];errors=[];original=stat5(a.path)
 buffered,_=page(a.path,a.page_offset);(a.output/'initial-buffered.bin').write_bytes(buffered);direct,_=page(a.path,a.page_offset,True);(a.output/'initial-direct.bin').write_bytes(direct)
 write(a.output/'initial.json',{'path':str(a.path),'offset':a.page_offset,'bytes':PAGE,'stat5':original,'buffered_sha256':digest(buffered),'direct_sha256':digest(direct),'expected_page_sha256':a.expected_page_sha256,'epoch':time.time()})
 require(digest(direct)==a.expected_page_sha256,'Current direct page differs from explicit expected hash; preserved before refusal')
 if a.single_page_dontneed_after_preservation:
  dontneed(a.path,a.page_offset);write(a.output/'single-page-dontneed.json',{'path':str(a.path),'offset':a.page_offset,'bytes':PAGE,'epoch':time.time(),'readonly_fd':True,'global_cache_flush':False})
 deadline=time.monotonic()+a.duration
 def watch():
  last=digest(buffered)
  try:
   with (a.output/'watch.jsonl').open('x')as f:
    while not stop.is_set()and not CANCEL.is_set()and time.monotonic()<deadline:
     raw,sig=page(a.path,a.page_offset);require(sig==original,'Watcher source identity changed');value=digest(raw);row={'seq':len(status),'epoch':time.time(),'sha256':value,'expected':value==a.expected_page_sha256,'stat5':sig}
     if last is not None and value!=last and not(a.output/'first-transition.json').exists():
      direct_now,_=page(a.path,a.page_offset,True);(a.output/'first-transition-buffered.bin').write_bytes(raw);(a.output/'first-transition-direct.bin').write_bytes(direct_now);write(a.output/'first-transition.json',{**row,'previous_sha256':last,'direct_sha256':digest(direct_now)})
     f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();status.append(row);last=value;stop.wait(min(a.interval,max(0,deadline-time.monotonic())))
  except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
 thread=threading.Thread(target=watch);thread.start();results=[]
 try:
  for row in rows:
   require(time.monotonic()<deadline,'Watcher deadline before CPU phase')
   raw,sig=page(a.path,a.page_offset);(a.output/(row['name']+'.before.bin')).write_bytes(raw);write(a.output/(row['name']+'.before.json'),{'epoch':time.time(),'sha256':digest(raw),'stat5':sig})
   results.append(run_phase(row,a.output,deadline))
   raw,sig=page(a.path,a.page_offset);(a.output/(row['name']+'.after.bin')).write_bytes(raw);write(a.output/(row['name']+'.after.json'),{'epoch':time.time(),'sha256':digest(raw),'stat5':sig})
  while not CANCEL.is_set()and time.monotonic()<deadline:stop.wait(min(.2,max(0,deadline-time.monotonic())))
 finally:
  stop.set();thread.join();final,_=page(a.path,a.page_offset);(a.output/'final-buffered.bin').write_bytes(final)
  report={'source_sha256':digest(Path(__file__).read_bytes()),'started_epoch':start,'finished_epoch':time.time(),'producer_pid':os.getpid(),'path':str(a.path),'page_offset':a.page_offset,'expected_page_sha256':a.expected_page_sha256,'stat5':original,'samples':len(status),'phase_receipts':results,'errors':errors,'watcher_retired':not thread.is_alive(),'page_content_transition_observed':(a.output/'first-transition.json').exists(),'full_model_identity_qualified':False,'cause_attributed':False,'GPU_touched':False,'source_file_written':False,'global_cache_flush':False}
  report['interrupted']=CANCEL.is_set();report['passed']=not CANCEL.is_set()and not errors and all(x['passed']for x in results);write(a.output/'report.json',report)
 require(report['passed'],'Observation failed; original evidence retained')
if __name__=='__main__':main()
