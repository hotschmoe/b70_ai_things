#!/usr/bin/env python3
"""Inside-owned-container IPC-only observation/original control, no GPU query."""
import argparse,hashlib,json,os,re,stat,struct,subprocess,time
from pathlib import Path

def require(ok,msg):
    if not ok:raise ValueError(msg)
def signature(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def process_identity(pid):
    raw=Path('/proc/'+str(pid)+'/stat').read_text();right=raw.rfind(')');fields=raw[right+2:].split();require(right>0 and len(fields)>19,'Actual target PID1 identity required');return dict(pid=pid,parent_pid=int(fields[1]),start_ticks=int(fields[19]),comm=raw[raw.index('(')+1:right])
def identity():return process_identity(1)
def session_name(pid,start_ticks,binding):
    require(type(pid)is int and pid>0 and type(start_ticks)is int and start_ticks>0 and type(binding)is str and re.fullmatch('[0-9a-f]{64}',binding),'Typed original producer/source required');return 'B70WarmP'+str(pid)+'T'+str(start_ticks)+'S'+binding[:20]
def snapshot(session,base=Path('/dev/shm')):
    require(type(session)is str and re.fullmatch(r'B70WarmP[1-9][0-9]*T[1-9][0-9]*S[0-9a-f]{20}',session),'Exact source-derived unique temporal session required')
    path=Path(base)/('uctrl'+session);require(path.parent.resolve()==path.parent and not path.is_symlink(),'Nonaliased original named IPC path required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_uid==os.geteuid()and before.st_size==1024,'Exact owned source TemporalControl1024 IPC extent required')
        raw=os.read(fd,1025);after=os.fstat(fd);pathstat=path.stat();require(len(raw)==1024 and signature(before)==signature(after)==signature(pathstat),'Original IPC object changed during observation')
        again=os.pread(fd,1024,0);require(again==raw,'IPC state changed during observation')
    finally:os.close(fd)
    state=struct.unpack('<i',raw[:4])[0];require(state in(0,1,2),'Actual source enum value required')
    return dict(path=str(path),state=state,raw_hex=raw.hex(),sha256=hashlib.sha256(raw).hexdigest(),stat=signature(after),uid=after.st_uid,mode=stat.S_IMODE(after.st_mode),read_started_or_finished_monotonic_ns=time.monotonic_ns())
def clock_boundary():
    path=Path('/results/unitrace/clock.1.jsonl');require(path.is_file()and not path.is_symlink(),'Exact original native clock file required')
    before=path.stat();require(before.st_size>0 and before.st_size<=512<<20,'Bounded original clock endpoint required')
    with path.open('rb')as stream:
        offset=max(0,before.st_size-32768);stream.seek(offset);tail=stream.read(32769)
    after=path.stat();require(signature(before)==signature(after)and tail.endswith(b'\n'),'Stable complete idle clock endpoint required')
    last=tail.splitlines()[-1];require(len(last)<=4096,'Source clock row bound required');row=json.loads(last)
    require(type(row['pid'])is int and row['pid']==1 and type(row['sequence'])is int and row['sequence']>0,'Actual source writer/sequence required')
    return dict(path=str(path),bytes=after.st_size,stat=signature(after),tail_offset=offset,tail_hex=tail.hex(),tail_sha256=hashlib.sha256(tail).hexdigest(),last_sequence=row['sequence'],last_line_hex=last.hex())

def check_transition(before,after,action):
    expected={'observe-paused':(1,1),'resume':(1,0),'observe-resumed':(0,0)};require(action in expected,'Exact owned control action required');a,b=expected[action]
    require(type(before['state'])is int and type(after['state'])is int,'Exact integer IPC state words required')
    require(before['state']==a and after['state']==b and before['path']==after['path']and before['stat'][:3]==after['stat'][:3]and before['uid']==after['uid']and before['mode']==after['mode'],'Original named IPC state/ownership changed incorrectly')
    left=bytes.fromhex(before['raw_hex']);right=bytes.fromhex(after['raw_hex']);require(len(left)==len(right)==1024 and left[4:]==right[4:],'Only original state word can change during explicit control')
    require(struct.unpack('<i',left[:4])[0]==a and struct.unpack('<i',right[:4])[0]==b,'State word bytes differ from declared observation')

def run(session,action,expected_tool_sha,output):
    output=Path(output);require(output.parent==Path('/results')and not output.exists(),'Exact new owned results file required');started=time.time();mono=time.monotonic_ns();owner=identity();before=snapshot(session);clock_before=clock_boundary();command=None;rc=None;stdout=b'';stderr=b'';error=None;after=None;clock_after=None
    try:
        require(before['state']==(0 if action=='observe-resumed'else 1),'Actual source-defined initial collection state differs')
        if action=='resume':
            tool=Path('/pti/bin/unitrace');raw=tool.read_bytes();require(raw[:4]==b'\x7fELF'and hashlib.sha256(raw).hexdigest()==expected_tool_sha,'Actual installed original control ELF changed')
            command=[str(tool),'--resume',session]
            # Container control never inherits injection/export flags. The
            # unchanged CLI only updates the original owned IPC state.
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C'},start_new_session=True)
            child=None
            try:
                if proc.poll()is None:
                    try:child=process_identity(proc.pid)
                    except FileNotFoundError:require(proc.poll()is not None,'Original control disappeared without terminal')
                stdout,stderr=proc.communicate(timeout=15)
            except BaseException:
                if proc.poll()is None:
                    now=process_identity(proc.pid)
                    require(child is not None and now==child and now['parent_pid']==os.getpid(),'Only original owned control child can be terminated')
                    proc.terminate()
                    try:stdout,stderr=proc.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        require(process_identity(proc.pid)==child,'Original control owner changed before kill');proc.kill();stdout,stderr=proc.communicate(timeout=5)
                raise
            finally:
                require(proc.poll()is not None,'Original control client must retire before returning')
                proc.wait()
                for stream in(proc.stdout,proc.stderr):
                    if stream is not None:stream.close()
            rc=proc.returncode;require(type(rc)is int and rc==0 and len(stdout)+len(stderr)<=65536,'Original temporal control must retire normally with bounded output')
        after=snapshot(session);clock_after=clock_boundary();check_transition(before,after,action);require(identity()==owner,'Actual original targetPID changed during control')
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    result=dict(schema=1,passed=error is None,error=error,session=session,action=action,target=owner,before=before,after=after,clock_before=clock_before,clock_after=clock_after,command=command,return_code=rc,stdout_hex=stdout.hex(),stderr_hex=stderr.hex(),started_epoch=started,finished_epoch=time.time(),started_monotonic_ns=mono,finished_monotonic_ns=time.monotonic_ns(),GPU_queries_added=False,process_memory_read=False,original_shared_IPC_bytes_read=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');require(result['passed'],'Owned temporal control failed; original failure retained');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--action',choices=('observe-paused','resume','observe-resumed'),required=True);p.add_argument('--expected-tool-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.session,a.action,a.expected_tool_sha256,a.output)
