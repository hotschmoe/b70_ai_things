#!/usr/bin/env python3
"""Inside CPU-only container: synthetic IPC, actual installed CLI/parser."""
import ast,hashlib,json,os,struct,subprocess,time
from pathlib import Path
import importlib.util
TOOL_SHA='9f481a37a2884f7c7d499f33b5f41dbc00e2c0d089a9e786d059881803417b0c'
def require(ok,msg):
    if not ok:raise ValueError(msg)
def load():
    spec=importlib.util.spec_from_file_location('helper','/helper.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run():
    out=Path('/results');helper=load();tool=Path('/pti/bin/unitrace');require(hashlib.sha256(tool.read_bytes()).hexdigest()==TOOL_SHA,'Actual fresh tool bytes required')
    owner=helper.process_identity(os.getpid());source_sha=hashlib.sha256(Path('/helper.py').read_bytes()).hexdigest();session=helper.session_name(os.getpid(),owner['start_ticks'],source_sha);path=Path('/dev/shm')/('uctrl'+session);missing=session+'Missing';unknown=Path('/dev/shm')/('uctrl'+missing);require(not path.exists()and not unknown.exists(),'Unique fresh synthetic IPC objects required')
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:os.write(fd,struct.pack('<i',0)+bytes(1020))
    finally:os.close(fd)
    rows=[];error=None
    try:
        before=helper.snapshot(session);require(before['state']==0,'Synthetic initial resumed state required')
        pause=[str(tool),'--pause',session];p=subprocess.run(pause,capture_output=True,timeout=10,env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C'});paused=helper.snapshot(session);require(p.returncode==0 and paused['state']==1,'Actual CLI pause must write state1')
        node=next(n for n in ast.parse(Path('/helper.py').read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='run');assigns=[n for n in ast.walk(node)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='command'and isinstance(n.value,ast.List)];require(len(assigns)==1,'Exact actual helper command source required');resume=eval(compile(ast.Expression(assigns[0].value),'[actual-helper-command]','eval'),{'tool':tool,'session':session})
        r=subprocess.run(resume,capture_output=True,timeout=10,env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C'});resumed=helper.snapshot(session);helper.check_transition(paused,resumed,'resume');require(r.returncode==0,'Actual helper argv must be accepted by installed CLI')
        u=subprocess.run([str(tool),'--resume',missing],capture_output=True,timeout=10,env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C'});refused=False
        try:helper.snapshot(missing)
        except (ValueError,FileNotFoundError):refused=True
        require(refused and not unknown.exists(),'Absent session cannot become a qualified control')
        rows=[dict(command=pause,return_code=p.returncode,stdout_hex=p.stdout.hex(),stderr_hex=p.stderr.hex()),dict(command=resume,return_code=r.returncode,stdout_hex=r.stdout.hex(),stderr_hex=r.stderr.hex()),dict(command=[str(tool),'--resume',missing],return_code=u.returncode,stdout_hex=u.stdout.hex(),stderr_hex=u.stderr.hex())]
        result=dict(schema=2,passed=True,errors=[],fixture='synthetic named1024 IPC; actual fresh CLI; no model process',tool_sha256=TOOL_SHA,helper_source_sha256=source_sha,session=session,before=before,paused=paused,resumed=resumed,commands=rows,missing_session_helper_refused=True,missing_session_object_created=False,GPU_devices_exposed=False,model_payload_read=False,model_runtime_qualification_transferred=False)
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc);result=dict(schema=2,passed=False,errors=[error],commands=rows,GPU_devices_exposed=False,model_payload_read=False)
    finally:
        if path.exists():path.unlink()
    result['synthetic_IPC_removed']=not path.exists();result['finished_epoch']=time.time();(out/'inside.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');require(result['passed'],'Actual NoDri temporal fixture failed');return result
if __name__=='__main__':run()
