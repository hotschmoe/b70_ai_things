"""Retain actual census failure evidence before rejecting; never abandon CLI."""
import hashlib,json,subprocess,time
from pathlib import Path
import pti_census_owned_exec_v2 as original
lifecycle=original.lifecycle;helper=original.helper;require=helper.require;HERE=original.HERE
command=original.command

def run(directory,name,native_command,image,binding,phase,sha):
    directory=Path(directory);started=time.time();started_mono=time.monotonic_ns();obj=None;after=None;proc=None;output=b'';cmd=None;errors=[];receipt=None;value=None
    def write(suffix,data):
        path=directory/(phase+suffix)
        if isinstance(data,bytes):path.write_bytes(data)
        else:path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='ascii')
    try:
        obj=lifecycle.inspect_owned(name,native_command,image,binding);write('-owner-before.json',obj)
        require(obj['State']['Running']is True and type(obj['State']['Pid'])is int and obj['State']['Pid']>0,'Exact live original native container required')
        require(str(HERE/'pti_mapped_injection_census_v1.py')+':/pti-census.py:ro'in obj['HostConfig']['Binds']and any(x.endswith(':/pti:ro')for x in obj['HostConfig']['Binds']),'Exact readonly census helper/install mounts required')
        cmd=command(obj['Id'],phase,sha);write('-exec.command.json',cmd)
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True);output,_=proc.communicate(timeout=90)
    except BaseException as exc:
        errors.append(type(exc).__name__+': '+str(exc))
    finally:
        # Keep the lease until the actual Dockerexec command resolves and drains;
        # do not kill a CLI and assume its daemon-side child cannot be created.
        if proc is not None:
            while proc.poll()is None:
                try:
                    data,_=proc.communicate(timeout=10)
                    if data is not None:output=data
                except BaseException as exc:errors.append('retirement: '+type(exc).__name__+': '+str(exc))
            try:
                data,_=proc.communicate()
                if data is not None:output=data
                proc.wait()
            except BaseException as exc:errors.append('terminal drain: '+type(exc).__name__+': '+str(exc))
            for stream in(proc.stdout,proc.stderr,proc.stdin):
                if stream is not None:
                    try:stream.close()
                    except BaseException as exc:errors.append('descriptor close: '+str(exc))
        write('-exec.stdout',output)
        try:
            after=lifecycle.inspect_owned(name,native_command,image,binding);write('-owner-after.json',after)
            require(obj is not None and after['Id']==obj['Id']and after['State']['Running']is True and after['State']['Pid']==obj['State']['Pid'],'Original native owner changed during census')
        except BaseException as exc:errors.append('owner-after: '+type(exc).__name__+': '+str(exc))
    try:
        require(proc is not None and type(proc.returncode)is int and proc.returncode==0,'Owned metadata census exec failed')
        root=directory/phase;receipt=json.loads((root/'receipt.json').read_bytes());require(receipt['passed']is True and receipt['sha256']==sha and receipt['phase']==phase and receipt['process']['pid']==1,'Actual original census receipt differs');raw=(root/'mapped-injection.elf').read_bytes();require(hashlib.sha256(raw).hexdigest()==sha and len(raw)==receipt['bytes'],'Actual exported backing bytes differ')
        value={'passed':not errors,'command':cmd,'container_id':obj['Id'],'host_native_pid':obj['State']['Pid'],'started_epoch':started,'finished_epoch':time.time(),'started_monotonic_ns':started_mono,'finished_monotonic_ns':time.monotonic_ns(),'return_code':proc.returncode,'original_container_path':receipt['original_container_path'],'exported_host_path':str(root/'mapped-injection.elf'),'exported_host_sha256':sha,'census':receipt,'metadata_stdout_bytes':len(output),'process_memory_bytes_read':False}
    except BaseException as exc:errors.append('census predicate: '+type(exc).__name__+': '+str(exc))
    diagnostic={'schema':1,'passed':not errors,'phase':phase,'command':cmd,'container_id':None if obj is None else obj['Id'],'host_native_pid':None if obj is None else obj['State']['Pid'],'started_epoch':started,'finished_epoch':time.time(),'return_code':None if proc is None else proc.returncode,'actual_exec_started':proc is not None,'actual_exec_terminal':proc is not None and proc.poll()is not None,'metadata_stdout_path':str(directory/(phase+'-exec.stdout')),'metadata_stdout_sha256':hashlib.sha256(output).hexdigest(),'metadata_stdout_bytes':len(output),'owner_after_observed':after is not None,'errors':errors}
    write('-exec.receipt.json',diagnostic)
    if errors:raise ValueError('Owned metadata census failed; original diagnostics retained: '+'; '.join(errors))
    return value
