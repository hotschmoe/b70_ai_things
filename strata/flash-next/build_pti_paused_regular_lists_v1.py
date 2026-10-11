#!/usr/bin/env python3
"""Root-executable fresh CPU-only tool builder; not invoked by author."""
import argparse,hashlib,json,os,shutil,subprocess,time,signal
from pathlib import Path
import prepare_pti_paused_regular_lists_v1 as patch
import prepare_pti_finalizer_build_proposal_v1 as recipe
import pti_unitrace_build_recipe_v1 as p
HERE=Path(__file__).resolve().parent
PROPOSAL=HERE/'pti-paused-regular-list-tool-build-proposal-v1.json'
PROPOSAL_SHA='42e4e3de36b3b376a254f46e0bc91648d9e4a64705421cd544e7f8ed68cd6c14'
SOURCE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-paused-regular-source-v1')
OUTPUT=Path('/mnt/vm_8tb/b70/build/pti-unitrace-paused-regular-c388-build-v1')
require=p.require

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(source,proposal):
    require(source.resolve()==source and source.is_dir(),'Canonical source directory required')
    actual={str(x.relative_to(source))for x in source.rglob('*')if(x.is_file()or x.is_symlink())and '.git'not in x.relative_to(source).parts}
    require(actual==set(proposal['expected_full_source_sha256'])|set(proposal['expected_source_symlinks']),'Exact full source file/link roster required')
    for n,want in proposal['expected_full_source_sha256'].items():
        path=source/n;require(path.is_file()and not path.is_symlink()and sha(path)==want,'Source bytes changed '+n)
    for n,target in proposal['expected_source_symlinks'].items():require((source/n).is_symlink()and str((source/n).readlink())==target,'Source link changed '+n)

def materialize():
    require(sha(PROPOSAL)==PROPOSAL_SHA,'New proposal changed');value=json.loads(PROPOSAL.read_bytes());old=json.loads((HERE/'pti-sampled-query-tool-build-proposal-v1.json').read_bytes());validate(patch.BASE,old)
    require(not SOURCE.exists()and SOURCE.parent.resolve()==SOURCE.parent,'Fresh canonical source only');SOURCE.mkdir()
    changed=patch.changes()
    for n in value['expected_full_source_sha256']:
        target=SOURCE/n;target.parent.mkdir(parents=True,exist_ok=True)
        if n in changed:target.write_bytes(changed[n])
        else:shutil.copy2(patch.BASE/n,target)
    for n,target in value['expected_source_symlinks'].items():
        path=SOURCE/n;path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(target)
    validate(SOURCE,value);validate(patch.BASE,old);return value

def inspection(obj,cmd,identity):
    require(obj['Id']==identity and obj['Config']['Image']==p.IMAGE,'Actual owned ID/image required')
    require(obj['Name'].lstrip('/')==cmd[cmd.index('--name')+1] and obj['Config']['Entrypoint']==['/bin/bash'] and obj['Config']['Cmd']==cmd[-2:] and obj['Config']['User']==cmd[cmd.index('--user')+1],'Actual full creator recipe differs')
    host=obj['HostConfig'];require(host['Privileged']is False and not host.get('DeviceCgroupRules')and not host.get('GroupAdd')and not host.get('CapAdd'),'Exact nonprivileged no-added-device-domain recipe required');require(not host['Devices']and not host['DeviceRequests'],'No GPU device access permitted')
    require(sorted(host['Binds'])==sorted(cmd[i+1]for i,v in enumerate(cmd)if v=='-v'),'Actual full mounts differ')
    for key,want in(('NanoCpus',2000000000),('Memory',4<<30),('MemorySwap',4<<30),('PidsLimit',512)):
        require(type(host[key])is int and host[key]==want,'Actual typed build resource differs '+key)
    require(host['NetworkMode']=='bridge'and obj['Config']['Labels'].get('b70.pti.source')==PROPOSAL_SHA,'Actual network/label changed')

def recover_created(cmd):
    """Exact original named recipe only; never removes a foreign name."""
    name=cmd[cmd.index('--name')+1]
    result=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=30)
    if result.returncode!=0:
        require('no such object'in result.stderr.lower(),'Authoritative original name absence required')
        return None,dict(return_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    objects=json.loads(result.stdout);require(type(objects)is list and len(objects)==1,'Unique original name required')
    obj=objects[0];inspection(obj,cmd,obj['Id']);return obj['Id'],obj

def main():
    value=materialize();source=p.source_binding();cmd=recipe.recipe(SOURCE,OUTPUT,PROPOSAL_SHA);cmd[-1]+='\nreadelf -Ws /out/install/lib/libunitrace_tool.so > /out/exported-symbols.txt\n';cmd[1]='create';require(not OUTPUT.exists()and OUTPUT.parent.resolve()==OUTPUT.parent,'Fresh canonical build only');OUTPUT.mkdir()
    def write(n,v):(OUTPUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    write('create-command.json',cmd);(OUTPUT/'controller.snapshot.py').write_bytes(Path(__file__).read_bytes());identity=None;proc=None;creator=None;errors=[];rc=None;removed=False;obj=None;started=time.time();creation_terminal=False
    def interrupted(signum,frame):raise InterruptedError('Owned CPU build interrupted '+str(signum))
    for signum in(signal.SIGINT,signal.SIGTERM):signal.signal(signum,interrupted)
    try:
        with(OUTPUT/'create.stdout').open('wb')as out,(OUTPUT/'create.stderr').open('wb')as err:
            creator=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True)
            try:creator.wait(timeout=60)
            except BaseException as exc:
                errors.append('create wait: '+type(exc).__name__+': '+str(exc))
                # Never abandon an ambiguous daemon request. Original client
                # must resolve before exact original name/ID census.
                while creator.poll()is None:
                    try:creator.wait(timeout=10)
                    except BaseException as retry:errors.append('creator settlement: '+str(retry))
            finally:creation_terminal=creator.poll()is not None
        require(creator.returncode==0,'Original create failed')
        identity=(OUTPUT/'create.stdout').read_text().strip();require(len(identity)==64 and all(x in '0123456789abcdef'for x in identity),'Exact actual create ID required')
        obj=json.loads(subprocess.check_output(['docker','inspect',identity],timeout=30))[0];inspection(obj,cmd,identity);write('created-inspection.json',obj)
        with(OUTPUT/'stdout.log').open('wb')as out,(OUTPUT/'stderr.log').open('wb')as err:
            proc=subprocess.Popen(['docker','start','-a',identity],stdout=out,stderr=err);rc=proc.wait(timeout=1200)
    except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
    finally:
        # Failure remains latched; original owned object/client must actually
        # settle. A foreign recipe is never stopped or treated as absence.
        prior_handlers={sig:signal.signal(sig,lambda *_:None)for sig in(signal.SIGINT,signal.SIGTERM)}
        rm_completed=False
        while True:
            try:
                if creator is not None and creator.poll()is None:
                    creator.wait(timeout=10);creation_terminal=True
                if identity is None and creation_terminal:
                    identity,recovery=recover_created(cmd);write('create-recovery.json',recovery)
                    if identity is None:break
                if identity is None:break
                if rm_completed:
                    absence=subprocess.run(['docker','inspect',identity],capture_output=True,text=True,timeout=30)
                    write('absence.json',dict(return_code=absence.returncode,stdout=absence.stdout,stderr=absence.stderr));require(absence.returncode!=0 and 'no such object'in absence.stderr.lower(),'Original removed ID absence must resolve');removed=True;break
                obj=json.loads(subprocess.check_output(['docker','inspect',identity],timeout=30))[0];inspection(obj,cmd,identity)
                if obj['State']['Running']:
                    subprocess.run(['docker','stop','--time','15',identity],check=True,capture_output=True,timeout=30)
                    obj=json.loads(subprocess.check_output(['docker','inspect',identity],timeout=30))[0];inspection(obj,cmd,identity)
                if proc is not None and proc.poll()is None:proc.wait(timeout=30)
                write('terminal-inspection.json',obj);require(obj['State']['Running']is False,'Actual build must retire')
                subprocess.run(['docker','rm',identity],check=True,capture_output=True,timeout=30);rm_completed=True;absence=subprocess.run(['docker','inspect',identity],capture_output=True,text=True,timeout=30)
                write('absence.json',dict(return_code=absence.returncode,stdout=absence.stdout,stderr=absence.stderr));require(absence.returncode!=0 and 'no such object'in absence.stderr.lower(),'Exact original ID absence required');removed=True;break
            except BaseException as exc:
                errors.append('cleanup settlement: '+type(exc).__name__+': '+str(exc));write('cleanup-pending.json',dict(original_id=identity,errors=errors,lease_release_safe=False));time.sleep(1)
        for sig,handler in prior_handlers.items():signal.signal(sig,handler)
        passed=not errors and type(rc)is int and rc==0 and obj is not None and type(obj['State']['ExitCode'])is int and obj['State']['ExitCode']==0 and obj['State']['OOMKilled']is False and obj['State']['Error']=='' and removed
        try:
            if passed:require(any('GLOBAL'in x and 'DEFAULT'in x and 'b70_pti_model_finalize_v1'in x and 'UND'not in x for x in(OUTPUT/'exported-symbols.txt').read_text().splitlines()),'Fresh finalizer export missing')
            require(p.source_binding()==source,'Original source changed');validate(SOURCE,value)
        except BaseException as exc:errors.append('closing source/export: '+type(exc).__name__+': '+str(exc));passed=False
        files={str(f.relative_to(OUTPUT)):dict(sha256=sha(f),bytes=f.stat().st_size)for f in(OUTPUT/'install').rglob('*')if f.is_file()and not f.is_symlink()}if(OUTPUT/'install').exists()else{}
        write('receipt.json',dict(passed=passed,build_rc=rc,error=None if not errors else '; '.join(errors),errors=errors,source_manifest_sha256=source,GPU_devices_exposed=False,normal_owned_removed=removed,original_creator_terminal=creation_terminal,original_start_client_terminal=proc is None or proc.poll()is not None,image=p.IMAGE,clock_source_root=str(SOURCE),clock_source_plan_sha256=PROPOSAL_SHA,all_patched_source_bytes_checked_before_after=True,installed_files=files,started_epoch=started,finished_epoch=time.time()))
    require(passed,'Fresh sampled tool build failed; raw evidence retained')
if __name__=='__main__':main()
