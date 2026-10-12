#!/usr/bin/env python3
"""Direct actual52 -> fresh58 NoDri build; no reused objects or archives."""
import argparse, hashlib, json, os, re, shlex, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
REPO=Path('/') if '--audit-build' in sys.argv else Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PLAN=HERE/'native58-paired-prefill-cooperative-engine-build-proposal-v1.json'
PLAN_SHA='92931ba113dd47241774a79f75669375f62bcd23313ee5617ea68ec91e9f6ab7'
PARENT=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww')
PARENT_RECEIPT='8f6c09aa60493dd6403771cd10592bdb780947110c3035379b4493d0a4298d94'
PARENT_PLAN='25c8b73f8fa0d92f0e7c562f232776844405dc62b7e3524d1584a92283f4630d'
def require(ok,msg):
    if not ok: raise ValueError(msg)
def sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Regular file required '+str(path))
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Changed while hashing '+str(path));return h.hexdigest()
def tree(path):
    result={}
    for leaf in sorted(path.rglob('*')):
        if '.git' in leaf.relative_to(path).parts:continue
        require(not leaf.is_symlink(),'Unexpected source symlink '+str(leaf))
        if leaf.is_file():result[str(leaf.relative_to(path))]=sha(leaf)
    return result
def output(argv):return subprocess.check_output(argv,text=True).strip()
def parent_proof():
    require(sha(PARENT/'receipt.json')==PARENT_RECEIPT and sha(PARENT/'plan.snapshot.json')==PARENT_PLAN,'Immutable actual52 receipt/plan required')
    a=json.loads((PARENT/'plan.snapshot.json').read_bytes());receipt=json.loads((PARENT/'receipt.json').read_bytes())
    require(receipt['build_rc']==0 and receipt['devices_exposed'] is False and receipt['external_source_unchanged'] is True and receipt['plan_snapshot_unchanged'] is True and receipt['plan_sha256']==PARENT_PLAN,'Completed source52 parent required')
    require(receipt['source_copy']==str(PARENT/'source') and receipt['ggml_source']==str(PARENT/'ggml-source') and receipt['build']==str(PARENT/'build'),'Exact source52 paths required')
    require(set(Path(p).name for p in receipt['binary_sha256'])==set(a['build_targets']),'All8 actual parent ELFs required')
    for p,want in receipt['binary_sha256'].items():require(Path(p).parent==PARENT/'build' and sha(p)==want,'Parent binary changed '+p)
    for p,want in a['expected_patched_source_sha256'].items():require(sha(PARENT/'source'/p)==want,'Parent consumed source changed '+p)
    return dict(receipt_sha256=PARENT_RECEIPT,plan_sha256=PARENT_PLAN,image=receipt['image'],source_tree=tree(PARENT/'source'),ggml_tree=tree(PARENT/'ggml-source'),binary_sha256=receipt['binary_sha256'])
def closure():
    recipe=HERE/'source58-fresh-from52-build-recipe-v2.json'
    for p,want in json.loads(recipe.read_bytes())['pins'].items():require(sha(REPO/p)==want,'Builder closure changed '+p)
    return sha(recipe)
def audit_build():
    require(not Path('/dev/dri').exists(),'NoDri compiler container required')
    expected=json.loads(Path('/expected-source.json').read_bytes())
    for p,want in expected.items():require(sha(Path('/src')/p)==want,'Current source58 changed '+p)
    text=Path('/build/fresh-dependencies.txt').read_text();seen=set();current=None;deps={}
    for line in text.splitlines():
        if ': #deps ' in line:current=line.split(': #deps ',1)[0];seen.add(current);continue
        if not line.strip():current=None;continue
        require(current is not None and line.startswith('    '),'Unparsed Ninja dependency record')
        p=os.path.normpath(line.strip());require(p.startswith(('/src/','/ggml/','/build/','/opt/intel/','/usr/')),'Foreign dependency '+p)
        deps[p]=sha(p)
    objects={str(p.relative_to('/build')):sha(p) for p in sorted(Path('/build').rglob('*.o'))}
    require(seen==set(objects),'Every fresh object needs actual compiler dependency records')
    compiles=json.loads(Path('/build/compile_commands.json').read_bytes());require(compiles,'Real compiler commands required')
    source_objects={}
    for row in compiles:
        command=row.get('command','');require(command and ' -c ' in command,'Actual compile command missing')
        source_objects[row['file']]=command
    for p in ('/src/sycl/src/prefill/prefill.cpp','/src/sycl/src/core/verify.cpp','/src/sycl/src/program/generate.cpp'):
        require(p in source_objects,'Changed translation unit not compiled '+p)
    require('/src/include/strata/prefill/cooperative.hpp' in deps,'New control header not consumed')
    result=dict(passed=True,object_sha256=objects,consumed_dependency_sha256=deps,compile_commands_sha256=sha('/build/compile_commands.json'),ninja_commands_sha256=sha('/build/fresh-commands.txt'),ninja_dependencies_sha256=sha('/build/fresh-dependencies.txt'),fresh_all_source_compilation=True,reused_objects=[],reused_archives=[])
    Path('/build/fresh-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
    return 0
def absent(stderr,name):
    if not isinstance(stderr,bytes) or len(stderr)>4096:return False
    try:text=stderr.decode('ascii').strip().lower()
    except UnicodeDecodeError:return False
    return any(text==prefix+'no such '+kind+': '+name.lower() for prefix in ('','error: ','error response from daemon: ') for kind in ('container','object'))
def owned_compile(command,mounts,directory,receipt,log,lease_card):
    name='b70-source58-build-v2-'+str(os.getpid());label='b70.source58.build';owner=str(os.getpid())
    command=command[:2]+['--name',name,'--label',label+'='+owner]+command[2:]
    receipt['commands'].append(command);receipt['owned_compiler_command']=command
    (directory/'compiler-command.json').write_text(json.dumps(command,indent=2)+'\n',encoding='ascii')
    proc=None;terminal=None;signals=[];errors=[];forced=False;attempts=0
    def write(name,row):(directory/name).write_text(json.dumps(row,indent=2)+'\n',encoding='ascii')
    def on_signal(number,frame):signals.append(number)
    saved={n:signal.signal(n,on_signal) for n in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    def inspect():
        result=subprocess.run(['docker','inspect',name],capture_output=True,timeout=30)
        if result.returncode:
            require(absent(result.stderr,name),'Docker compiler state unknown; lease retained');return None
        rows=json.loads(result.stdout);require(len(rows)==1,'Exactly one original compiler required');row=rows[0];cfg=row['Config'];host=row['HostConfig']
        require(cfg['Labels'].get(label)==owner and cfg['Image']==receipt['image'] and cfg['Entrypoint']==['/bin/bash'] and cfg['Cmd']==command[-2:] and cfg['User']==str(os.getuid())+':'+str(os.getgid()),'Compiler owner/command changed; lease retained')
        require(sorted(host['Binds'])==sorted(mounts) and host['NetworkMode']=='none' and not host.get('Devices') and not host.get('DeviceRequests') and host['Privileged'] is False,'Original NoDri binds/devices/network changed; lease retained')
        require(all(type(host[k]) is int and host[k]==v for k,v in {'NanoCpus':4000000000,'Memory':16<<30,'MemorySwap':16<<30,'PidsLimit':512}.items()),'Typed compiler resources changed; lease retained')
        require(host['AutoRemove'] is False and type(row['RestartCount']) is int and row['RestartCount']==0,'Original compiler removal/restart contract changed; lease retained')
        return row
    try:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(8+lease_card,))
        write('compiler-client.json',{'pid':proc.pid,'session':os.getpgid(proc.pid),'started_epoch':time.time()});deadline=time.monotonic()+1800
        while proc.poll() is None:
            if signals or time.monotonic()>deadline or (directory/'build.log').stat().st_size>512<<20:
                raise RuntimeError('Compiler signal/deadline/log limit')
            time.sleep(.25)
        require(proc.returncode==0,'Fresh compiler failed; raw build log preserved')
    except BaseException as error:errors.append(type(error).__name__+': '+str(error))
    finally:
        # Keep the selected lease until exact owner terminal + creator EOF + absence.
        while True:
            attempts+=1
            try:
                row=inspect();require(proc is not None,'Original compiler creator unavailable; lease retained')
                if row and row['State']['Running']:
                    forced=True;errors.append('Forced original compiler cleanup');subprocess.run(['docker','stop','--time','20',row['Id']],capture_output=True,check=True,timeout=40);continue
                if proc.poll() is None:time.sleep(.25);continue
                if row:
                    terminal=row;write('compiler-terminal.json',row)
                    subprocess.run(['docker','rm',row['Id']],capture_output=True,check=True,timeout=30);continue
                proc.wait();receipt['owned_removed']=True;receipt['original_foreground_EOF']=True;receipt['compiler_return_code']=proc.returncode
                write('compiler-client-terminal.json',{'pid':proc.pid,'return_code':proc.returncode,'original_foreground_EOF':True,'terminal_epoch':time.time()})
                write('compiler-removed.json',{'name':name,'owned_removed':True,'creator_EOF':True,'terminal_epoch':time.time()});break
            except BaseException as error:
                message='Retirement '+type(error).__name__+': '+str(error)
                if message not in errors and len(errors)<64:errors.append(message)
                try:write('compiler-retirement-pending.json',{'attempts':attempts,'error':message[:4096],'lease_retained':True,'name':name,'owner':owner})
                except BaseException:pass
                time.sleep(1)
        if signals:errors.append('Signals latched '+repr(signals))
        receipt.update(forced_cleanup=forced,retirement_attempts=attempts,compiler_errors=errors,compiler_resource_bounds={'cpus':4,'memory_bytes':16<<30,'memory_swap_bytes':16<<30,'pids_limit':512,'network':'none','NoDri':True})
        try:
            require(terminal is not None and proc is not None and proc.returncode==0 and not forced,'Original normal compiler retirement required')
            state=terminal['State'];require(type(state['ExitCode']) is int and state['ExitCode']==0 and state['Running'] is False and state['Paused'] is False and state['Restarting'] is False and state['Dead'] is False and state['OOMKilled'] is False and state['Error']=='' and type(state['Pid']) is int and state['Pid']==0,'Typed original compiler terminal required')
            require((directory/'build.log').stat().st_size<=512<<20,'Final compiler log limit exceeded')
            receipt['owned_normal_exit']=True;receipt['compiler_terminal_sha256']=sha(directory/'compiler-terminal.json');receipt['compiler_client_terminal_sha256']=sha(directory/'compiler-client-terminal.json');receipt['compiler_removed_sha256']=sha(directory/'compiler-removed.json')
        except BaseException as error:errors.append('Terminal '+str(error));receipt['owned_normal_exit']=False
        for number,handler in saved.items():signal.signal(number,handler)
    require(not errors,'Owned compiler failed: '+'; '.join(errors))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,choices=range(1,9),default=4);parser.add_argument('--lease-card',type=int,choices=(0,1),default=1);parser.add_argument('--leased',action='store_true');parser.add_argument('--inspect-only',type=Path);parser.add_argument('--audit-build',action='store_true');args=parser.parse_args()
    if args.audit_build:return audit_build()
    require(sha(PLAN)==PLAN_SHA,'Frozen generation58 plan required');plan=json.loads(PLAN.read_bytes());builder_closure=closure();proof=parent_proof()
    require(plan['image']==proof['image'],'Same authoritative compiler image required')
    for p in plan['patches'][-3:]:require(sha(REPO/p['path'])==p['sha256'],'New source58 patch changed')
    if args.inspect_only:
        require(not args.inspect_only.exists(),'New inspection output required');args.inspect_only.write_text(json.dumps(proof,indent=2)+'\n',encoding='ascii');return 0
    if not args.leased:os.execv(str(REPO/'bin/gpu-run'),['gpu-run','--card',str(args.lease_card),sys.executable,__file__,*sys.argv[1:],'--leased'])
    require(os.path.samefile('/proc/self/fd/'+str(8+args.lease_card),'/mnt/vm_8tb/b70/gpu.lock.'+str(args.lease_card)),'Selected compiler lease required')
    require(output(['docker','image','inspect',plan['image'],'--format','{{.Id}}'])==plan['image'],'Pinned compiler image required')
    directory=Path(tempfile.mkdtemp(prefix='strata-native58-fresh52-v2-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-',dir='/mnt/vm_8tb/b70/build'))
    source=directory/'source';dependency=directory/'ggml-source';build=directory/'build';build.mkdir()
    (directory/'plan.snapshot.json').write_bytes(PLAN.read_bytes());(directory/'controller.py').write_bytes(Path(__file__).read_bytes());(directory/'parent-provenance.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='ascii');(directory/'expected-source.json').write_text(json.dumps(plan['expected_patched_source_sha256'],indent=2)+'\n',encoding='ascii')
    receipt=dict(schema=1,build_strategy='direct52_to58_fresh_all_source_v1',builder_generation=2,fresh_all_source_compilation=True,reused_objects=[],reused_archives=[],image=plan['image'],devices_exposed=False,parent_root=str(PARENT),external_source_root=str(PARENT/'source'),external_source_identity_kind='immutable_actual52_source_snapshot',parent_receipt_sha256=PARENT_RECEIPT,parent_provenance_sha256=sha(directory/'parent-provenance.json'),source_revision=plan['source_revision'],ggml_revision=plan['ggml']['revision'],source_copy=str(source),ggml_source=str(dependency),build=str(build),plan_snapshot=str(directory/'plan.snapshot.json'),plan_sha256=PLAN_SHA,patches=plan['patches'],builder_sha256=sha(Path(__file__)),builder_source_plan_sha256=builder_closure,commands=[],build_rc=1,status='preparing fresh58')
    def run(argv,log):receipt['commands'].append(argv);subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        with (directory/'build.log').open('w',encoding='ascii') as log:
            for old,new in ((PARENT/'source',source),(PARENT/'ggml-source',dependency)):
                shutil.copytree(old,new,copy_function=shutil.copy2)
                require(all(p.stat().st_ino!=(old/p.relative_to(new)).stat().st_ino or p.stat().st_dev!=(old/p.relative_to(new)).stat().st_dev for p in new.rglob('*') if p.is_file()),'Parent copies must have independent inodes')
            require(tree(source)==proof['source_tree'] and tree(dependency)==proof['ggml_tree'],'Copied authoritative inputs differ')
            for patch in plan['patches'][-3:]:
                path=str(REPO/patch['path']);run(['git','-C',str(source),'apply','--check',path],log);run(['git','-C',str(source),'apply',path],log)
            for p,want in plan['expected_patched_source_sha256'].items():require(sha(source/p)==want,'Exact75 source58 identity required '+p)
            altered=tree(source);changed={p for p,w in altered.items() if proof['source_tree'].get(p)!=w}
            require(changed==set(plan['source58_expected_delta_sha256']) and set(altered)-set(proof['source_tree'])=={'include/strata/prefill/cooperative.hpp'},'Only five reviewed files and new control header may change')
            configure=['cmake','-S','/src/sycl','-B','/build','-G','Ninja']+['-D'+k+'='+v for k,v in sorted(plan['cmake_flags'].items())]
            script='\n'.join(['set -Eeo pipefail',"trap 'rc=$?; echo fresh58-failed line=$LINENO rc=$rc >&2; exit $rc' ERR",'test ! -e /dev/dri','if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh; fi','icpx --version > /build/compiler-version.txt','dpkg-query -W > /build/packages.txt',shlex.join(configure),'ninja -C /build -t commands '+shlex.join(plan['build_targets'])+' > /build/fresh-commands.txt','ninja -C /build -v '+shlex.join(plan['build_targets'])+' -j '+str(args.jobs),'ninja -C /build -t deps > /build/fresh-dependencies.txt','python3 /controller.py --audit-build'])
            receipt['container_script']=script
            mounts=[str(source)+':/src:ro',str(dependency)+':/ggml:ro',str(build)+':/build',str(directory/'controller.py')+':/controller.py:ro',str(directory/'expected-source.json')+':/expected-source.json:ro']
            command=['docker','run','--network','none','--cpus','4','--memory','16g','--memory-swap','16g','--pids-limit','512','--user',str(os.getuid())+':'+str(os.getgid())]
            for mount in mounts:command+=['-v',mount]
            command+=['--entrypoint','/bin/bash',plan['image'],'-lc',script]
            owned_compile(command,mounts,directory,receipt,log,args.lease_card)
        for target in plan['build_targets']:
            with (build/target).open('rb') as handle:require(handle.read(4)==b'\x7fELF','Fresh ELF required '+target)
        receipt['binary_sha256']={str(build/t):sha(build/t) for t in plan['build_targets']};receipt['patched_source_sha256']={p:sha(source/p) for p in plan['overlay_files']};receipt['cmake_cache_sha256']=sha(build/'CMakeCache.txt');receipt['fresh_dependency_audit_sha256']=sha(build/'fresh-audit.json');receipt['fresh_dependency_audit']=json.loads((build/'fresh-audit.json').read_bytes());receipt['ggml_tree']=output(['git','-C',str(dependency),'rev-parse','HEAD^{tree}']);receipt.update(build_rc=0,status='Fresh all-source58/all8 build; model/math/latency/cancellation unqualified')
    except BaseException as error:receipt.update(build_rc=1,status='failed fresh58; preserve failed tree',error=type(error).__name__+': '+str(error))
    finally:
        try:
            require(parent_proof()==proof,'Parent source52 changed');require(closure()==builder_closure,'Builder source closure changed');receipt['parent_unchanged']=True
        except BaseException as error:receipt.update(build_rc=1,parent_unchanged=False,closing_error=str(error))
        receipt['external_source_unchanged']=receipt.get('parent_unchanged',False);receipt['plan_snapshot_unchanged']=sha(directory/'plan.snapshot.json')==PLAN_SHA and sha(PLAN)==PLAN_SHA
        if not receipt['plan_snapshot_unchanged']:receipt['build_rc']=1
        (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='ascii')
    print(json.dumps(dict(receipt=str(directory/'receipt.json'),build_rc=receipt['build_rc'])));return receipt['build_rc']
if __name__=='__main__':raise SystemExit(main())
