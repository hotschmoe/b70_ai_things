#!/usr/bin/env python3
"""Root-owned fixture-conditional Unitrace capability probe, never a model gate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PLAN=HERE/'unitrace-indexer40-capability-source-plan-v1.json'
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
TOOL='/tool/bin/unitrace'
TOOL_SHA='4a24077796cb38c170159390aae02ff319d946d4198e2dd3e83549196030fa1b'
TOOL_LIBRARY='/tool/lib/libunitrace_tool.so'
TOOL_LIBRARY_SHA='003d0b068da6a16d6ed6bb937416f113bb0ff0bbd06ae332ae186dedb7930982'
INSTALL=Path('/mnt/vm_8tb/b70/build/pti-unitrace-6c0d6b0b-c388-build-v1/install')
SUPPORTED=('--chrome-call-logging','--chrome-kernel-logging','--chrome-device-logging','--device-timing','--kernel-submission','--device-timeline','--output-dir-path','--output','--pid','--tid','--verbose')
F17=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
HELPER=F17/'indexer-materialized40-v3-control-run-v1/build/indexer-materialized40'
FIXTURE=F17/'indexer-materialized40-v3-owned-inputs-v1'
MODES=('single_raw','single_materialized','steps_raw','steps_materialized','batch_raw','batch_materialized')
FIELDS={'input.f32':2048,'gamma.f32':512,'materialized.f16':1024,'expanded.f32':2048,'steps.i32':64,'wrong_raw_store.f32':1536,'before_states.bin':9228,'after_states.bin':9228}
FAULT=r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed'
STOP=False

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n')
def typed_equal(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def read(path):
 def unique(pairs):
  value={}
  for k,v in pairs:
   require(k not in value,'Duplicate JSON key');value[k]=v
  return value
 return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def file_binding(path,limit=64<<20):
 p=Path(path);require(not p.is_symlink(),'Regular nonsymlink evidence required');before=p.stat();require(stat.S_ISREG(before.st_mode)and before.st_size<=limit,'Regular evidence bound exceeded')
 def identity(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
 digest=sha(p);require(not p.is_symlink()and identity(p.stat())==identity(before),'Evidence identity/content metadata changed during hashing');return {'path':str(p),'sha256':digest,'bytes':before.st_size}
def source_binding():
 plan=read(PLAN)
 for name,digest in plan['files'].items():require(sha(ROOT/name)==digest,'Probe source changed '+name)
 for name,digest in plan['fixture_and_helper_files'].items():require(file_binding(name)['sha256']==digest,'Fixture/helper changed '+name)
 for name,digest in plan['current_tool_and_device_metadata_files'].items():require(file_binding(name)['sha256']==digest,'Current tool/device metadata changed '+name)
 require({p.name for p in FIXTURE.iterdir()}==set(plan['fixture_names']),'Exact five-file fixture required')
 pci=F17/'gpu0-pci-property-probe-v1';receipt=read(pci/'receipt.json');command=read(pci/'command.json');require(type(receipt['return_code'])is int and receipt['return_code']==0 and receipt['state']['ExitCode']==0 and receipt['state']['Running']is False and receipt['model_or_math_workload_started']is False and receipt['source_sha256']==sha(pci/'probe.cpp')and receipt['ZE_AFFINITY_MASK']=='0'and receipt['ONEAPI_DEVICE_SELECTOR']=='level_zero:gpu','Actual sole-SYCL-GPU metadata proof differs');require(IMAGE in command and 'ZE_AFFINITY_MASK=0'in command and 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu'in command and (pci/'stdout.log').read_bytes()==b'GPU_PCI=0000:0b:00.0\n','Actual masked GPU0/PCI target differs')
 meta=F17/'unitrace-c388-no-device-metadata-v1';r=read(meta/'receipt.json')
 require(r['passed']is True and r['normal_owned_removed']is True and r['GPU_devices_exposed']is False and r['tool_workload_started']is False and r['installed_bytes_unchanged']is True and r['build_receipt_sha256']==plan['build_receipt_sha256'],'Actual fresh metadata/build association differs')
 build=read(INSTALL.parent/'receipt.json');require(build['passed']is True and build['build_rc']==0 and build['error']is None and build['GPU_devices_exposed']is False and build['normal_owned_removed']is True and build['source_manifest_sha256']==plan['pti_source_manifest_sha256']and build['image']==IMAGE,'Actual fresh no-device build failed')
 for row in read(meta/'tool-commands.json'):
  expected={'help':[TOOL,'--help'],'version':[TOOL,'--version'],'ldd-unitrace':['/usr/bin/ldd',TOOL],'ldd-library':['/usr/bin/ldd',TOOL_LIBRARY],'symbols':['/usr/bin/readelf','--wide','--dyn-syms',TOOL_LIBRARY]}[row['label']]
  require(row['command']==expected and type(row['return_code'])is int and row['return_code']==0 and sha(meta/(row['label']+'.stdout'))==row['stdout_sha256']and sha(meta/(row['label']+'.stderr'))==row['stderr_sha256'],'Actual current metadata command/raw differs')
 require(all(k in(meta/'help.stdout').read_text()for k in SUPPORTED)and (meta/'version.stdout').read_bytes()==b'2.4.0 ()\n','Exact supported options/informational version differs')
 require(all('not found'not in(meta/(n+'.stdout')).read_text()for n in('ldd-unitrace','ldd-library')),'Actual unresolved dependency refused')
 source_root=Path(plan['pti_source_root']);manifest=read(HERE/'pti-unitrace-source-manifest-v1.json')
 for name,digest in manifest['tracked_files'].items():
  path=source_root/name
  # Original source manifest stores file bodies separately from symbolic links.
  require(not path.is_symlink()and path.is_file()and path.stat().st_size==digest['bytes']and sha(path)==digest['sha256'],'Pinned PTI source changed '+name)
 return sha(PLAN)
def helper_argv():return ['/leaf/indexer-materialized40','--inputs','/inputs','--output','/results/build/native-output','--maps-prefix','/results/build/idxmat40']
def tool_argv():return [TOOL,*SUPPORTED[:6],'--pid','--tid','--verbose','--output','/results/unitrace/trace',*helper_argv()]

def dependency_record_binding(executable,record):
 require(record['ldd_command']==['/usr/bin/ldd',executable]and type(record['ldd_return_code'])is int and record['ldd_return_code']==0,'Actual original ldd command/return differs')
 require('not found'not in record['ldd_stdout']and record['files']and record['explicit_auxiliary_scope']is None and record['actual_dynamic_metadata']==[],'All tool/helper hard dependency identities required; no AUX exception')

def libraries(executable):
 row=subprocess.run(['/usr/bin/ldd',executable],capture_output=True,text=True,timeout=30);require(type(row.returncode)is int and row.returncode==0,'Current dependency census failed');paths=[];missing=[]
 for line in row.stdout.splitlines():
  if 'not found'in line:
   match=re.fullmatch(r'\s*(\S+)\s+=>\s+not found\s*',line);require(match is not None,'Unclassified unresolved loader diagnostic');missing.append(match[1])
  for token in line.split():
   if token.startswith('/')and Path(token).is_file():paths.append(str(Path(token).resolve()))
 require(paths and not missing,'Current dependency census empty or unresolved');files=[file_binding(p,512<<20)for p in sorted(set(paths))]
 return {'ldd_command':['/usr/bin/ldd',executable],'ldd_return_code':row.returncode,'ldd_stdout':row.stdout,'ldd_stderr':row.stderr,'files':files,'actual_dynamic_metadata':[],'explicit_auxiliary_scope':None}

def runtime_projection(value):
 result=dict(value)
 for key in('helper_libraries','tool_libraries','injection_libraries'):
  row=value[key];result[key]={**row,'ldd_stdout':re.sub(r'\(0x[0-9a-fA-F]+\)','(ADDRESS)',row['ldd_stdout'])}
 return result
def runtime_equal(before,after):require(runtime_projection(before)==runtime_projection(after),'Current helper/tool/library identity changed; ASLR addresses are descriptive raw fields only')
def inside(arm):
 out=Path('/results');(out/'build').mkdir();require(arm in('off','on'),'Exact capability arm required')
 before={'source_sha256':sha(__file__),'python_version':sys.version,'helper':file_binding('/leaf/indexer-materialized40'),'helper_libraries':libraries('/leaf/indexer-materialized40'),'tool':file_binding(TOOL),'tool_libraries':libraries(TOOL),'injection':file_binding(TOOL_LIBRARY),'injection_libraries':libraries(TOOL_LIBRARY)}
 require(before['tool']['sha256']==TOOL_SHA and before['injection']['sha256']==TOOL_LIBRARY_SHA,'Actual inventoried Unitrace differs');write(out/'runtime-before.json',before)
 for label,args in [('version',[TOOL,'--version']),('help',[TOOL,'--help'])]:
  started=time.time();write(out/('unitrace-'+label+'.command.json'),args)
  with (out/('unitrace-'+label+'.stdout')).open('wb')as stdout,(out/('unitrace-'+label+'.stderr')).open('wb')as stderr:
   row=subprocess.run(args,stdout=stdout,stderr=stderr,timeout=30)
  write(out/('unitrace-'+label+'.receipt.json'),{'command':args,'started_epoch':started,'finished_epoch':time.time(),'return_code':row.returncode,'stdout':file_binding(out/('unitrace-'+label+'.stdout')),'stderr':file_binding(out/('unitrace-'+label+'.stderr'))})
  require(row.returncode==0,'Unitrace metadata command failed '+label)
 require(all(k in(out/'unitrace-help.stdout').read_text()for k in SUPPORTED),'Actual current help lacks declared options')
 (out/'unitrace').mkdir()
 command=helper_argv()if arm=='off'else tool_argv();write(out/'leaf.command.json',command);started=time.time()
 with(out/'leaf.stdout').open('wb')as stdout,(out/'leaf.stderr').open('wb')as stderr:
  row=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=420)
 write(out/'leaf.receipt.json',{'command':command,'started_epoch':started,'finished_epoch':time.time(),'return_code':row.returncode,'stdout':file_binding(out/'leaf.stdout'),'stderr':file_binding(out/'leaf.stderr')})
 after={'source_sha256':sha(__file__),'python_version':sys.version,'helper':file_binding('/leaf/indexer-materialized40'),'helper_libraries':libraries('/leaf/indexer-materialized40'),'tool':file_binding(TOOL),'tool_libraries':libraries(TOOL),'injection':file_binding(TOOL_LIBRARY),'injection_libraries':libraries(TOOL_LIBRARY)};write(out/'runtime-after.json',after);runtime_equal(before,after);require(row.returncode==0,'Capability execution failed; outputs do not qualify')
 return 0

def recipe(arm,out,name,source):
 gid=str(Path('/dev/dri/renderD128').stat().st_gid)
 command=['docker','create','--name',name,'--label','b70.unitrace.indexer40='+source,'--network','none','--device','/dev/dri','--group-add',gid,'--user',str(os.getuid())+':'+str(os.getgid()),'--memory','4g','--memory-swap','4g','--cpus','2','--pids-limit','512','--entrypoint','/bin/bash']
 for p,target in[(HELPER,'/leaf/indexer-materialized40'),(FIXTURE,'/inputs'),(Path(__file__).resolve(),'/harness/strata/flash-next/probe.py'),(out,'/results'),(INSTALL,'/tool')]:command+=['-v',str(p)+':'+target+(':rw'if target=='/results'else ':ro')]
 for k,v in sorted({'ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','ZE_AFFINITY_MASK':'0','SYCL_CACHE_PERSISTENT':'0','SYCL_UR_TRACE':'0'}.items()):command+=['-e',k+'='+v]
 return command+[IMAGE,'-c','set -e; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; exec /opt/b70-c1-python/bin/python /harness/strata/flash-next/probe.py --inside '+arm]
def inspected(identity):
 row=subprocess.run(['docker','inspect',identity],capture_output=True,text=True,timeout=20)
 require(row.returncode==0,'Actual owned inspection unavailable; never infer absence');objects=json.loads(row.stdout);require(len(objects)==1,'Unique container required');return objects[0]
def owned(obj,command,identity):
 c,h=obj['Config'],obj['HostConfig'];require(obj['Id']==identity and obj['Name']=='/'+command[command.index('--name')+1]and c['Image']==IMAGE and c['Labels'].get('b70.unitrace.indexer40')==command[command.index('--label')+1].split('=',1)[1],'Foreign capability actor')
 binds=[command[i+1]for i,v in enumerate(command)if v=='-v'];require(len(binds)==len(set(binds))==5 and len(h['Binds'])==5 and sorted(h['Binds'])==sorted(binds),'Exact mount bijection required')
 require(c['Entrypoint']==['/bin/bash']and c['Cmd']==command[-2:]and c['User']==command[command.index('--user')+1]and c.get('WorkingDir','')==''and h['NetworkMode']=='none'and h['Privileged']is False and not h.get('DeviceRequests')and h['GroupAdd']==[command[command.index('--group-add')+1]],'Actual config/owner recipe differs')
 require(all(type(h[k])is int for k in('Memory','MemorySwap','NanoCpus','PidsLimit'))and h['Memory']==h['MemorySwap']==4<<30 and h['NanoCpus']==2_000_000_000 and h['PidsLimit']==512 and h['Devices']==[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}],'Actual resources/devices differ')
 env=dict(x.split('=',1)for x in c['Env']);require(all(env.get(command[i+1].split('=',1)[0])==command[i+1].split('=',1)[1]for i,v in enumerate(command)if v=='-e'),'Actual pin/runtime environment differs');return obj

def normal_terminal(obj):
 state=obj['State'];require(type(obj['RestartCount'])is int and obj['RestartCount']==0 and type(state['ExitCode'])is int and state['ExitCode']==0 and type(state['Pid'])is int and state['Pid']==0 and state['Status']=='exited'and state['Error']==''and all(state[k]is False for k in('Running','OOMKilled','Dead','Paused','Restarting')),'Exact typed normal terminal without restart required')

def run_command(command,out,label,timeout=210,cleanup=False):
 started=time.time();write(out/(label+'.command.json'),command)
 with(out/(label+'.stdout')).open('wb')as stdout,(out/(label+'.stderr')).open('wb')as stderr:
  proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True);deadline=time.monotonic()+timeout
  try:
   while proc.poll()is None:
    if(STOP and not cleanup)or time.monotonic()>deadline:raise TimeoutError('Interrupted/deadline '+label)
    time.sleep(.1)
  finally:
   if proc.poll()is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 row={'command':command,'started_epoch':started,'finished_epoch':time.time(),'return_code':proc.returncode,'stdout':file_binding(out/(label+'.stdout'),32<<20),'stderr':file_binding(out/(label+'.stderr'),32<<20)};write(out/(label+'.receipt.json'),row);require(proc.returncode==0,'Command failed '+label);return row
def health(out,stage):
 rows=[run_command([str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],out,stage+'-strict',180),run_command([str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'],out,stage+'-compiled',210)]
 health_text(rows)
 return rows
def health_text(rows):
 strict=Path(rows[0]['stderr']['path']).read_text();compiled=Path(rows[1]['stdout']['path']).read_text();require(all(x in strict for x in('card 0: OK','card 1: OK','xpu-health: HEALTHY (cards 0 1)'))and re.search(r'COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10 p2p=0\b',compiled)and not re.search(r'HEALTH_FAIL|COLLECTIVE_HEALTH_FAIL|WEDGED|INCONCLUSIVE',strict+compiled),'Actual strict/compiled P2P0 health sentinels absent or failed')
def kernel(out,stage,started):
 row=run_command(['journalctl','-k','--since','@'+str(int(started)),'--no-pager'],out,stage+'-kernel',20);require(not re.search(FAULT,Path(row['stdout']['path']).read_text(),re.I),'Kernel fault');return row

def arm_run(arm,out,source):
 out.mkdir();name='b70-unitrace-idx40-'+str(os.getpid())+'-'+arm;command=recipe(arm,out,name,source);write(out/'create.command.json',command);identity=None;failure=None
 try:
  row=run_command(command,out,'create',60);identity=Path(row['stdout']['path']).read_text().strip();require(re.fullmatch('[0-9a-f]{64}',identity),'Actual created ID required');write(out/'created-inspection.json',owned(inspected(identity),command,identity))
  run_command(['docker','start','-a',identity],out,'start',480)
 except BaseException as error:failure=type(error).__name__+': '+str(error)
 finally:
  # Removal is never attempted on a name or unknown owner. Any recovery fails
  # normal qualification and retains exclusion until ID terminal+removal proved.
  if identity is None and(out/'create.stdout').exists():
   value=(out/'create.stdout').read_text().strip()
   if re.fullmatch('[0-9a-f]{64}',value):identity=value
  while identity is None:
   # A timed-out creation is not absence, even after its CLI exits. Retain
   # exclusion until an exact-created owner appears or root resolves transition.
   failure=failure or 'Unresolved create transition requires recovery'
   write(out/'cleanup-pending.json',{'error':failure,'name':name})
   try:
    candidate=inspected(name);value=candidate['Id'];owned(candidate,command,value);identity=value
   except BaseException:time.sleep(1)
  removal=None;state=None
  while True:
   try:
    if removal is None:
     obj=owned(inspected(identity),command,identity)
     if obj['State']['Running']:
      failure=failure or 'Recovery stop required';run_command(['docker','stop','-t','30',identity],out,'recovery-stop',50,cleanup=True);obj=owned(inspected(identity),command,identity)
     require(obj['State']['Running']is False,'Owned actor still running');write(out/'terminal-inspection.json',obj)
     run_command(['docker','logs',identity],out,'terminal-logs',30,cleanup=True)
     state=obj['State'];removal=run_command(['docker','rm',identity],out,'remove',30,cleanup=True)
    absent=subprocess.run(['docker','inspect',identity],capture_output=True,text=True,timeout=20)
    require(absent.returncode==1 and re.fullmatch(r'(?is)\s*(?:Error response from daemon:|error:)\s*no such (?:object|container):\s*'+identity+r'\s*',absent.stderr),'Exact ID absence not proven')
    write(out/'absence.json',{'return_code':absent.returncode,'stdout':absent.stdout,'stderr':absent.stderr,'epoch':time.time()});break
   except BaseException as error:
    failure=failure or 'Retirement: '+type(error).__name__+': '+str(error);write(out/'cleanup-pending.json',{'error':failure,'identity':identity});time.sleep(1)
 require(failure is None,'Non-normal capability actor: '+str(failure));normal_terminal(obj);return state

def output_binding(root):
 root=Path(root);native=root/'build/native-output';require({p.name for p in native.iterdir()}=={'input_echo',*(m+'-route'+str(r)for m in MODES for r in range(3))},'Complete18-frame roster required');result={}
 for mode in MODES:
  routes=[]
  for route in range(3):
   path=native/(mode+'-route'+str(route));require({p.name for p in path.iterdir()}==set(FIELDS),'Exact8-field frame roster');rows={name:file_binding(path/name,size)['sha256']for name,size in FIELDS.items()};require(all((path/name).stat().st_size==size for name,size in FIELDS.items()),'Exact output extents');routes.append(rows);result[mode+'-route'+str(route)]=rows
  require(routes[0]==routes[1]==routes[2],'Direct/two replay output differs')
 for name in('raw.f32','gamma.f32','config.txt'):require(sha(native/'input_echo'/name)==sha(FIXTURE/name),'Actual consumed input echo differs')
 text=(root/'leaf.stdout').read_text()+'\n'+(root/'leaf.stderr').read_text();markers=[s for f in ('leaf.stdout','leaf.stderr')for s in(root/f).read_text().splitlines(keepends=True)if s.startswith('IDXMAT40_')];require(all(s.endswith('\n')for s in markers)and len(markers)==22,'Complete original marker roster');frames=[f'IDXMAT40_FRAME mode={m} route={r} graph_replay={r} fields=8 own_zero_restored=1\n'for m in MODES for r in range(3)];require(re.fullmatch(r'IDXMAT40_DEVICE backend=level_zero affinity=0 fp16=1 vendor=.+ driver=.+ name=.+\n',markers[0])and re.fullmatch(r'IDXMAT40_FP_CONFIG flags=(?:[0-9]+,)+ observed_device_flags_only=1 compiler_lowering_unobserved=1\n',markers[1])and markers[2]=='IDXMAT40_CONFIG actual_resolved_checked=1 rows=4 dim=128 windows=2,1,1 source_functions_unchanged=1\n'and markers[3:21]==frames and markers[-1]=='IDXMAT40_RESULT frames=18 graph_retired=1 owned_allocations_freed=1 full_model_math_qualified=0 semantic_model_fix_qualified=0\n'and 'IDXMAT40_ERROR'not in text,'Original graph/free/output roster differs');runtime_equal(read(root/'runtime-before.json'),read(root/'runtime-after.json'));return result

def artifacts(root):
 root=Path(root);result={};total=0
 for p in sorted(root.rglob('*')):
  require(not p.is_symlink(),'Artifact aliases refused')
  if p.is_file():
   row=file_binding(p,4<<30);total+=row['bytes'];require(total<=8<<30 and len(result)<30000,'Bounded full profiler evidence roster exceeded');result[str(p.relative_to(root))]=row
 return result

def receipt_binding(root,label,expected,reported=None):
 root=Path(root);row=read(root/(label+'.receipt.json'));require(row['command']==expected==read(root/(label+'.command.json'))and type(row['return_code'])is int and row['return_code']==0,'Exact successful command recipe required')
 require(all(type(row[k])in(int,float)and math.isfinite(row[k])for k in('started_epoch','finished_epoch'))and row['started_epoch']<=row['finished_epoch'],'Typed actual command chronology')
 if reported is not None:require(row==reported,'Original report/receipt row differs')
 for stream in('stdout','stderr'):require(row[stream]==file_binding(root/(label+'.'+stream),32<<20),'Original full command EOF bytes differ')
 return row

def unitrace_metadata_binding(path,started,leaf_started):
 path=Path(path);last=started
 for label,argv in(('version',[TOOL,'--version']),('help',[TOOL,'--help'])):
  row=read(path/('unitrace-'+label+'.receipt.json'));require(row['command']==argv==read(path/('unitrace-'+label+'.command.json'))and type(row['return_code'])is int and row['return_code']==0 and all(type(row[k])in(int,float)and math.isfinite(row[k])for k in('started_epoch','finished_epoch'))and last<=row['started_epoch']<=row['finished_epoch']<=leaf_started,'Original actual Unitrace metadata recipe/normal chronology differs');last=row['finished_epoch']
  for stream in('stdout','stderr'):
   raw=file_binding(path/('unitrace-'+label+'.'+stream));require(typed_equal(row[stream],{**raw,'path':'/results/unitrace-'+label+'.'+stream}),'Original actual Unitrace metadata output SHA/path differs')
 require(all(k in(path/'unitrace-help.stdout').read_text()for k in SUPPORTED)and (path/'unitrace-version.stdout').read_bytes()==b'2.4.0 ()\n','Original actual version/help differs; empty git stamp is informational only')

def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();report=read(root/'report.json');require(report['passed']is True and report['errors']==[]and report['source_plan_sha256']==source and read(root/'source-plan.snapshot.json')==read(PLAN),'Actual successful source capability report required');require(report['artifact_sha256']=={k:v['sha256']for k,v in artifacts(root).items()if k!='report.json'},'Whole current evidence roster/hash differs')
 require(report['CPU_identity_spoofing_used']is False and typed_equal(report['real_host_visible_cpu_identity'],file_binding(root/'real-host-cpuinfo.txt'))and report['CPU_identity_scope']=='original host-visible proc metadata, not bare-metal identity qualification','Actual original CPU provenance changed or spoofing claimed')
 require(output_binding(root/'off')==output_binding(root/'on')and report['OFF_ON_all18frames_bitwise_equal']is True,'Actual OFF/ON output equality differs');require(report['profiler_raw_result_roster']and typed_equal(report['profiler_raw_result_roster'],artifacts(root/'on/unitrace')),'Original nonempty full profiler raw result changed')
 for arm in('off','on'):
  path=root/arm;command=read(path/'create.command.json');require(type(report['producer_pid'])is int and report['producer_pid']>0 and command==recipe(arm,path,'b70-unitrace-idx40-'+str(report['producer_pid'])+'-'+arm,source),'Exact producer-owned recipe differs');obj=read(path/'terminal-inspection.json');owned(obj,command,obj['Id']);normal_terminal(obj);require(typed_equal(report['arms'][arm],obj['State'])and not(path/'cleanup-pending.json').exists(),'Original report/terminal State or recovery scope differs');owned(read(path/'created-inspection.json'),command,identity=read(path/'created-inspection.json')['Id']);require(read(path/'created-inspection.json')['Id']==obj['Id'],'Created/terminal ID differs')
  for label in('create','start','terminal-logs','remove'):
   row=read(path/(label+'.receipt.json'));require(row['return_code']==0 and row['command']==read(path/(label+'.command.json')),'Actual original command receipt differs')
   for stream in('stdout','stderr'):require(file_binding(row[stream]['path'])==row[stream],'Actual original command bytes differ')
  identity=obj['Id'];require(Path(read(path/'create.receipt.json')['stdout']['path']).read_text().strip()==identity,'Actual ID/creation crossjoin differs')
  expected=[command,['docker','start','-a',identity],['docker','logs',identity],['docker','rm',identity]];rows=[receipt_binding(path,l,c)for l,c in zip(('create','start','terminal-logs','remove'),expected)];require(all(rows[i]['finished_epoch']<=rows[i+1]['started_epoch']for i in range(3)),'Actor EOF/logs/removal chronology differs')
  absence=read(path/'absence.json');require(absence['return_code']==1 and re.fullmatch(r'(?is)\s*(?:Error response from daemon:|error:)\s*no such (?:object|container):\s*'+identity+r'\s*',absence['stderr'])and rows[-1]['finished_epoch']<=absence['epoch'],'Original exact ID removal/absence missing');require(read(path/'leaf.command.json')==(helper_argv()if arm=='off'else tool_argv()),'Actual leaf recipe differs')
  leaf=read(path/'leaf.receipt.json');require(leaf['command']==read(path/'leaf.command.json')and type(leaf['return_code'])is int and leaf['return_code']==0 and rows[1]['started_epoch']<=leaf['started_epoch']<=leaf['finished_epoch']<=rows[1]['finished_epoch'],'Actual helper/profiler completion differs')
  for stream in('stdout','stderr'):
   actual=file_binding(path/('leaf.'+stream));require(leaf[stream]=={**actual,'path':'/results/leaf.'+stream},'Exact in-image helper/profiler output bytes differ')
  runtime=read(path/'runtime-before.json');require(runtime['source_sha256']==sha(__file__)and runtime['helper']['sha256']==sha(HELPER)and runtime['tool']['sha256']==TOOL_SHA and runtime['injection']['sha256']==TOOL_LIBRARY_SHA,'Current consumed helper/source/tool identity differs');dependency_record_binding('/leaf/indexer-materialized40',runtime['helper_libraries']);dependency_record_binding(TOOL,runtime['tool_libraries']);dependency_record_binding(TOOL_LIBRARY,runtime['injection_libraries'])
  unitrace_metadata_binding(path,rows[1]['started_epoch'],leaf['started_epoch'])

 for stage in('pre','post'):
  expected=[[str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],[str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'],['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager']]
  values=report[stage+'_health']+[report[stage+'_kernel']];rows=[receipt_binding(root,label,c,r)for label,c,r in zip((stage+'-strict',stage+'-compiled',stage+'-kernel'),expected,values)];require(rows[0]['finished_epoch']<=rows[1]['started_epoch']<=rows[1]['finished_epoch']<=rows[2]['started_epoch'],'Sequential original health/kernel chronology differs')
  health_text(rows[:2])
  require(not re.search(FAULT,(root/(stage+'-kernel.stdout')).read_text(),re.I),'Current original kernel log contains fault')
 require(report['pre_kernel']['finished_epoch']<=read(root/'off/create.receipt.json')['started_epoch']and read(root/'off/absence.json')['epoch']<=read(root/'on/create.receipt.json')['started_epoch']and read(root/'on/absence.json')['epoch']<=report['post_health'][0]['started_epoch']<=report['post_kernel']['finished_epoch']<=report['finished_epoch'],'Actual prehealth/OFF/ON/retirement/posthealth chronology differs')
 require(all(report[k]is False for k in('device_timeline_coverage_qualified','timestamp_calibration_qualified','zero_added_waits_proven','model_math_qualified','serving_latency_qualified')),'Capability cannot claim unsupported timeline/calibration/model proof')
 require(source_binding()==source,'Current source/helper/fixture changed during recollection');return {'fixture_conditional_OFF_ON_all18frames_bitwise_equal':True,'all_original_profiler_files_preserved':True,'device_timeline_coverage_qualified':False,'timestamp_calibration_qualified':False,'zero_added_waits_proven':False,'model_math_qualified':False,'serving_latency_qualified':False}

def main():
 if '--inside-inventory'in sys.argv:
  value={'schema':1,'source_sha256':sha(__file__),'helper':file_binding('/leaf/indexer-materialized40'),'helper_libraries':libraries('/leaf/indexer-materialized40'),'tool':file_binding(TOOL),'tool_libraries':libraries(TOOL),'injection':file_binding(TOOL_LIBRARY),'injection_libraries':libraries(TOOL_LIBRARY),'GPU_collection_started':False,'helper_executed':False};require(value['tool']['sha256']==TOOL_SHA and value['injection']['sha256']==TOOL_LIBRARY_SHA,'Actual tool differs');print(json.dumps(value,indent=2,ensure_ascii=True));return 0
 if '--inside-bootstrap'in sys.argv:
  print(json.dumps({'stdlib_only_imported':True,'source_sha256':sha(__file__),'root':str(ROOT),'device_query_performed':False}));return 0
 if '--inside'in sys.argv:return inside(sys.argv[sys.argv.index('--inside')+1])
 require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),'Ordinary Python required');p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--leased',action='store_true');a=p.parse_args();source=source_binding()
 if not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 # Model-free leaf is pinned0; pair reservation authorizes required compiled
 # two-card health, which must never run under a card0-only exclusion.
 for fd,card in((8,0),(9,1)):require(os.path.samefile('/proc/self/fd/'+str(fd),'/mnt/vm_8tb/b70/gpu.lock.'+str(card)),'Owned pair health exclusion required')
 a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);write(a.output/'source-plan.snapshot.json',read(PLAN));report={'schema':1,'producer_pid':os.getpid(),'started_epoch':time.time(),'passed':False,'source_plan_sha256':source,'scope':'fixture-conditional profiler capability only','workload_cards':[0],'reserved_health_cards':[0,1],'errors':[],'CPU_identity_spoofing_used':False,'device_timeline_coverage_qualified':False,'timestamp_calibration_qualified':False,'zero_added_waits_proven':False,'model_math_qualified':False,'serving_latency_qualified':False};global STOP
 with Path('/proc/cpuinfo').open('rb')as f:cpu=f.read((1<<20)+1)
 require(len(cpu)<=1<<20,'Bounded host-visible CPU identity required');(a.output/'real-host-cpuinfo.txt').write_bytes(cpu);report['real_host_visible_cpu_identity']=file_binding(a.output/'real-host-cpuinfo.txt');report['CPU_identity_scope']='original host-visible proc metadata, not bare-metal identity qualification'
 def stop(*args):
  global STOP;STOP=True
 for s in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
 try:
  report['pre_health']=health(a.output,'pre');report['pre_kernel']=kernel(a.output,'pre',report['started_epoch']);report['arms']={arm:arm_run(arm,a.output/arm,source)for arm in('off','on')};off=output_binding(a.output/'off');on=output_binding(a.output/'on');require(off==on,'Unitrace changed actual18-frame8-field outputs');require((a.output/'on/unitrace').is_dir()and any((a.output/'on/unitrace').iterdir()),'Actual profiler raw result absent');report['OFF_ON_all18frames_bitwise_equal']=True;report['profiler_raw_result_roster']=artifacts(a.output/'on/unitrace')
 except BaseException as error:report['errors'].append(type(error).__name__+': '+str(error))
 finally:
  STOP=False
  try:report['post_health']=health(a.output,'post');report['post_kernel']=kernel(a.output,'post',report['started_epoch']);require(source_binding()==source,'Probe source/fixture changed');require(read(a.output/'source-plan.snapshot.json')==read(PLAN),'Source snapshot differs')
  except BaseException as error:report['errors'].append('post: '+type(error).__name__+': '+str(error))
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['artifact_sha256']={k:v['sha256']for k,v in artifacts(a.output).items()};write(a.output/'report.json',report)
 require(report['passed'],'Capability run failed: '+str(report['errors']));return 0
if __name__=='__main__':raise SystemExit(main())
