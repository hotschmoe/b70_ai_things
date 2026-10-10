#!/usr/bin/env python3
"""Root-executed passive CPU metadata observer; never changes memory guards."""
import argparse,hashlib,json,math,os,re,subprocess,time,signal,stat
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
import qualify_api_positive_overlap_cpu_screen_v3 as screen
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SCREEN_PLAN=HERE/'api-positive-overlap-cpu-screen-source-plan-v3.json'
SCREEN_SHA='a3de902172c24fb11a7c1c09ac1a096a6e7cdfa671403901ae58dd5c830c6492'
MAX_PIDS=4096;MAX_SWAP_ROWS=256;MAX_SAMPLES=7201

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
 with Path(path).open('x') as handle:handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def source_binding():
 require(sha(SCREEN_PLAN)==SCREEN_SHA,'Unchanged e633 screen recipe required');plan=read_unique(SCREEN_PLAN)
 require(sha(Path(screen.__file__))==plan['runner_sha256'],'Frozen screen runner differs')
 for name,digest in plan['source_bindings'].items():require(sha(ROOT/name)==digest,'Frozen complete screen source differs '+name)
 return plan,{'screen_plan_sha256':SCREEN_SHA,'screen_runner_sha256':plan['runner_sha256'],'observer_sha256':sha(__file__)}
def kv_bytes(text,keys):
 rows={}
 for line in text.splitlines():
  if ':' not in line:continue
  key,value=line.split(':',1)
  if key in keys:
   fields=value.split();require(len(fields)==2 and fields[1]=='kB','Expected kernel kB field '+key);rows[key]=int(fields[0])*1024
 require(set(rows)==set(keys),'Missing kernel byte fields');return rows

def process_metadata(proc,pid):
 base=Path(proc)/str(pid);text=(base/'stat').read_text();end=text.rfind(')');fields=text[end+2:].split();require(end>=0 and len(fields)>=20,'Malformed process stat');start=int(fields[19])
 status=kv_bytes((base/'status').read_text(),('VmSwap',))
 # comm only: no environment, command line, open-file list or model bytes.
 return {'pid':pid,'start_ticks':start,'comm':(base/'comm').read_text().strip(),'VmSwap':status['VmSwap']}

def global_sample(proc='/proc',previous=None):
 start=time.time();base=Path(proc);host=kv_bytes((base/'meminfo').read_text(),('MemAvailable','SwapFree','SwapTotal'));vm=dict(line.split() for line in (base/'vmstat').read_text().splitlines());host.update(pswpin=int(vm['pswpin']),pswpout=int(vm['pswpout']))
 pids=sorted(int(p.name) for p in base.iterdir() if p.name.isdigit());require(len(pids)<=MAX_PIDS,'PID census exceeds declared bound');rows=[];errors=[];current={}
 for pid in pids:
  try:r=process_metadata(base,pid)
  except FileNotFoundError:errors.append({'pid':pid,'error':'vanished_during_census'});continue
  except PermissionError:errors.append({'pid':pid,'error':'PermissionError'});continue
  # Kernel threads lack VmSwap: represented as unreadable coverage, never zero.
  except ValueError:errors.append({'pid':pid,'error':'missing_or_malformed_VmSwap'});continue
  key=(pid,r['start_ticks']);current[key]=r['VmSwap'];old=previous['processes'] if previous is not None else {};r['VmSwap_delta']=None if key not in old else r['VmSwap']-old[key]
  if r['VmSwap'] or r['VmSwap_delta']:rows.append(r)
 require(len(rows)<=MAX_SWAP_ROWS,'Swapped PID rows exceed declared bound')
 deltas={k:None if previous is None else host[k]-previous['host'][k] for k in ('pswpin','pswpout','SwapFree','MemAvailable')}
 return {'started_epoch':start,'finished_epoch':time.time(),'host':host,'host_delta':deltas,'host_page_bytes':os.sysconf('SC_PAGE_SIZE'),'pid_census_count':len(pids),'swap_processes':rows,'process_coverage_errors':errors,'all_enumerated_user_process_status_read':not errors},{'processes':current,'host':host}

def owned_sample(obj,command,plan,proc='/proc',cgroup_root='/sys/fs/cgroup'):
 name=command[command.index('--name')+1];require(screen.owned(obj,name,plan['image'],plan['runner_sha256']),'Foreign CPU container refused');screen.runtime_recipe_gate(obj,command)
 require(obj['State']['Running'] is True and type(obj['State']['Pid'])is int and obj['State']['Pid']>0,'Live exact owned PID required');pid=obj['State']['Pid'];base=Path(proc)/str(pid);before=process_metadata(proc,pid);rollup=kv_bytes((base/'smaps_rollup').read_text(),('Rss','Pss','Private_Clean','Private_Dirty','Shared_Clean','Shared_Dirty','Anonymous','Swap','SwapPss'))
 cg=next(line.split('::',1)[1] for line in (base/'cgroup').read_text().splitlines() if line.startswith('0::'));require(cg.startswith('/') and '..' not in Path(cg).parts,'Invalid cgroup path');cgpath=Path(cgroup_root)/cg.lstrip('/')
 values={name:(cgpath/name).read_text().strip() for name in ('memory.current','memory.peak','memory.max','memory.swap.current','memory.swap.max','memory.events','memory.stat')}
 require(int(values['memory.max'])==plan['memory_cap_bytes'] and int(values['memory.swap.max'])==0,'Unchanged owned CPU memory/swap cap required');after=process_metadata(proc,pid);require((before['pid'],before['start_ticks'])==(after['pid'],after['start_ticks']),'Owned PID reused while sampled')
 return {'container':name,'image':obj['Image'],'owner_label':plan['runner_sha256'],'pid':pid,'start_ticks':before['start_ticks'],'comm':before['comm'],'VmSwap':after['VmSwap'],'smaps_rollup_bytes':rollup,'cgroup_path':cg,'cgroup':values,'command_sha256':hashlib.sha256(canonical(command).encode()).hexdigest(),'devices_granted':False,'inspection':{'Name':obj['Name'],'Image':obj['Image'],'Config':{k:obj['Config'][k] for k in ('Image','Labels','Cmd','Entrypoint','User','WorkingDir')},'HostConfig':{k:obj['HostConfig'].get(k) for k in ('NetworkMode','Memory','MemorySwap','NanoCpus','PidsLimit','Devices','DeviceRequests','Privileged','GroupAdd')},'Mounts':obj['Mounts'],'State':obj['State']}}

def idle_owned_absence(plan):
 r=subprocess.run(['docker','ps','-aq','--filter','label=b70.api-overlap.cpu-screen='+plan['runner_sha256']],capture_output=True,text=True,timeout=10);require(r.returncode==0,'Idle owned-container census failed');running=[]
 for identity in r.stdout.split():
  obj=inspect(identity)
  if obj is not None and obj['State']['Running']:running.append(obj['Name'])
 require(not running,'Model-free idle window has live owned CPU screen container');return {'owned_screen_running':running,'census_epoch':time.time()}

def inspection_result(name, result):
 require(type(name)is str and name and type(result.returncode)is int,'Typed exact Docker inspection required')
 if result.returncode:
  require(result.returncode==1 and result.stdout.strip()in ('','[]'),'Docker absence requires exact rc1/empty result')
  require(re.fullmatch(r'(?:Error response from daemon: |[Ee]rror: )?[Nn]o such object: '+re.escape(name)+r'\s*',result.stderr) is not None,'Unknown Docker inspection error is not absence')
  return None
 require(not result.stderr.strip(),'Successful inspect has unexpected stderr')
 objects=json.loads(result.stdout);require(type(objects)is list and len(objects)==1 and type(objects[0])is dict and objects[0].get('Name')=='/'+name,'Exact named container inspection required');return objects[0]
def inspect(name):
 return inspection_result(name,subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10))
def expected_command(path,plan,pid):
 command=read_unique(path);name=command[command.index('--name')+1];case=path.parent.name;match=re.fullmatch(r'case([0-7])-repeat([01])',case);require(match is not None,'Bounded preregistered case path required');require(name=='b70-cpu-overlap-screen-'+str(pid)+'-'+match[1]+'-'+match[2],'Exact producer PID/case container name required')
 lock=read_unique(HERE/'model-lock.json');shards=[ROOT/lock['destination']/row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')];require(len(shards)==4,'Full original shard roster metadata required')
 expected=screen.server_command(name,path.parent,plan,Path(plan['build_root']),shards,Path(screen.__file__).resolve(),None);require(canonical(command)==canonical(expected),'Unchanged exact screen CPU command required');return command

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('idle','observe'));p.add_argument('--output',type=Path,required=True);p.add_argument('--screen-root',type=Path);p.add_argument('--producer-pid',type=int);p.add_argument('--idle-receipt',type=Path);p.add_argument('--seconds',type=int,default=30);a=p.parse_args();require(not a.output.exists(),'New immutable observer output required');plan,binding=source_binding()
 require(1<=a.seconds<=7200 and (a.mode!='idle' or a.seconds==30),'Idle baseline exactly30s; bounded runtime required');require(a.mode=='idle' or a.screen_root is not None and a.producer_pid is not None and a.producer_pid>0 and a.idle_receipt is not None,'Exact runtime associations required')
 # Exclusion lease is CPU-only and does not grant or enumerate devices.
 for card in (0,1):require(os.path.samefile('/proc/self/fd/'+str(8+card),'/mnt/vm_8tb/b70/gpu.lock.'+str(card)),'Exclusive inherited CPU pair exclusion lease required')
 idle=None
 if a.mode=='observe':
  require(a.idle_receipt.name=='report.json','Exact idle report path required');finalized_binding(a.idle_receipt.parent);idle=read_unique(a.idle_receipt);require(idle['mode']=='idle' and idle['passed']is True and idle['seconds']==30 and idle['source_binding']==binding,'Fresh matching complete idle baseline required');require(0<=time.time()-idle['finished_epoch']<=300,'Idle baseline stale')
 observer_plan=HERE/'cpu-swap-attribution-finite-source-plan-v6.json';observer_raw=observer_plan.read_bytes();observer_sha=hashlib.sha256(observer_raw).hexdigest();observer=read_unique(observer_plan)
 for name,digest in observer['files'].items():require(sha(ROOT/name)==digest,'Observer current source changed '+name)
 a.output.mkdir();(a.output/'observer-source-plan.snapshot.json').write_bytes(observer_raw);write_new(a.output/'source-plan.snapshot.json',read_unique(SCREEN_PLAN));start=time.time();deadline=time.monotonic()+a.seconds;previous=None;count=0;errors=[];owned_count=0;stopping=[]
 for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda sig,frame:stopping.append(sig))
 try:
  while True:
   sample,previous=global_sample(previous=previous);sample.update(index=count,owned=[])
   if a.mode=='idle':sample['idle_owned_absence']=idle_owned_absence(plan)
   if a.mode=='observe':
    for path in sorted(a.screen_root.glob('case*-repeat*/command.json')):
     command=expected_command(path,plan,a.producer_pid);obj=inspect(command[command.index('--name')+1])
     if obj is not None and obj['State']['Running']:
      row=owned_sample(obj,command,plan);row.update(command_path=str(path.resolve()),command_file_sha256=sha(path));sample['owned'].append(row);owned_count+=1
   sample['finished_epoch']=time.time();require(count<MAX_SAMPLES,'Sample bound exceeded');write_new(a.output/('sample-'+str(count).zfill(5)+'.json'),sample);count+=1
   if stopping or time.monotonic()>=deadline:break
   time.sleep(min(1,max(0,deadline-time.monotonic())))
 except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
 final=source_binding()[1];require(final==binding and observer_plan.read_bytes()==observer_raw,'Observer/screen source changed')
 for name,digest in observer['files'].items():require(sha(ROOT/name)==digest,'Observer final current source changed '+name)
 report={'schema':1,'mode':a.mode,'passed':not errors and count>=2 and (a.mode=='idle' and time.monotonic()>=deadline or a.mode=='observe' and owned_count>0),'seconds':a.seconds,'duration_completed':time.monotonic()>=deadline,'stop_signals':stopping,'observer_source_plan_sha256':observer_sha,'started_epoch':start,'finished_epoch':time.time(),'source_binding':binding,'sample_count':count,'owned_sample_count':owned_count,'errors':errors,'screen_root':str(a.screen_root.resolve()) if a.screen_root else None,'producer_pid':a.producer_pid,'idle_receipt_path':str(a.idle_receipt.resolve()) if a.idle_receipt else None,'idle_receipt_sha256':sha(a.idle_receipt) if a.idle_receipt else None,'sample_sha256':{p.name:sha(p) for p in sorted(a.output.glob('sample-*.json'))},'inference_settings_changed':False,'memory_guards_changed':False,'actual_GPU_touch':False,'model_payload_read':False,'causal_attribution_qualified':False,'screen_success_transferred':False};write_new(a.output/'report.json',report);print(json.dumps({'passed':report['passed'],'samples':count,'diagnostic_only':True}));return 0 if report['passed']else 1

def exact_artifacts(root,names):
 require(root.is_dir() and not root.is_symlink(),'Actual regular nonsymlink observer root required')
 entries=list(root.iterdir());require({p.name for p in entries}==set(names),'Exact observer artifact roster differs')
 for path in entries:require(stat.S_ISREG(path.lstat().st_mode),'Observer artifact must be regular and nonsymlink')
 return {p.name:sha(p) for p in entries}

# Import-only read-only closure; no Docker/proc observation or inference.
def finalized_binding(root):
 root=Path(root).absolute();require(not root.is_symlink(),'Observer root symlink refused');report=read_unique(root/'report.json');plan,binding=source_binding();observer=HERE/'cpu-swap-attribution-finite-source-plan-v6.json'
 require(report.get('errors')==[] and report['mode'] in ('idle','observe'),'Observer errors/mode invalid')
 artifacts=exact_artifacts(root,{'report.json','observer-source-plan.snapshot.json','source-plan.snapshot.json',*report['sample_sha256']})
 observer_raw=observer.read_bytes();observer_closure=read_unique(observer)['files']
 require(report['schema']==1 and report['passed']is True and report['source_binding']==binding,'Actual observer report/source differs');require(sha(observer)==report['observer_source_plan_sha256']==sha(root/'observer-source-plan.snapshot.json'),'Current observer source plan differs')
 for name,digest in read_unique(observer)['files'].items():require(sha(ROOT/name)==digest,'Current observer closure differs '+name)
 require(sha(root/'source-plan.snapshot.json')==SCREEN_SHA,'Exact unchanged screen source plan snapshot required');require(report['memory_guards_changed']is False and report['inference_settings_changed']is False and report['actual_GPU_touch']is False and report['model_payload_read']is False and report['causal_attribution_qualified']is False,'Diagnostic scope changed')
 if report['mode']=='observe':
  idle_path=Path(report['idle_receipt_path']).resolve();require(idle_path.name=='report.json' and sha(idle_path)==report['idle_receipt_sha256'],'Exact original idle receipt path/hash differs');idle=read_unique(idle_path);require(idle['mode']=='idle' and idle['finished_epoch']<=report['started_epoch']<=idle['finished_epoch']+300,'Actual original idle/runtime chronology differs');finalized_binding(idle_path.parent)
 names=sorted(p.name for p in root.glob('sample-*.json'));require(names==sorted(report['sample_sha256']) and 2<=len(names)==report['sample_count']<=MAX_SAMPLES,'Exact complete bounded sample roster required');previous=None;owned=0;last=report['started_epoch']
 for index,name in enumerate(names):
  path=root/name;require(name=='sample-'+str(index).zfill(5)+'.json' and sha(path)==report['sample_sha256'][name],'Actual immutable sample bytes/index differ');row=read_unique(path);require(row['index']==index and all(type(row[k])in (int,float) and math.isfinite(row[k]) for k in ('started_epoch','finished_epoch')) and last<=row['started_epoch']<=row['finished_epoch']<=report['finished_epoch'],'Actual sample chronology differs');last=row['finished_epoch']
  for key in ('MemAvailable','SwapFree','SwapTotal','pswpin','pswpout'):require(type(row['host'][key])is int and row['host'][key]>=0,'Typed actual host counters required')
  require(row['host_delta']=={k:None if previous is None else row['host'][k]-previous[k] for k in ('pswpin','pswpout','SwapFree','MemAvailable')},'Actual host deltas differ');previous=row['host']
  require(len(row['swap_processes'])<=MAX_SWAP_ROWS and row['pid_census_count']<=MAX_PIDS,'Actual PID census bounds differ')
  if report['mode']=='idle':require(row['owned']==[] and row['idle_owned_absence']['owned_screen_running']==[] and row['started_epoch']<=row['idle_owned_absence']['census_epoch']<=row['finished_epoch'],'Idle had owned model activity/incorrect chronology')
  for item in row['owned']:
   path=Path(item['command_path']);require(path.parent.parent==Path(report['screen_root']).resolve() and sha(path)==item['command_file_sha256'],'Original current case command association differs');cmd=expected_command(path,plan,report['producer_pid']);obj=item['inspection'];require(screen.owned(obj,item['container'],plan['image'],plan['runner_sha256']) and obj['State']['Running']is True and obj['State']['Pid']==item['pid'],'Saved actual owned CPU identity differs');screen.runtime_recipe_gate(obj,cmd);require(item['command_sha256']==hashlib.sha256(canonical(cmd).encode()).hexdigest(),'Saved exact command content differs');owned+=1
 require(owned==report['owned_sample_count'] and (report['mode']=='idle' and report['seconds']==30 and report['duration_completed']is True and report['finished_epoch']-report['started_epoch']>=30 or report['mode']=='observe' and owned>0),'Actual idle/runtime scope incomplete')
 require(exact_artifacts(root,artifacts)==artifacts,'Observer artifacts changed during recursive validation')
 require(source_binding()[1]==binding and observer.read_bytes()==observer_raw,'Observer/screen source changed after recursive validation')
 for name,digest in observer_closure.items():require(sha(ROOT/name)==digest,'Final complete observer source differs '+name)
 return {'report_sha256':sha(root/'report.json'),'observer_source_plan_sha256':sha(observer),'sample_count':len(names),'owned_sample_count':owned,'diagnostic_only':True,'causal_attribution_qualified':False,'memory_guards_changed':False,'model_payload_read':False}

if __name__=='__main__':raise SystemExit(main())
