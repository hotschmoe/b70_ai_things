"""ROOT-only owned PID/cgroup/fdinfo metadata; never reads model/device payloads."""
import hashlib,json,time
from pathlib import Path
from full_cache_shared_history_v2 import require

MAX_RAW=65536
CGROUP_FILES=('memory.current','memory.peak','memory.max','memory.swap.current','memory.swap.max','memory.events','memory.stat')

def parse_stat(raw,pid):
 require(type(pid)is int and pid>0,'Actual positive host PID required');text=raw.decode('ascii');at=text.rfind(')');parts=text[at+2:].split();require(at>=0 and len(parts)>=20 and int(text.split(' ',1)[0])==pid,'Actual owned process stat incomplete')
 return {'pid':pid,'state':parts[0],'ppid':int(parts[1]),'start_ticks':int(parts[19])}

def pid_row(proc,pid):return parse_stat((Path(proc)/str(pid)/'stat').read_bytes(),pid)

def stable(a,b):return all(a[k]==b[k] for k in ('pid','ppid','start_ticks')) and b['state'] not in ('Z','X')

def numbers(raw):
 out={}
 for line in raw.decode('ascii').splitlines():
  words=line.split();require(len(words)==2 and words[0] not in out and words[1].isdigit(),'Exact unique kernel accounting integer fields required');out[words[0]]=int(words[1])
 return out

def recipe_binding(obj,command,image):
 h=obj['HostConfig'];c=obj['Config'];at=command.index(image)
 require(obj['Image']==c['Image']==image and c['Cmd']==command[at+1:] and c['Entrypoint']==['/bin/bash'] and c['User']==command[command.index('--user')+1],'Actual actor executable/user recipe differs')
 require(h['NetworkMode']==command[command.index('--network')+1] and h['Memory']==h['MemorySwap']==105*1024**3 and not h.get('Privileged') and not h.get('DeviceRequests'),'Actual actor network/resource scope differs')
 require(h['Devices']==[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}] and h['GroupAdd']==[command[i+1] for i,v in enumerate(command) if v=='--group-add'],'Actual actor devices/groups differ')
 binds=[command[i+1] for i,v in enumerate(command) if v=='-v'];require(h['Binds']==binds,'Actual actor declared binds differ')
 declared=sorted((a,b,m=='rw','bind') for a,b,m in (v.rsplit(':',2) for v in binds));actual=sorted((m['Source'],m['Destination'],m['RW'],m['Type']) for m in obj['Mounts']);require(actual==declared,'Actual actor complete mount roster differs')
 expected=[command[i+1] for i,v in enumerate(command) if v=='-e'];observed=c['Env'];require(type(observed)is list and len({v.split('=',1)[0] for v in observed})==len(observed) and all(v in observed for v in expected),'Actual actor explicit environment differs')
 return True

def capture(obj,command,plan,out,*,proc='/proc',cgroup='/sys/fs/cgroup'):
 """Call only inside the root-owned pair lease, with actual terminal-free actor."""
 from run_full_cache_shared_runtime_v3 import command_recipe
 from full_cache_shared_runtime_v3 import c1
 c1.leased([0,1]);out=Path(out);require(not out.exists(),'Fresh owned metadata sample output required')
 name=command[command.index('--name')+1];require(obj['Name']=='/'+name and obj['Image']==plan['image'] and obj['Config']['Image']==plan['image'] and obj['State']['Running'] is True and type(obj['State']['Pid'])is int and obj['State']['Pid']>0,'Exact live owned actor/container required')
 recipe_binding(obj,command,plan['image']);controller_pid=int(name.split('-')[2]);actor=out.parent.parent;require(command==command_recipe(plan,actor,controller_pid),'Exact original actor launch recipe required');digest=hashlib.sha256((actor/'plan.snapshot.json').read_bytes()).hexdigest();require(obj['Config']['Labels'].get('b70.prefix.plan')==digest,'Actual actor snapshot owner label differs')
 sample_started=time.time();pid=obj['State']['Pid'];before=pid_row(proc,pid);require(before['state'] not in ('Z','X'),'Actual actor root PID not live');out.mkdir(parents=True)
 rows=[]
 def preserve(path,label):
  start=time.time();
  with Path(path).open('rb') as stream:raw=stream.read(MAX_RAW+1)
  require(len(raw)<=MAX_RAW,'Bounded owned kernel metadata required');target=out/(label+'.raw');require(not target.exists(),'Exclusive kernel metadata receipt required');target.write_bytes(raw);row={'kernel_path':str(path),'path':str(target.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'started_epoch':start,'finished_epoch':time.time()};rows.append(row);return raw
 base=Path(proc)/str(pid);membership=preserve(base/'cgroup','root-cgroup');lines=membership.decode('ascii').splitlines();require(len(lines)==1 and lines[0].startswith('0::/'),'Exact unified owned cgroup membership required');relative=lines[0][3:];require('..'not in Path(relative).parts,'Cgroup traversal refused');cg=Path(cgroup)/relative.lstrip('/');require(cg.resolve().is_relative_to(Path(cgroup).resolve()),'Owned cgroup escaped');values={n:preserve(cg/n,'cgroup-'+n.replace('.','-')) for n in CGROUP_FILES}
 def scalar(name):
  raw=values[name].strip();require(raw.isdigit(),'Exact bounded cgroup scalar required');return int(raw)
 stat=numbers(values['memory.stat']);require({'anon','file','shmem'}<=set(stat),'Actual host backing categories missing');pids=[before];queue=[pid];seen={pid}
 while queue:
  parent=queue.pop(0);task=Path(proc)/str(parent)/'task';tids=sorted(int(p.name) for p in task.iterdir() if p.name.isdigit());require(parent in tids and len(tids)<=256,'Bounded actual task roster required');children=[]
  preserve_data=(' '.join(map(str,tids))+'\n').encode();target=out/('tasks-'+str(parent)+'.raw');target.write_bytes(preserve_data);rows.append({'kernel_path':str(task),'path':str(target.resolve()),'sha256':hashlib.sha256(preserve_data).hexdigest(),'bytes':len(preserve_data),'started_epoch':time.time(),'finished_epoch':time.time()})
  for tid in tids:children+=preserve(task/str(tid)/'children','children-'+str(parent)+'-'+str(tid)).decode('ascii').split()
  require(all(x.isdigit() for x in children) and len(set(children))==len(children),'Actual descendant PID list invalid')
  for word in children:
   child=int(word);require(child not in seen and len(seen)<64,'Bounded nonduplicated owned descendant tree required');identity=pid_row(proc,child);require(identity['ppid']==parent and identity['state']not in ('Z','X'),'Actual owned descendant ancestry changed');seen.add(child);pids.append(identity);queue.append(child)
 for identity in pids:
  p=Path(proc)/str(identity['pid']);observed=parse_stat(preserve(p/'stat','stat-'+str(identity['pid'])),identity['pid']);require(stable(identity,observed),'Original process identity changed');preserve(p/'status','status-'+str(identity['pid']));preserve(p/'smaps_rollup','smaps-'+str(identity['pid']));fds=p/'fdinfo';entries=sorted(fds.iterdir(),key=lambda path:int(path.name));require(len(entries)<=256 and all(e.name.isdigit() and not e.is_symlink() for e in entries),'Bounded regular owned fdinfo roster required')
  for entry in entries:preserve(entry,'fdinfo-'+str(identity['pid'])+'-'+entry.name)
  require(stable(identity,pid_row(proc,identity['pid'])),'Actual owned PID/start/ancestry changed during sampling')
 require(stable(before,pid_row(proc,pid)),'Actual actor PID changed during capture');sample={'kind':'owned_host_memory','started_epoch':sample_started,'finished_epoch':time.time(),'host_controller_pid':controller_pid,'container_id':obj['Id'],'container_root_host_pid':pid,'owned_process_identities':pids,'actor_plan_sha256':digest,'actor_command_sha256':hashlib.sha256(json.dumps(command,separators=(',',':'),ensure_ascii=True).encode()).hexdigest(),'cgroup_path':relative,'cgroup_memory_current_bytes':scalar('memory.current'),'cgroup_memory_peak_bytes':scalar('memory.peak'),'cgroup_anon_bytes':stat['anon'],'cgroup_file_bytes':stat['file'],'cgroup_shmem_bytes':stat['shmem'],'cgroup_swap_current_bytes':scalar('memory.swap.current'),'raw_receipts':rows,'whole_system_physical_peak_qualified':False,'per_expert_device_residency_qualified':False,'kernel_snapshot_atomicity_qualified':False,'actual_GPU_queries_executed':False,'drm_fdinfo_raw_observed':any(b'drm-' in (out/Path(r['path']).name).read_bytes() for r in rows if Path(r['path']).name.startswith('fdinfo-')),'drm_residency_semantics_or_per_allocation_attribution_qualified':False}
 require(values['memory.max'].strip()==str(105*1024**3).encode() and values['memory.swap.max'].strip()==b'0','Actual 105GiB/no-swap actor cgroup recipe required')
 sample['ownership_receipt']={'inspection':obj,'command':command,'actor_root':str(actor.resolve())}
 from full_cache_shared_memory_v2 import host_observation
 host_observation(sample);(out/'sample.json').write_text(json.dumps(sample,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii');return sample

def recollect(root,saved):
 root=Path(root).resolve();require(json.loads((root/'sample.json').read_bytes())==saved,'Original memory sample changed');receipts=saved['raw_receipts'];require(len({r['path']for r in receipts})==len(receipts),'Duplicate raw receipt path')
 for row in receipts:
  require(type(row['started_epoch']) in (int,float) and type(row['finished_epoch']) in (int,float) and saved['started_epoch']<=row['started_epoch']<=row['finished_epoch']<=saved['finished_epoch'],'Original kernel sample chronology differs')
  p=Path(row['path']);require(p.is_absolute() and p.parent==root and not p.is_symlink() and p.is_file(),'Confined regular original kernel receipt required');raw=p.read_bytes();require(len(raw)==row['bytes']<=MAX_RAW and hashlib.sha256(raw).hexdigest()==row['sha256'] and p.read_bytes()==raw,'Actual original kernel receipt changed')
 expected={Path(r['path']).name for r in receipts}|{'sample.json'};require({p.name for p in root.iterdir()}==expected,'Exact original memory sample raw roster required')
 raw={Path(r['path']).name:(Path(r['path']).read_bytes()) for r in receipts};stat=numbers(raw['cgroup-memory-stat.raw']);computed={'cgroup_memory_current_bytes':int(raw['cgroup-memory-current.raw'].strip()),'cgroup_memory_peak_bytes':int(raw['cgroup-memory-peak.raw'].strip()),'cgroup_swap_current_bytes':int(raw['cgroup-memory-swap-current.raw'].strip()),'cgroup_anon_bytes':stat['anon'],'cgroup_file_bytes':stat['file'],'cgroup_shmem_bytes':stat['shmem']};require(all(type(saved[k])is int and saved[k]==v for k,v in computed.items()),'Actual raw kernel byte counts differ')
 membership=raw['root-cgroup.raw'].decode('ascii').splitlines();require(membership==['0::'+saved['cgroup_path']],'Actual raw cgroup membership differs')
 require(raw['cgroup-memory-max.raw'].strip()==str(105*1024**3).encode() and raw['cgroup-memory-swap-max.raw'].strip()==b'0','Original actor cgroup limits changed')
 identities=saved['owned_process_identities'];require(type(identities)is list and 0<len(identities)<=64 and len({i['pid'] for i in identities})==len(identities),'Exact bounded original PID roster required')
 bypid={i['pid']:i for i in identities};require(identities[0]['pid']==saved['container_root_host_pid'],'Original root PID association differs');seen={identities[0]['pid']}
 for identity in identities:
  pid=identity['pid'];require(stable(identity,parse_stat(raw['stat-'+str(pid)+'.raw'],pid)),'Original raw PID/start identity differs')
  tids=raw['tasks-'+str(pid)+'.raw'].decode('ascii').split();require(all(w.isdigit() for w in tids) and str(pid) in tids and len(set(tids))==len(tids)<=256,'Original task roster malformed');children=[]
  for tid in tids:children+=raw['children-'+str(pid)+'-'+tid+'.raw'].decode('ascii').split()
  require(all(w.isdigit() for w in children) and len(set(children))==len(children),'Original child roster malformed')
  for w in children:
   child=int(w);require(child not in seen and child in bypid and bypid[child]['ppid']==pid,'Original descendant ancestry differs');seen.add(child)
 require(seen==set(bypid),'Original descendant roster incomplete')
 require(saved['drm_fdinfo_raw_observed'] is any(b'drm-' in v for k,v in raw.items() if k.startswith('fdinfo-')),'DRM raw observation scope differs')
 owner=saved['ownership_receipt'];obj=owner['inspection'];command=owner['command'];name=command[command.index('--name')+1];actor=Path(owner['actor_root']);require(obj['Id']==saved['container_id'] and obj['Name']=='/'+name and obj['State']['Pid']==saved['container_root_host_pid'] and obj['State']['Running'] is True,'Original owned inspection changed')
 digest=hashlib.sha256((actor/'plan.snapshot.json').read_bytes()).hexdigest();require(digest==saved['actor_plan_sha256']==obj['Config']['Labels']['b70.prefix.plan'],'Original actor snapshot association differs')
 from run_full_cache_shared_runtime_v3 import command_recipe
 plan=json.loads((actor/'plan.snapshot.json').read_bytes());recipe_binding(obj,command,plan['image']);require(command==command_recipe(plan,actor,saved['host_controller_pid']) and obj['Image']==plan['image']==obj['Config']['Image'] and saved['actor_command_sha256']==hashlib.sha256(json.dumps(command,separators=(',',':'),ensure_ascii=True).encode()).hexdigest(),'Original full actor recipe differs')
 from full_cache_shared_memory_v2 import host_observation
 result=host_observation(saved);result['original_raw_receipts_still_need_owned_recollection']=False;return result
