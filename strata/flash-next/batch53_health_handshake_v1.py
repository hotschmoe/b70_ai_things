"""Owned parent/child READY/ACK protocol; only root execution observes processes."""
import hashlib,json,math,os,stat,time
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
ROOT=Path(__file__).resolve().parents[2]
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pid_identity(pid):
 require(type(pid)is int and pid>0,'Exact positive process PID required');text=Path('/proc',str(pid),'stat').read_text();end=text.rfind(')');fields=text[end+2:].split();require(end>=0 and len(fields)>=20 and fields[0]not in ('Z','X'),'Live owned process identity required');return {'pid':pid,'start_ticks':int(fields[19])}
def leases():
 rows=[]
 for card in (0,1):
  path=Path('/mnt/vm_8tb/b70/gpu.lock.'+str(card));require(os.path.samefile('/proc/self/fd/'+str(8+card),path),'Inherited exact pair lease required');s=os.fstat(8+card);rows.append({'card':card,'device':s.st_dev,'inode':s.st_ino})
 return rows
def regular(path):
 path=Path(path).absolute()
 for ancestor in [path,*path.parents]:require(not ancestor.is_symlink(),'Handshake symlink refused')
 require(stat.S_ISREG(path.lstat().st_mode) and path.stat().st_size<=1<<20,'Bounded regular handshake record required');return path
def read(path):return read_unique(regular(path))
def write_new(path,value):
 path=Path(path);raw=(canonical(value)+'\n').encode('ascii');require(len(raw)<=1<<20,'Handshake record bound');temp=path.with_name(path.name+'.writing-'+str(os.getpid()))
 with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 try:os.link(temp,path)
 finally:temp.unlink()
 return sha(path)
def finite(value):return type(value)in (int,float) and math.isfinite(value) and value>0

def record_types(row):
 require(type(row['schema'])is int and row['schema']==1 and type(row['generation'])is int and row['generation']==53,'Exact typed handshake schema/generation required')
 for key in ('parent','child'):
  if key in row:
   identity=row[key];require(set(identity)=={'pid','start_ticks'} and type(identity['pid'])is int and identity['pid']>0 and type(identity['start_ticks'])is int and identity['start_ticks']>=0,'Exact typed process identity required')
 require(type(row['lease_identity'])is list and len(row['lease_identity'])==2,'Exact two lease identities required')
 for card,value in enumerate(row['lease_identity']):require(set(value)=={'card','device','inode'} and all(type(v)is int for v in value.values()) and value['card']==card,'Exact typed lease identity required')
 if 'max_wait_seconds'in row:require(type(row['max_wait_seconds'])is int and 1<=row['max_wait_seconds']<=10800,'Exact typed bounded handshake deadline required')

def offer(root,plan_sha,parent_source,controller_source,max_seconds):
 require(type(max_seconds)is int and 1<=max_seconds<=10800,'Bounded handshake deadline required');root=Path(root);root.mkdir(exist_ok=False)
 row={'schema':1,'generation':53,'parent':pid_identity(os.getpid()),'plan_sha256':plan_sha,'parent_source_sha256':sha(parent_source),'controller_source_sha256':sha(controller_source),'lease_identity':leases(),'created_epoch':time.time(),'max_wait_seconds':max_seconds}
 write_new(root/'offer.json',row);return row

def ready(root,admitted,pack,sdk,parent_source,controller_source,*,logical_ready_boundary):
 root=Path(root);o=read(root/'offer.json');record_types(o);require(o['schema']==1 and o['generation']==53 and o['plan_sha256']==admitted.verify(),'Exact offered admitted plan required');require(o['parent']==pid_identity(os.getppid()) and o['parent_source_sha256']==sha(parent_source) and o['controller_source_sha256']==sha(controller_source) and o['lease_identity']==leases(),'Actual parent/source/lease differs')
 require(pack.phase==sdk.phase=='admission','Independent child admission before READY required');pb=pack.ready_seal();sb=sdk.ready_seal()
 row={'schema':1,'generation':53,'offer_sha256':sha(root/'offer.json'),'parent':o['parent'],'child':pid_identity(os.getpid()),'plan_sha256':admitted.verify(),'parent_source_sha256':o['parent_source_sha256'],'controller_source_sha256':o['controller_source_sha256'],'lease_identity':leases(),'logical_ready_boundary':logical_ready_boundary,'pack_ready_boundary':pb,'sdk_ready_boundary':sb,'ready_epoch':time.time()};write_new(root/'ready.json',row);return row

def validate_ready(root,offer_row,child_pid,plan_sha):
 root=Path(root);r=read(root/'ready.json');record_types(r);record_types(offer_row);require(read(root/'offer.json')==offer_row and r['schema']==1 and r['generation']==53 and r['offer_sha256']==sha(root/'offer.json'),'Original offered READY binding differs');require(r['parent']==offer_row['parent']==pid_identity(os.getpid()) and r['child']==pid_identity(child_pid),'Exact live parent/child READY identity differs');require(r['plan_sha256']==offer_row['plan_sha256']==plan_sha and r['lease_identity']==offer_row['lease_identity']==leases(),'READY plan/pair lease differs')
 require(r['parent_source_sha256']==offer_row['parent_source_sha256'] and r['controller_source_sha256']==offer_row['controller_source_sha256'],'READY source differs');require(finite(r['ready_epoch']) and offer_row['created_epoch']<=r['ready_epoch']<=time.time(),'READY epoch differs')
 for row,label in ((r['pack_ready_boundary'],'ready_complete_byte_recheck'),(r['sdk_ready_boundary'],'ready')):require(row['label']==label and finite(row['started_epoch']) and finite(row['finished_epoch']) and offer_row['created_epoch']<=row['started_epoch']<=row['finished_epoch']<=r['ready_epoch'],'READY complete-byte chronology invalid')
 row=r['logical_ready_boundary'];require(row['label']=='ready' and finite(row['started_epoch']) and finite(row['finished_epoch']) and offer_row['created_epoch']<=row['started_epoch']<=row['finished_epoch']<=r['ready_epoch'],'Actual complete logical READY edge invalid')
 return r

def health_binding(path,ready_epoch):
 path=regular(path);h=read(path);require(type(h['schema'])is int and h['schema']==1 and type(h['cards'])is list and all(type(c)is int for c in h['cards']),'Exact typed health schema/cards required');require(h['passed']is True and h['cards']==[0,1] and h['health_image']==HEALTH and finite(h['started_epoch']) and finite(h['finished_epoch']) and ready_epoch<=h['started_epoch']<=h['finished_epoch'],'Actual health must follow child READY')
 expected=[[str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],[str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180']]
 require(len(h['files'])==len(h['checks'])==2 and h['files']==h['checks'],'Exact two original health rows required')
 for row,argv,label in zip(h['files'],expected,('strict','compiled-pair')):
  cmd=path.parent/('pre-'+label+'.command.json');log=path.parent/('pre-'+label+'.log');require(row['return_code']==0 and type(row['return_code'])is int and row['error']is None and row['command']==read(cmd)==argv and row['command_file_sha256']==sha(cmd) and row['path']==str(log) and row['sha256']==row['stdout_sha256']==sha(regular(log)),'Actual health rc/error/command/log differs');require(finite(row['started_epoch']) and finite(row['finished_epoch']) and h['started_epoch']<=row['started_epoch']<=row['finished_epoch']<=h['finished_epoch'],'Actual health row chronology differs')
 require(h['source_sha256']=={str(p):sha(p) for p in (ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh',ROOT/'bin/xpu-collective-health',ROOT/'bin/xpu-collective-health.py')},'Actual current health source differs');return h

def acknowledge(root,ready_row,health_path,journal_path):
 root=Path(root);require(read(root/'ready.json')==ready_row,'READY changed before ACK');health_binding(health_path,ready_row['ready_epoch']);j=read(journal_path);require(j['return_code']==0 and type(j['return_code'])is int and j['error']is None and j['sha256']==j['stdout_sha256']==sha(regular(j['path'])) and j['command']==read(journal_path.with_name('pre-kernel-journal.command.json')) and j['command_file_sha256']==sha(journal_path.with_name('pre-kernel-journal.command.json')),'Prejournal failed before ACK')
 row={'schema':1,'generation':53,'ready_sha256':sha(root/'ready.json'),'offer_sha256':sha(root/'offer.json'),'parent':pid_identity(os.getpid()),'child':ready_row['child'],'plan_sha256':ready_row['plan_sha256'],'lease_identity':leases(),'health_path':str(Path(health_path).resolve()),'health_sha256':sha(health_path),'journal_path':str(Path(journal_path).resolve()),'journal_sha256':sha(journal_path),'ack_epoch':time.time()};write_new(root/'ack.json',row);return row

def wait_ack(root,ready_row,admitted,health_path,parent_source,controller_source):
 root=Path(root);o=read(root/'offer.json');record_types(o);deadline=time.monotonic()+o['max_wait_seconds']
 while not (root/'ack.json').exists():
  require(time.monotonic()<deadline,'Owned READY/ACK timeout');require(pid_identity(os.getppid())==o['parent'],'Parent exited/reused while waiting health');admitted.verify();require(leases()==o['lease_identity'],'Inherited lease changed while waiting health');time.sleep(.1)
 ack=read(root/'ack.json');record_types(ack);require(ack['schema']==1 and ack['generation']==53 and ack['ready_sha256']==sha(root/'ready.json') and ack['offer_sha256']==sha(root/'offer.json') and read(root/'ready.json')==ready_row,'ACK actual READY/offer changed');require(ack['parent']==o['parent']==pid_identity(os.getppid()) and ack['child']==ready_row['child']==pid_identity(os.getpid()) and ack['plan_sha256']==admitted.verify() and ack['lease_identity']==leases(),'ACK process/plan/lease differs');require(sha(parent_source)==o['parent_source_sha256'] and sha(controller_source)==o['controller_source_sha256'],'ACK source changed');require(Path(ack['health_path'])==Path(health_path).resolve() and ack['health_sha256']==sha(health_path) and Path(ack['journal_path'])==root.parent/'pre-kernel-receipt.json' and ack['journal_sha256']==sha(ack['journal_path']),'ACK actual health/journal paths differ');health=health_binding(health_path,ready_row['ready_epoch']);require(finite(ack['ack_epoch']) and health['finished_epoch']<=ack['ack_epoch']<=time.time(),'ACK health chronology differs');return ack

def freshness(health,plan,now):
 require(type(health['cards'])is list and all(type(c)is int for c in health['cards']) and finite(health['finished_epoch']) and finite(now),'Exact typed health freshness inputs required')
 require(health['passed']is True and set(plan['cards'])<=set(health['cards']) and 0<=now-health['finished_epoch']<=300,'Fresh parent actual prehealth required')

def saved_binding(root,parent,plan,child):
 """Read-only original records; no live process, lease, Docker or device query."""
 root=Path(root).resolve();directory=root/'handshake';require(directory.is_dir() and not directory.is_symlink() and {p.name for p in directory.iterdir()}=={'offer.json','ready.json','ack.json'},'Exact immutable handshake artifact roster required');o=read(directory/'offer.json');r=read(directory/'ready.json');a=read(directory/'ack.json')
 for row in (o,r,a):record_types(row)
 require(o['schema']==r['schema']==a['schema']==1 and o['generation']==r['generation']==a['generation']==53,'Original handshake generation differs')
 require(parent['handshake_offer_sha256']==r['offer_sha256']==a['offer_sha256']==sha(directory/'offer.json') and parent['handshake_ready_sha256']==child['health_handshake']['ready_sha256']==a['ready_sha256']==sha(directory/'ready.json') and parent['handshake_ack_sha256']==child['health_handshake']['ack_sha256']==sha(directory/'ack.json'),'Original READY/ACK byte roster differs')
 require(o['parent']==r['parent']==a['parent'] and o['parent']['pid']==parent['parent_pid'] and r['child']==a['child']==parent['child_process_identity'] and r['child']['pid']==parent['child_pid'],'Original PID/start identity joins differ')
 for identity in (o['parent'],r['child']):require(set(identity)=={'pid','start_ticks'} and type(identity['pid'])is int and identity['pid']>0 and type(identity['start_ticks'])is int and identity['start_ticks']>=0,'Original typed PID/start identity required')
 require(o['plan_sha256']==r['plan_sha256']==a['plan_sha256']==parent['plan_sha256']==sha(root/'admitted-plan.snapshot.json'),'Exact offered plan differs')
 require(o['lease_identity']==r['lease_identity']==a['lease_identity'] and len(o['lease_identity'])==2,'Original pair lease identity differs')
 for card,row in enumerate(o['lease_identity']):
  st=Path('/mnt/vm_8tb/b70/gpu.lock.'+str(card)).stat();require(row=={'card':card,'device':st.st_dev,'inode':st.st_ino} and all(type(v)is int for v in row.values()),'Original lease inode/card identity differs')
 require(o['parent_source_sha256']==r['parent_source_sha256']==parent['wrapper_sha256']==sha(ROOT/'strata/flash-next/qualify_batch_numerical_v53.py') and o['controller_source_sha256']==r['controller_source_sha256']==parent['controller_sha256']==sha(ROOT/'strata/flash-next/batch_numerical_execution_v53.py'),'Actual handshake source joins differ')
 require(type(o['max_wait_seconds'])is int and 1<=o['max_wait_seconds']<=10800 and parent['handshake_ready_observed']is True,'Bounded actual READY observation required')
 values=[parent['started_epoch'],o['created_epoch'],parent['child_started_epoch'],r['ready_epoch'],parent['handshake_ready_observed_epoch']]
 require(all(finite(v) for v in values) and values==sorted(values),'Original child admission/READY chronology differs')
 for row,label in ((r['pack_ready_boundary'],'ready_complete_byte_recheck'),(r['sdk_ready_boundary'],'ready')):require(row['label']==label and finite(row['started_epoch']) and finite(row['finished_epoch']) and parent['child_started_epoch']<=row['started_epoch']<=row['finished_epoch']<=r['ready_epoch'],'Original READY byte seals before health required')
 h=health_binding(root/'pre-health.json',r['ready_epoch']);j=read(root/'pre-kernel-receipt.json');require(a['health_path']==str(root/'pre-health.json') and a['health_sha256']==sha(root/'pre-health.json') and a['journal_path']==str(root/'pre-kernel-receipt.json') and a['journal_sha256']==sha(root/'pre-kernel-receipt.json'),'Original ACK health/journal exact association differs')
 require(parent['pre_health_finished_epoch']==h['finished_epoch'] and parent['kernel_journal_receipts']['pre']==j and j['return_code']==0 and type(j['return_code'])is int and j['error']is None,'Original ACK completed health/journal differs')
 marks=child['health_handshake'];chain=[parent['handshake_ready_observed_epoch'],h['started_epoch'],h['finished_epoch'],j['started_epoch'],j['finished_epoch'],a['ack_epoch'],marks['before_seal_checked_epoch'],marks['after_seal_checked_epoch'],child['leaf_launch_started_epoch'],child['owned_leaf_terminal_epoch'],child['finished_epoch'],parent['child_terminal_epoch']]
 require(all(finite(v) for v in chain) and chain==sorted(chain) and a['ack_epoch']==parent['handshake_ack_epoch']==marks['ack_epoch'] and marks['ready_epoch']==r['ready_epoch'],'Original READY-health-ACK-reseal-leaf-terminal chronology differs')
 for epoch in (marks['before_seal_checked_epoch'],marks['after_seal_checked_epoch'],child['leaf_launch_started_epoch']):freshness(h,plan,epoch)
 for name in ('pack','sdk'):
  proof=read(root/'child'/(name+'-operation-witness.json'));require(len(proof['boundaries'])==4 and proof['boundaries'][1]==r[name+'_ready_boundary'] and a['ack_epoch']<=marks['before_seal_checked_epoch']<=proof['boundaries'][2]['started_epoch']<=proof['boundaries'][2]['finished_epoch']<=marks['after_seal_checked_epoch'],'Original ACK actual full-byte reseals differ')
 return {'ready_sha256':sha(directory/'ready.json'),'ack_sha256':sha(directory/'ack.json'),'actual_health_after_READY':True,'actual_full_byte_reseal_after_ACK':True,'health_freshness_limit_seconds':300,'health_epoch_rewritten':False,'runtime_proof_transferred':False}
