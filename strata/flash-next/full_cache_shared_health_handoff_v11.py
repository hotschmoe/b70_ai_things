"""NEW source40 child semantic READY, actual parent health ACK, no clock rewrite."""
import hashlib,json,math,os,time
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical

def require(ok,message):
 if not ok:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def start_ticks(pid,proc='/proc'):
 text=(Path(proc)/str(pid)/'stat').read_text();at=text.rfind(')');parts=text[at+2:].split();require(type(pid)is int and pid>0 and at>=0 and len(parts)>=20 and parts[0] not in ('Z','X') and int(text.split(' ',1)[0])==pid,'Actual live owned child PID required');return int(parts[19])
def atomic(path,value):
 path=Path(path);temporary=path.with_suffix('.tmp');require(not temporary.exists(),'Fresh exclusive handoff temporary required');temporary.write_text(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False)+'\n');temporary.replace(path)
def leases():
 rows=[]
 for card in (0,1):
  path=Path('/mnt/vm_8tb/b70/gpu.lock.'+str(card));require(os.path.samefile('/proc/self/fd/'+str(8+card),path),'Actual inherited pair lease required');st=os.fstat(8+card);rows.append({'card':card,'device':st.st_dev,'inode':st.st_ino})
 return rows

def process(pid):return {'pid':pid,'start_ticks':start_ticks(pid)}
def typed_identity(row):require(type(row)is dict and set(row)=={'pid','start_ticks'} and all(type(v)is int for v in row.values()) and row['pid']>0 and row['start_ticks']>=0,'Exact typed original PID/start identity required')
def typed_leases(rows):
 require(type(rows)is list and len(rows)==2,'Actual two pair lease records required')
 for card,row in enumerate(rows):require(type(row)is dict and set(row)=={'card','device','inode'} and all(type(v)is int for v in row.values()) and row['card']==card,'Exact typed pair lease inode/card identity required')

def offer(root,plan_sha,wrapper,controller,max_wait,parent_started):
 require(type(max_wait)is int and 1<=max_wait<=10800,'Exact bounded original parent offer deadline required');row={'schema':3,'kind':'source40_actual_parent_offer','parent':process(os.getpid()),'lease_identity':leases(),'plan_sha256':plan_sha,'parent_source_sha256':sha(wrapper),'controller_source_sha256':sha(controller),'max_wait_seconds':max_wait,'created_epoch':time.time(),'parent_started_epoch':parent_started};atomic(Path(root)/'leaf-offer.json',row);return row

def parent_continuity(offered):
 typed_identity(offered['parent']);typed_leases(offered['lease_identity']);require(process(os.getppid())==offered['parent'] and leases()==offered['lease_identity'],'Actual offered parent disappeared/reused or inherited lease changed')

def ready_packet(plan,manifest,pid,ticks,epoch,offered):
 return {'schema':3,'kind':'source40_postsemantic_READY','producer_pid':pid,'producer_start_ticks':ticks,'driver_sha256':plan['driver_sha256'],'plan_semantic_sha256':digest(plan),'manifest_sha256':digest(manifest),'engine_receipt_sha256':plan['engine_receipt_sha256'],'prepared_sha256':plan['prepared_sha256'],'observed_epoch':epoch,'GPU_leaf_launched':False,'parent':offered['parent'],'lease_identity':offered['lease_identity'],'offer_semantic_sha256':digest(offered)}
def admit_ready(packet,plan,manifest,child_pid,ticks,launch_epoch,offered):
 typed_identity(packet['parent']);typed_leases(packet['lease_identity']);expected=ready_packet(plan,manifest,child_pid,ticks,packet['observed_epoch'],offered);require(packet==expected and type(packet['schema'])is int and type(packet['producer_pid'])is int and type(packet['producer_start_ticks'])is int and type(packet['observed_epoch']) in (int,float) and math.isfinite(packet['observed_epoch']) and launch_epoch<=packet['observed_epoch'],'Exact actual owned postsemantic READY/source/epoch required');return packet

def child_wait(plan,manifest,out,deadline=1200):
 """ROOT execution only, before every independently owned GPU leaf."""
 out=Path(out);root=out.parent;offered=read_unique(root/'leaf-offer.json');require(offered['schema']==3 and offered['kind']=='source40_actual_parent_offer' and offered['plan_sha256']==sha(root/'input-plan.snapshot.json') and offered['controller_source_sha256']==plan['driver_sha256'] and offered['parent_source_sha256']==sha(root/'wrapper.py'),'Exact actual offered parent source/plan required');parent_continuity(offered);packet=ready_packet(plan,manifest,os.getpid(),start_ticks(os.getpid()),time.time(),offered);require(not (root/'leaf-ready.json').exists() and not (root/'leaf-ack.json').exists(),'Fresh parent/child handoff paths required');atomic(root/'leaf-ready.json',packet);limit=time.monotonic()+deadline
 while True:
  parent_continuity(offered)
  if (root/'leaf-ack.json').is_file():
   ack=read_unique(root/'leaf-ack.json');health=read_unique(root/'leaf-health.json');health_raw_admit(root,health);admit_ack(packet,ack,health);require(ack['parent']==offered['parent']==process(os.getppid()) and ack['lease_identity']==offered['lease_identity']==leases() and ack['offer_semantic_sha256']==digest(offered),'Actual live parent/lease ACK differs');require(ack['health_sha256']==sha(root/'leaf-health.json') and ack['kernel_receipt_sha256']==sha(root/'leaf-kernel-receipt.json'),'Actual ACK-bound raw health/journal changed');kernel=read_unique(root/'leaf-kernel-receipt.json');kernel_admit(root,kernel,offered,health,ack);parent_continuity(offered);return {'ready':packet,'ack':ack,'actual_health':health,'offered':offered}
  require(time.monotonic()<limit,'Actual parent health ACK deadline; no GPU launch authorized');time.sleep(.05)
def admit_ack(ready,ack,health):
 require(set(ack)=={'schema','kind','ready_sha256','producer_pid','producer_start_ticks','health_sha256','kernel_receipt_sha256','observed_epoch','parent','lease_identity','offer_semantic_sha256'} and type(ack['schema'])is int and ack['schema']==3 and ack['kind']=='source40_actual_parent_health_ACK' and ack['ready_sha256']==digest(ready) and ack['producer_pid']==ready['producer_pid'] and ack['producer_start_ticks']==ready['producer_start_ticks'],'Exact owned source40 ACK request association required')
 typed_identity(ack['parent']);typed_leases(ack['lease_identity']);require(ack['parent']==ready['parent'] and ack['lease_identity']==ready['lease_identity'] and ack['offer_semantic_sha256']==ready['offer_semantic_sha256'],'Original offered parent/lease copies differ');require(health['passed'] is True and type(health['cards'])is list and all(type(c)is int for c in health['cards']) and health['cards']==[0,1] and all(type(health[k]) in (int,float) and math.isfinite(health[k]) for k in ('started_epoch','finished_epoch')) and ready['observed_epoch']<=health['started_epoch']<=health['finished_epoch']<=ack['observed_epoch'],'Actual fresh strict+compiled health must follow complete child semantics');require(type(ack['observed_epoch']) in (int,float) and math.isfinite(ack['observed_epoch']) and 0<=time.time()-health['finished_epoch']<=300,'Unchanged 300s actual health gate required')
 return health

def health_raw_admit(root,health):
 import qualify_batch_numerical_v7 as original
 root=Path(root);repo=Path(__file__).resolve().parents[2];require(type(health['schema'])is int and health['schema']==1 and type(health['cards'])is list and all(type(c)is int for c in health['cards']) and health['cards']==[0,1] and health['health_image']==original.HEALTH and health['passed'] is True and health['files']==health['checks'] and len(health['files'])==2,'Exact actual typed pair-health receipt/roster required')
 expected=[[str(repo/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',original.HEALTH],[str(repo/'bin/xpu-collective-health'),'--img',original.HEALTH,'--p2p','0','--timeout','180']]
 for row,argv,label in zip(health['files'],expected,('strict','compiled-pair')):
  path=root/('leaf-'+label+'.log');cmd=root/('leaf-'+label+'.command.json');require(type(row['return_code'])is int and row['return_code']==0 and row['error'] is None and row['path']==str(path) and row['sha256']==sha(path) and row['command']==argv==read_unique(cmd) and row['command_file_sha256']==sha(cmd),'Exact ACK health argv/log/hash/error required before leaf');require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('started_epoch','finished_epoch')) and health['started_epoch']<=row['started_epoch']<=row['finished_epoch']<=health['finished_epoch'],'Actual original health command chronology differs')
 require(health['source_sha256']=={str(p):sha(p) for p in (repo/'vllm/int4/diagnostics/xpu_health_strict.sh',repo/'bin/xpu-collective-health',repo/'bin/xpu-collective-health.py')},'Actual ACK health-source bytes changed');return health

def kernel_admit(root,row,offered,health,ack):
 import qualify_batch_numerical_v7 as original
 root=Path(root);command=['journalctl','-k','--since','@'+str(int(offered['parent_started_epoch'])),'--no-pager'];path=root/'leaf-kernel-journal.log';cmd=root/'leaf-kernel-journal.command.json'
 require(type(row['return_code'])is int and row['return_code']==0 and row['error'] is None and row['path']==str(path) and row['command']==command==read_unique(cmd) and row['command_file_sha256']==sha(cmd) and row['sha256']==sha(path) and not original.FAULT.search(path.read_text()),'Exact actual ACK-bound kernel argv/hash/error/fault required before leaf')
 require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('started_epoch','finished_epoch')) and health['finished_epoch']<=row['started_epoch']<=row['finished_epoch']<=ack['observed_epoch'],'Actual post-health kernel chronology differs');return row

def final_binding(root,parent,plan,child,controller=None):
 """Original parent health/journal records reexecuted alongside existing gates."""
 import qualify_batch_numerical_v7 as original
 from validate_batch_api_cache_positive_buffered_v5 import exact_health_gate
 root=Path(root).resolve();offered=read_unique(root/'leaf-offer.json');ready=read_unique(root/'leaf-ready.json');ack=read_unique(root/'leaf-ack.json');manifest=parent['prepared_chain'];admit_ready(ready,plan,manifest,parent['child_pid'],parent['leaf_child_start_ticks'],parent['child_launch_started_epoch'],offered);require(offered==parent['leaf_offer'] and offered['parent']==ready['parent']==ack['parent'] and offered['lease_identity']==ready['lease_identity']==ack['lease_identity'] and ack['offer_semantic_sha256']==digest(offered),'Original offer/READY/ACK parent/lease differed');health=exact_health_gate(root,'leaf');health_raw_admit(root,health);require(health==child['health_handoff']['actual_health'] and ready==child['health_handoff']['ready'] and ack==child['health_handoff']['ack'] and parent['leaf_ready']==ready and parent['leaf_ack']==ack,'Actual original parent/child handoff copies differ')
 row=parent['leaf_kernel_journal'];command=['journalctl','-k','--since','@'+str(int(parent['started_epoch'])),'--no-pager'];path=root/'leaf-kernel-journal.log';cmd=root/'leaf-kernel-journal.command.json';require(row==read_unique(root/'leaf-kernel-receipt.json') and row['command']==command==read_unique(cmd) and row['command_file_sha256']==sha(cmd) and row['sha256']==sha(path) and row['return_code']==0 and row['error'] is None and not original.FAULT.search(path.read_text()),'Actual postsemantic raw journal command/error/fault differs')
 require(ack['health_sha256']==sha(root/'leaf-health.json') and ack['kernel_receipt_sha256']==sha(root/'leaf-kernel-receipt.json') and ack['ready_sha256']==digest(ready),'Original ACK health/journal/raw request hashes differ')
 epochs=[ready['observed_epoch'],health['started_epoch'],health['finished_epoch'],row['started_epoch'],row['finished_epoch'],ack['observed_epoch'],child['started_epoch'],child['finished_epoch']];require(all(type(v) in (int,float) and math.isfinite(v) for v in epochs) and epochs==sorted(epochs),'Actual semantics/health/journal/ACK/leaf/terminal chronology differs')
 seal=child['post_ACK_byte_seal'];require(seal['between_boundaries_byte_identity_observed'] is False and ack['observed_epoch']<=seal['started_epoch']<=seal['finished_epoch']<=child['started_epoch'] and seal['finished_epoch']-health['finished_epoch']<=300 and len(seal['SDK8_actual_byte_rows'])==8,'Original complete postACK seal/leaf chronology differs')
 prepared=read_unique(Path(plan['prepared'])/'prepared.json');receipt_path=Path(prepared['engine_receipt']);receipt=read_unique(receipt_path);require(sha(receipt_path)==plan['engine_receipt_sha256'] and {r['path'] for r in seal['SDK8_actual_byte_rows']}==set(receipt['binary_sha256']),'Original exact all8 SDK byte seal roster differs')
 for row in seal['SDK8_actual_byte_rows']:
  path=Path(row['path']);before=path.stat();current=sha(path);after=path.stat();require(before==after and row['sha256']==current==receipt['binary_sha256'][str(path)] and row['bytes']==after.st_size and row['stat5']==[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns],'Current saved original ABI byte/stat association differs')
 require(ack['producer_pid']==parent['child_pid']==child['producer_pid'] and ack['producer_start_ticks']==ready['producer_start_ticks'] and child['started_epoch']-health['finished_epoch']<=300,'Original actual owned leaf must start within unchanged health window')
 require(type(parent['parent_pid'])is int and type(parent['parent_start_ticks'])is int and offered['parent']=={'pid':parent['parent_pid'],'start_ticks':parent['parent_start_ticks']},'Original parent process record and actual offered identity differ')
 if controller is not None:seal_context_binding(seal,plan,controller)
 return {'actual_source40_semantics_then_health_then_leaf_joined':True,'original_epochs_or_receipts_rewritten':False}

def seal_context_binding(seal,plan,ctrl):
 """Rerun original source/input-identity predicates; page epochs remain original."""
 require(digest(seal['current_source_binding'])==digest(ctrl.source_binding()),'Actual saved postACK source proof changed')
 lock=ctrl.read(ctrl.HERE/'model-lock.json');shards=[ctrl.ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')]
 require(sha(Path(plan['model_identity']['path']))==plan['model_identity']['sha256'],'Original input full-four identity receipt changed')
 current=ctrl.verify_model_identity(Path(plan['model_identity']['path']),lock,shards);saved=seal['current_model_identity_and_pages'];original_page=saved['current_known_pages'];current_page=current['current_known_pages']
 require(type(original_page['epoch'])in(int,float) and math.isfinite(original_page['epoch']) and seal['started_epoch']<=original_page['epoch']<=seal['finished_epoch'],'Actual saved page read must occur inside original postACK seal')
 require(digest({k:v for k,v in original_page.items()if k!='epoch'})==digest({k:v for k,v in current_page.items()if k!='epoch'}) and digest({k:v for k,v in saved.items()if k!='current_known_pages'})==digest({k:v for k,v in current.items()if k!='current_known_pages'}),'Original postACK input identity/page extent/hash/stat predicate changed')
 return True

def post_ack_seal(plan,ctrl,handoff):
 """ROOT ONLY: independently reread actual8 SDK bytes, metadata and source pages."""
 parent_continuity(handoff['offered']);started=time.time();prepared=ctrl.read(Path(plan['prepared'])/'prepared.json');receipt_path=Path(prepared['engine_receipt']);require(sha(receipt_path)==plan['engine_receipt_sha256'] and sha(Path(plan['prepared'])/'prepared.json')==plan['prepared_sha256'],'Actual post-ACK prepared/SDK receipt changed');receipt=ctrl.read(receipt_path);engine=receipt_path.parent;build=ctrl.read(ctrl.ENGINE_PLAN);targets=build['build_targets'];require(len(targets)==8 and set(receipt['binary_sha256'])=={str(engine/'build'/n) for n in targets},'Actual complete8 ABI seal required');rows=[]
 for name in targets:
  path=engine/'build'/name;before=path.stat();actual=sha(path);after=path.stat();require(before==after and actual==receipt['binary_sha256'][str(path)],'Actual post-ACK full ABI bytes/stat changed');rows.append({'path':str(path),'sha256':actual,'bytes':after.st_size,'stat5':[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns]})
 source=ctrl.source_binding();lock=ctrl.read(ctrl.HERE/'model-lock.json');shards=[ctrl.ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];identity=ctrl.verify_model_identity(Path(plan['model_identity']['path']),lock,shards)
 parent_continuity(handoff['offered']);finished=time.time();require(0<=finished-handoff['actual_health']['finished_epoch']<=300,'Actual health expired during complete post-ACK seal; no timestamp refresh')
 return {'started_epoch':started,'finished_epoch':finished,'SDK8_actual_byte_rows':rows,'current_source_binding':source,'current_model_identity_and_pages':identity,'between_boundaries_byte_identity_observed':False}
