"""New source40 native observer producer. Root-only leased serving."""
import argparse,hashlib,json,os,re,shlex,subprocess,sys,time,signal
from pathlib import Path
import native_qsa40_matched_experiment_v1 as spec
import c1_serve_controller_combined_v140_v3 as c
import serial_prefix_qualification_v6 as numeric
from serial_prefix_qualification_v8 import request_command
from serial37_canonical_json_v3 import read_unique,canonical
from native_qsa40_protocol_v6 import Protocol
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'native-qsa40-owned-runtime-source-plan-v6.json';require=spec.require
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return read_unique(p)
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='ascii')
def closure():
 source=read(PLAN)
 for path,digest in source['files'].items():require(sha(ROOT/path if not Path(path).is_absolute() else path)==digest,'Current native QSA40 source closure changed '+path)
 return sha(PLAN)
def own_binding(root):
 from owned_layer3_qsa_control_v4 import producer_binding,arrays
 b=producer_binding(root);arrays(root);return b

def config(prepared):
 cfg=read(Path(prepared)/'server-config.json');args=list(cfg['args']);env=dict(cfg['env'])
 for key in ('STRATA_BATCH_FULL_STATE_CHAIN','STRATA_BATCH_PUBLIC_PREFIX','STRATA_BATCH_FIDELITY_DIAG','STRATA_SLOT_OWNER_TRACE','STRATA_MIRROR_OWNER_TRACE','STRATA_LAYER0_Q8_DIAG','STRATA_PREFIX30','STRATA_PLE_INPUT33','STRATA_QSA3_TARGET'):require(env.get(key,'0')=='0','Required source40 baseline feature OFF '+key)
 for flag,value in {'--max-context':'2048','--prefill':'64','--ple-row-cache':'65536','--kv':'fp16','--prompt-cache':'0','--conversation-cache-mib':'0','--batch':'0','--suffix-draft':'0','--lookup-chain':'0','--adapt-swaps':'0','--adapt-every':'0'}.items():
  require(args.count(flag)<=1,'Duplicate actual config flag '+flag);numeric.option(args,flag,value)
 require('--no-prefill-borrow'in args and '--mtp'not in args and '--pipeline-windows'not in args,'Fixed native source40 profile required')
 env.update({'STRATA_LAYER0_Q8_DIAG':'0','STRATA_BATCH_FIDELITY_DIAG':'0','STRATA_SLOT_OWNER_TRACE':'0','STRATA_MIRROR_OWNER_TRACE':'0','SYCL_UR_TRACE':'2','SYCL_CACHE_PERSISTENT':'0'})
 return args,env,cfg

def kernel_binding(engine):
 import native_rms_rsqrt37_proposal_v1 as prior
 from owned_layer3_qsa_runtime_contract_v4 import qsa_builder_flags
 root=Path(engine);plan=read(spec.ENGINE_PLAN);facts={}
 for name in ('native_qsa.dp.cpp','native_rope.dp.cpp','native_qsa_indexer.dp.cpp','qsa_decode_attn.dp.cpp','qsa.dp.cpp'):
  rel='sycl/src/kernels/cuda/'+name;current=root/'source'/rel;old=prior.SDK/'source'/rel;expected=read(HERE/'owned-layer3-qsa-control-source-plan-v4.json')['actual_consumed_source_bindings'].get(str(old));require(expected is not None and sha(current)==expected==sha(old),'Declared source40 vs own-control unchanged QSA kernel bytes differ '+name);facts[rel]=expected
 flags=qsa_builder_flags(root,prior.builder_flags())
 return {'source40_actual_kernel_sha256':facts,'actual_object_and_link_flag_binding':flags,'source37_runtime_proof_transferred':False,'internal_arithmetic_qualified':False}

def registry_binding():
 from registry_c140_shared_association_v3 import association
 return association(ROOT/'evals/configs/models.yaml',include_shared=True)

def prepare(a):
 from native_qsa40_byte_epoch_v6 import metadata_preflight
 metadata_preflight(a.prepared,a.own_producer)
 closed=spec.candidate_binding(a.prepared);m=closed['prepared'];args,env,cfg=config(a.prepared);spec.recipe(m['cards']);require(numeric.expected_stage_ranges(args)==[tuple(v)for v in spec.recipe(m['cards'])['stages'].values()],'Exact source40 layer split/topology required');spec.arm_environment(env,False,'0'*64,'/results/qsa');own=own_binding(a.own_producer)
 value={'schema':1,'prepared':str(a.prepared.resolve()),'candidate_binding':closed,'prepared_sha256':sha(a.prepared/'prepared.json'),'server_config_sha256':sha(a.prepared/'server-config.json'),'args':args,'base_env':env,'image':m['runtime']['image'],'pack':m['pack'],'engine_root':str(Path(m['engine_receipt']).resolve().parent),'cards':m['cards'],'recipe':spec.recipe(m['cards']),'own_producer_root':str(a.own_producer.resolve()),'own_producer_binding':own,'registry_sha256':sha(ROOT/'evals/configs/models.yaml'),'registry_entry_preservation':registry_binding(),'source_plan_sha256':closure(),'driver_sha256':sha(__file__),'kernel_source_flags':kernel_binding(Path(m['engine_receipt']).resolve().parent),'captured_operands_used':False,'full_model_math_qualified':False}
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',value);manifest(value);return value

def manifest(plan):
 require(type(plan['schema'])is int and plan['schema']==1 and plan['driver_sha256']==sha(__file__) and plan['source_plan_sha256']==closure(),'Exact source40 runtime controller/schema/closure required')
 require(canonical(plan['registry_entry_preservation'])==canonical(registry_binding()),'Exact165+C1404+shared2 current entry preservation required')
 require(plan['registry_sha256']==sha(ROOT/'evals/configs/models.yaml'),'Exact current canonical registry association required')
 closed=spec.candidate_binding(plan['prepared']);require(canonical(closed)==canonical(plan['candidate_binding']),'Actual current correctedSDK/C140/upload association differs');m=closed['prepared'];args,env,_=config(plan['prepared'])
 require(numeric.expected_stage_ranges(args)==[tuple(v)for v in plan['recipe']['stages'].values()],'Exact source40 stage topology required')
 require(plan['prepared_sha256']==sha(Path(plan['prepared'])/'prepared.json') and plan['server_config_sha256']==sha(Path(plan['prepared'])/'server-config.json'),'Actual prepared bytes changed')
 require(plan['args']==args and plan['base_env']==env and plan['image']==m['runtime']['image'] and plan['pack']==m['pack'] and plan['engine_root']==str(Path(m['engine_receipt']).resolve().parent) and plan['cards']==m['cards'] and canonical(plan['recipe'])==canonical(spec.recipe(m['cards'])),'Exact complete actual source40 recipe differs')
 require(canonical(plan['kernel_source_flags'])==canonical(kernel_binding(plan['engine_root'])),'Actual source40 QSA source/object/link flags changed')
 require(plan['captured_operands_used']is False and plan['full_model_math_qualified']is False,'Source-only scope cannot grant model math')
 require(canonical(own_binding(plan['own_producer_root']))==canonical(plan['own_producer_binding']),'Independent own original input/reference proof changed')
 return m

def command(plan,directory,pid,on,render_gid):
 binding=sha(Path(directory).parent/'plan.snapshot.json');env=spec.arm_environment(plan['base_env'],on,binding,'/results/qsa');engine=Path(plan['engine_root']);model=ROOT/read(HERE/'model-lock.json')['destination'];name='b70-qsa40-'+str(pid)+'-'+('on'if on else'off')
 argv=['docker','run','-i','--name',name,'--label','b70.qsa40.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--entrypoint','/bin/bash','--group-add',str(render_gid)]
 for path,target,mode in [(engine/'build','/build','ro'),(engine/'source','/src','ro'),(Path(plan['pack']),'/pack','ro'),(model,'/model','ro'),(Path(directory),'/results','rw')]:argv+=['-v',str(path.resolve())+':'+target+':'+mode]
 for key,value in sorted(env.items()):argv+=['-e',key+'='+value]
 shell='exec 2>&1; cd /src; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; exec '+shlex.join(['/build/strata','--serve']+plan['args'])
 return argv+[plan['image'],'-c',shell],env,name

def inspection(obj,argv,plan):
 name=argv[argv.index('--name')+1];require(obj['Name']=='/'+name and obj['Image']==plan['image'] and obj['Config']['Image']==plan['image'],'Actual owned QSA40 name/image differs');cfg=obj['Config'];h=obj['HostConfig'];label=argv[argv.index('--label')+1].split('=',1)
 require(cfg['Labels'].get(label[0])==label[1] and cfg['User']=='1000:1000' and cfg['Entrypoint']==['/bin/bash'] and cfg['Cmd']==argv[argv.index(plan['image'])+1:] and cfg['OpenStdin']is True and cfg['Tty']is False,'Actual owned label/shell/user/interactive recipe differs')
 require(h['NetworkMode']=='none' and not h['Privileged'] and h['Memory']==h['MemorySwap']==105*1024**3 and h.get('NanoCpus',0)==0 and h.get('DeviceRequests')in(None,[]) and h['Devices']==[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}] and h['GroupAdd']==[argv[argv.index('--group-add')+1]],'Actual device/resource/network recipe differs')
 expected=[];env={}
 for i,word in enumerate(argv):
  if word=='-v':
   source,target,mode=argv[i+1].rsplit(':',2);expected.append((source,target,mode=='rw','bind'))
  if word=='-e':key,value=argv[i+1].split('=',1);env[key]=value
 require(sorted((m['Source'],m['Destination'],m['RW'],m['Type'])for m in obj['Mounts'])==sorted(expected),'Actual exact source/SDK/pack/model/output mounts differ');effective=dict(v.split('=',1)for v in cfg['Env']);require(all(effective.get(k)==v for k,v in env.items())and not any(k in effective for k in spec.ABSENT),'Actual full environment/math presence differs')
 return obj

def arm(plan,directory,on):
 for name in ('captures','p30','ple-input','qsa'):(directory/name).mkdir()
 gid=os.stat('/dev/dri/renderD128').st_gid;argv,env,name=command(plan,directory,os.getpid(),on,gid);write(directory/'command.json',argv);write(directory/'environment.json',env)
 protocol=None;active=[];rows=[];errors=[];removed=False;obj=None;forced=False
 def cleanup():
  nonlocal forced
  forced=True
  if c.absent(name):return
  obj=c.inspected(name);inspection(obj,argv,plan);write(directory/'cleanup-inspection.json',obj)
  if obj['State']['Running']:subprocess.run(['docker','stop','--time','15',name],check=True,capture_output=True,timeout=30)
  obj=c.inspected(name);inspection(obj,argv,plan);require(not obj['State']['Running'],'Owned QSA engine still running');subprocess.run(['docker','rm',name],check=True,capture_output=True);require(c.absent(name),'Owned QSA cleanup removal missing')
 try:
  protocol=Protocol(argv,directory,cleanup,active=active);(directory/'ARM').write_text('ARM exact fresh source40 prefix4\n',encoding='ascii')
  for row in plan['recipe']['requests']:
   raw=protocol.request('prefix4-'+str(row['ordinal']),row['ids'],fresh=1,max_new=1,pin=None);write(directory/('request-'+str(row['ordinal'])+'.json'),raw);rows.append({'ordinal':row['ordinal'],'raw':raw});write(directory/'requests.json',rows)
 except BaseException as e:errors.append(type(e).__name__+': '+str(e))
 finally:
  protocol=protocol or(active[0]if active else None)
  rc=protocol.close(failed=bool(errors))if protocol else None
  if not c.absent(name):
   obj=c.inspected(name);inspection(obj,argv,plan);write(directory/'inspection.json',obj)
   if not obj['State']['Running']:subprocess.run(['docker','rm',name],check=True,capture_output=True);removed=c.absent(name)
   else:cleanup()
  else:removed=True
 result={'schema':1,'passed':not errors and rc==0 and removed and not forced and obj is not None and obj['State']['ExitCode']==0 and obj['State']['OOMKilled']is False and not obj['State'].get('Error'),'errors':errors,'engine_rc':rc,'removed':removed,'forced_cleanup':forced,'inspection':obj,'producer_pid':os.getpid(),'render_gid':gid,'on':on,'stdout_capture':protocol.capture if protocol else None,'finished_epoch':time.time()};write(directory/'result.json',result);require(result['passed'],'Actual owned QSA40 arm failed');return result

def child(a):
 from native_qsa40_byte_epoch_v6 import ByteEpoch
 for sig in(signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,lambda number,frame:(_ for _ in()).throw(InterruptedError('Controlled child stop '+str(number))))
 plan=read(a.plan);require(os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0')and os.path.samefile('/proc/self/fd/9','/mnt/vm_8tb/b70/gpu.lock.1'),'Inherited owned pair leases required');require(plan['source_plan_sha256']==closure()and plan['driver_sha256']==sha(__file__),'Exact source/driver before byte access required');write(a.output/'plan.snapshot.json',plan)
 lock=read(HERE/'model-lock.json');shards=[ROOT/lock['destination']/x['path']for x in lock['files']if x['path'].startswith('UD-Q4_K_XL/')];source_files=[ROOT/p if not Path(p).is_absolute()else Path(p)for p in read(PLAN)['files']]+[PLAN]
 epoch=ByteEpoch(plan,source_files,shards);manifest(plan);ready_bytes=epoch.ready();write(a.output/'byte-ready.json',ready_bytes)
 from batch54_health_handshake_v1 import pid_identity,write_new
 ready={'pid':pid_identity(os.getpid()),'parent':pid_identity(os.getppid()),'plan_sha256':sha(a.plan),'manifest_completed_epoch':time.time(),'source_plan_sha256':closure(),'byte_READY_sha256':sha(a.output/'byte-ready.json')};write_new(a.output/'READY.json',ready)
 deadline=time.monotonic()+10800
 while not(a.output/'ACK.json').exists():require(time.monotonic()<deadline,'Actual parent ACK deadline');require(pid_identity(os.getppid())==ready['parent'],'Owned parent changed');time.sleep(.1)
 ack=read(a.output/'ACK.json');health=read(a.output.parent/(('on'if a.on else'off')+'-pre-health.json'));require(health['passed']is True and ack['health_sha256']==sha(a.output.parent/(('on'if a.on else'off')+'-pre-health.json'))and health['finished_epoch']==ack['health_finished_epoch'],'Actual original healthACK differs');require(ack['ready_sha256']==sha(a.output/'READY.json')and ack['plan_sha256']==ready['plan_sha256'],'Actual semantic READY/ACK association differs')
 # All unchanged historical predicates ran above before READY. The live owned
 # epoch now re-reads current SDK/pack/source/evidence bytes, never old parsers.
 from native_qsa40_evidence_v6 import health as admit_health,journal as admit_journal
 label='on'if a.on else'off';prejournal=read(a.output.parent/(label+'-pre-kernel.receipt.json'));admit_health(a.output.parent,label+'-pre',health,ready['manifest_completed_epoch']);admit_journal(a.output.parent,label+'-pre',prejournal,read(a.output.parent/'parent-start.json')['started_epoch'],health['finished_epoch']);require(prejournal['finished_epoch']<=ack['ack_epoch']and prejournal['sha256']==ack['journal_sha256'],'Actual raw journalACK differs')
 seal_bytes=epoch.predevice();write(a.output/'byte-seal.json',seal_bytes);require(closure()==plan['source_plan_sha256'],'Current source changed after healthACK');require(pid_identity(os.getppid())==ready['parent']and pid_identity(os.getpid())==ready['pid']and os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0')and os.path.samefile('/proc/self/fd/9','/mnt/vm_8tb/b70/gpu.lock.1'),'Actual parent/child/lease changed after complete seal');seal=time.time();require(0<=seal-ack['health_finished_epoch']<=300,'Actual health/current complete-byte seals older than300s; no launch');write(a.output/'GPU-seal.json',{'finished_epoch':seal,'plan_sha256':sha(a.plan),'manifest_current':True,'current_byte_seal_sha256':sha(a.output/'byte-seal.json')})
 result=arm(plan,a.output,a.on);write(a.output/'byte-epoch.json',epoch.finish());return result

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True);p=sub.add_parser('prepare');p.add_argument('--prepared',type=Path,required=True);p.add_argument('--own-producer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p=sub.add_parser('child');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--on',action='store_true');a=ap.parse_args();prepare(a)if a.mode=='prepare'else child(a)
if __name__=='__main__':main()
