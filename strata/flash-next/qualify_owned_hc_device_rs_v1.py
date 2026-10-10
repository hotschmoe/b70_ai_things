#!/usr/bin/env python3
"""ROOT-only fresh build/GPU lifecycle. Source preparation executes no runtime."""
import argparse,hashlib,json,os,re,shlex,subprocess,sys,threading,time,traceback,signal
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'owned-hc-device-rs-source-plan-v1.json'
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
RUNTIME_IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
SETVARS_PREFIX='set -e; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; '
CONTAINER_RUNNER='/harness/strata/flash-next/qualifier.py'
PROPOSAL_SHA='31d294c1412e79444a45d1a8dd90ea856712fc1a9a2799b562e6964618ea6602'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_bytes())
def write(p,r):Path(p).write_text(json.dumps(r,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inside_runtime(helper,output,maps=None):
 text=subprocess.check_output(['ldd',str(helper)],text=True);require('not found' not in text,'Actual intended runtime ELF library unresolved');paths={x for line in text.splitlines() for x in line.split() if x.startswith('/')}
 if maps:
  for line in Path(maps).read_text().splitlines():
   cols=line.split(maxsplit=5)
   if len(cols)==6 and cols[5].startswith('/') and '.so' in Path(cols[5]).name:paths.add(cols[5])
 require(paths,'Actual runtime dependency roster missing');rows={str(Path(p).resolve()):{'sha256':sha(Path(p).resolve()),'bytes':Path(p).stat().st_size} for p in sorted(paths)};write(output,{'helper_sha256':sha(helper),'libraries':rows,'mapped_GPU_libraries_included':maps is not None,'environment':dict(os.environ),'actual_model_payload_read':False})

def modules():
 import native_rms_rsqrt37_proposal_v1 as p
 import serial37_stdout_capture_v2 as capture
 from source_page_watchdog_v3 import guard,preserve
 from run_source_upload_oracle_full_v2 import full_buffered_identity
 import c137_baseline_admission_v4 as baseline
 return p,capture,guard,preserve,full_buffered_identity,baseline

def source_binding():
 row=read(PLAN)
 for n,w in row['files'].items():require(sha(ROOT/n)==w,'Owned RMS source changed '+n)
 from native_rms_entry_derivation_v7 import admit as entry_admit
 derivation=entry_admit(HERE)
 import owned_hc_device_rs_experiment_v1 as experiment
 operations=experiment.source_binding()
 p,*_=modules();require(sha(p.PLAN)==PROPOSAL_SHA,'Frozen preregistered RMS proposal changed');p.source_binding();return {'source_plan_sha256':sha(PLAN),'files':row['files'],'preregistered_proposal_sha256':PROPOSAL_SHA,'entry_derivation':derivation,'device_operation_preregistration':operations}

def fixture_binding(root):
 p,*_=modules();root=Path(root).resolve();r=read(root/'fixture.json');require(r['source_binding']==p.source_binding() and r['builder_flags']==p.builder_flags() and r['compile_argv']==p.compile_argv(),'Actual preregistration source/flags changed');residual,b=p.saved_original_input();require(r['original_saved_input_binding']==b and (root/'residual.f32').read_bytes()==residual.tobytes(),'Original saved residual fixture changed')
 expected=p.mathref.variants(residual);require(set(r['hypotheses'])==set(expected),'Complete preregistered families required')
 for family,fields in expected.items():
  require(set(r['hypotheses'][family])==set(fields),'Complete preregistered variant roster differs')
  for name,raw in fields.items():
   path=root/(family+'-'+name+'.f32');item=r['hypotheses'][family][name];require(item['path']==str(path) and item['sha256']==sha(path) and item['bytes']==len(raw) and path.read_bytes()==raw,'Preregistered hypothesis bytes changed')
 return {'fixture_sha256':sha(root/'fixture.json'),'root':str(root),'input_sha256':sha(root/'residual.f32'),'record':r}

def observed_container(obj,name,image,command):
 require(obj['Config'].get('OpenStdin',False)==('-i' in command) and obj['Config'].get('Tty',False) is False,'Actual helper interactive stdin/TTY differs');require(name==command[command.index('--name')+1],'Actual container name differs from fullcommand');require(obj['Name']=='/'+name and obj['Image']==image and obj['Config']['Image']==image and obj['Config']['Labels'].get('b70.rms37.plan')==sha(PLAN),'Owned RMS image/name/source label differs');h=obj['HostConfig'];require(h['NetworkMode']=='none' and not h.get('Privileged') and h['Memory']==h['MemorySwap']==2<<30 and h['NanoCpus']==2000000000 and h['PidsLimit']==256,'Owned RMS exact bounds differ');require(h.get('DeviceRequests') in (None,[]) and h['Devices']==[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}],'Owned RMS exact GPU grant differs');require(obj['Config']['User']=='1000:1000' and obj['Config']['Entrypoint']==['/bin/bash'] and obj['Config']['Cmd']==command[command.index(image)+1:],'Actual owned shell/user/argv differs')
 expected=[];env={}
 for i,word in enumerate(command):
  if word=='-v':
   source,destination,mode=command[i+1].rsplit(':',2);expected.append((str(Path(source).resolve()),destination,mode=='rw','bind'))
  if word=='-e':
   key,value=command[i+1].split('=',1);env[key]=value
 actual=sorted((m['Source'],m['Destination'],m['RW'],m['Type']) for m in obj['Mounts']);require(actual==sorted(expected) and all('/model' not in m[1] and '/pack' not in m[1] for m in actual),'Exact source/input/output-only mount association differs')
 effective={k:v for k,v in (word.split('=',1) for word in obj['Config']['Env'])};require(all(effective.get(k)==v for k,v in env.items()) and 'STRATA_VERIFY_EAGER' not in effective,'Actual affinity/selector/observer/EAGER presence differs');return obj

def docker_recipe(name,image,mounts,env,shell):
 argv=['docker','run','--name',name,'--label','b70.rms37.plan='+sha(PLAN),'--network','none','--device','/dev/dri','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'--user','1000:1000','--memory','2g','--memory-swap','2g','--cpus','2','--pids-limit','256','--entrypoint','/bin/bash']
 for src,dst,mode in mounts:argv+=['-v',str(src)+':'+dst+':'+mode]
 for key,val in sorted(env.items()):argv+=['-e',key+'='+val]
 return argv+[image,'-c',shell]

def expected_recipes(out,fixture_root,pid):
 p,*_=modules();out=Path(out).resolve();build=out/'build';argv=p.compile_argv();argv[argv.index('/leaf/native_rms_rsqrt37_gpu_v1.cpp')]='/leaf/owned_hc_rsqrt37_service_v1.cpp'
 compile=docker_recipe('b70-rms37-build-'+str(pid),p.IMAGE,[(p.SDK/'source','/sdk/source','ro'),(p.SDK/'build','/sdk/build','ro'),(HERE,'/leaf','ro'),(build,'/out','rw')],{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu'},SETVARS_PREFIX+'/opt/intel/oneapi/compiler/2026.1/bin/icpx --version; exec '+shlex.join(argv))
 shell=SETVARS_PREFIX+'/opt/b70-c1-python/bin/python /harness/strata/flash-next/qualifier.py --inside-runtime /out/native-rms-rsqrt37 --binding-output /results/runtime-before.json; /out/native-rms-rsqrt37 --maximum-requests 512 --maps-prefix /out/own-reference; /opt/b70-c1-python/bin/python /harness/strata/flash-next/qualifier.py --inside-runtime /out/native-rms-rsqrt37 --binding-output /results/runtime-after.json --maps /out/own-reference.after'
 runtime=docker_recipe('b70-rms37-run-'+str(pid),RUNTIME_IMAGE,[(build,'/out','rw'),(Path(fixture_root).resolve(),'/inputs','ro'),(out,'/results','rw'),(Path(__file__).resolve(),CONTAINER_RUNNER,'ro')],{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_UR_TRACE':'0','SYCL_CACHE_PERSISTENT':'0'},shell)
 runtime.insert(2,'-i');return argv,compile,runtime

def command_binding(row,path,command_path,receipt_path):
 from serial37_canonical_json_v3 import matches_saved
 require(matches_saved(row,receipt_path),'Report command row contradicts original saved typed receipt')
 require(row['phase_cleanup_error'] is None,'Actual phase cleanup error cannot qualify')
 require(row['passed'] is True and row['reader_retired'] is True and row['eof'] is True and row['error'] is None and row['command_error'] is None and row['return_code']==0 and row['path']==str(path) and sha(path)==row['sha256'] and read(command_path)==row['command'] and sha(command_path)==row['command_sha256'],'Actual command/EOF/retirement/log association changed')
 return row

def runtime_receipt_binding(before,after,helper):
 require(before['helper_sha256']==after['helper_sha256']==sha(helper) and all(after['libraries'].get(k)==v for k,v in before['libraries'].items()) and after['mapped_GPU_libraries_included'] is True,'Original runtime ELF/library subset/postexecution map invariant differs')
 return {'mapped_library_roster_independently_reconstructed_from_raw_maps':False}

def terminal_receipt_binding(obj,path):
 from serial37_canonical_json_v3 import matches_saved
 require(matches_saved(obj,path),'Report terminal contradicts original saved typed inspection')
 terminal_binding(obj['State']);return obj

def terminal_binding(state):
 require(state['Running'] is False and state['ExitCode']==0 and not state['OOMKilled'] and not state.get('Error'),'Actual normal owned terminal differs');return state

def chronology(report):
 import math
 epochs=[report['GPU_terminal_epoch'],report['post_health']['finished_epoch'],report['post_journal']['started_command_epoch'],report['post_journal']['finished_epoch'],report['post_full4']['started'],report['post_full4']['finished'],report['before_post_full4_pages']['epoch'],report['post_pages']['epoch'],report['finished_epoch']]
 require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in epochs),'Finite actual owned GPU/source chronology required')
 require(report['run_command']['finished_epoch']<=report['GPU_terminal_epoch']<=report['post_health']['finished_epoch']<=report['post_journal']['started_command_epoch']<=report['post_journal']['finished_epoch']<=report['post_full4']['started']<=report['post_full4']['finished']<=report['post_pages']['epoch']<=report['finished_epoch'],'Actual terminal/health/journal/new4 chronology invalid');require(report['before_post_full4_pages']['epoch']<=report['post_full4']['started'],'Both-page view must bracket actual full4')
 return {'actual_ordered_terminal_health_journal_new4':True,'fitted_epoch_used':False}

def finalized_binding(root):
 root=Path(root).resolve();r=read(root/'report.json');require(r['passed'] is True and r['errors']==[] and r['actual_leaf_execution_observed'] is True and r['source_binding']==source_binding(),'Actual owned RMS runtime/source prerequisites failed');require(read(root/'source-plan.snapshot.json')==read(PLAN) and (root/'qualifier.snapshot.py').read_bytes()==Path(__file__).read_bytes(),'Actual owned producer/source snapshot changed');from native_rms_health_journal_binding_v3 import admit as health_admit
 health_admit(r,ROOT,HEALTH)
 from owned_hc_device_rs_experiment_v1 import prior_binding as admit_prior
 prior_proof=admit_prior(r['prior_rms_binding']['root']);require(prior_proof==r['prior_rms_binding'],'Actual prior RMS proof changed')
 fixture=fixture_binding(r['fixture_binding']['root']);argv,compile,runtime=expected_recipes(root,fixture['root'],r['producer_pid']);require(r['compile_argv']==argv and r['compile_command']['command']==compile and r['run_command']['command']==runtime,'Actual complete fresh compile/runtime recipes differ');require(fixture==r['fixture_binding'],'Actual original fixture changed');chronology(r)
 for n,w in r['artifact_sha256'].items():require(not (root/n).is_symlink() and sha(root/n)==w,'Original owned RMS artifact changed '+n)
 for label,row in [('compile',r['compile_command']),('runtime',r['run_command'])]+[(name,row) for stage in ('pre','post') for name,row in zip((stage+'-strict-health',stage+'-compiled-health'),r[stage+'_health']['rows'])]+[(stage+'-kernel',r[stage+'_journal']) for stage in ('pre','post')]:
  require(label+'.receipt.json' in r['artifact_sha256'],'Original command receipt absent from artifact closure');command_binding(row,root/(label+'.log'),root/(label+'.command.json'),root/(label+'.receipt.json'))
 from native_rms_kernel_journal_v5 import admit as kernel_admit
 kernel_admit(root,r)
 for stage,name in [('compile','compile'),('run','runtime')]:
  require(name+'-inspection.json' in r['artifact_sha256'],'Original terminal inspection absent from artifact closure');obj=r[stage+'_terminal'];terminal_receipt_binding(obj,root/(name+'-inspection.json'));observed_container(obj,obj['Name'][1:],modules()[0].IMAGE if stage=='compile' else RUNTIME_IMAGE,r[stage+'_command']['command']);terminal_binding(obj['State'])
 p,capture,guard,preserve,full4,baseline=modules();_,current=baseline.finalized_binding(r['baseline_binding']['baseline_root'],r['baseline_binding'].get('adjudication_receipt'));require(current==r['baseline_binding'],'Current genuine SDK37/C137 baseline changed')
 lock=read(HERE/'model-lock.json');from native_rms_device_ops_lifecycle_v8 import native_trace,docker_epochs,pre_publisher
 docker_epochs(r);require('pre-full4.json' in r['artifact_sha256'],'Original pre full4 absent from artifact closure');pre_publisher(root,r,lock,sha(HERE/'model-lock.json'),ROOT)
 shards=[ROOT/lock['destination']/x['path'] for x in lock['files'] if x['path'].startswith('UD-Q4_K_XL/')];require(len(shards)==4 and r['post_full4']==read(root/'post-full4.json') and r['post_full4']['passed'] is True and len(r['post_full4']['rows'])==4,'Original newfull4 source record changed');require(r['post_full4']['lock_sha256']==sha(HERE/'model-lock.json') and r['post_full4']['model_revision']==lock['revision'],'Original publisher lock/revision differs')
 for row,path in zip(r['post_full4']['rows'],shards):
  stat=path.stat();require(row['passed'] is True and row['path']==str(path) and row['stat_before']==row['stat_after']==[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns],'Current original model stat/source changed')
 from native_rms_publisher_binding_v2 import publisher_binding
 publisher_binding(root,r,lock,sha(HERE/'model-lock.json'),ROOT)
 require(guard(shards[2])['passed'],'Current source knownpage guard failed')
 require(r['helper_sha256']==sha(root/'build/native-rms-rsqrt37') and read(root/'runtime-before.json')==r['runtime_binding_before'],'Actual fresh helper/runtime-before receipt changed')
 actual_after=read(root/'runtime-after.json');actual_after['transient_unmapped_libraries_observed']=False;require(actual_after==r['runtime_binding_after'],'Actual runtime-after receipt changed')
 runtime_receipt_binding(r['runtime_binding_before'],r['runtime_binding_after'],root/'build/native-rms-rsqrt37')
 for row in r['runtime_binding_after']['libraries'].values():require(row['sha256'] and row['bytes']>0,'Actual intended runtime resolved library record absent')
 from owned_hc_device_rs_experiment_v1 import recollect
 from owned_hc_device_rs_work_v1 import OriginalWork
 context=OriginalWork(r['original_work_config']['first_native_root'],r['original_work_config']['model_identity'],r['original_work_config']['bulk_root'],r['original_work_config']['prefix_native_root'],root/'reference-work');work=read(root/'reference-work/work-report.json');require(__import__('serial37_canonical_json_v3').canonical(work)==__import__('serial37_canonical_json_v3').canonical(r['reference_phase']['work']),'Original CPU work report changed');recollect(root/'runtime.log',r['reference_phase'],work,root/'reference-work');context.recheck(work);require(source_binding()==r['source_binding'] and fixture_binding(fixture['root'])==fixture,'Source/current original inputs changed during readonly work');require(admit_prior(prior_proof['root'])==prior_proof,'Actual prior V8 changed during readonly comparison')
 return {'report_sha256':sha(root/'report.json'),'actual_owned_device_RS_reference_observed':True,'reference_first_gate':work['first_gate'],'prefix4_attempted':work['prefix4_attempted'],'full_model_math_qualified':False,'normal_model_graph_qualified':False,'latency_qualified':False,'mapped_library_roster_independently_reconstructed_from_raw_maps':False}

def main():
 a=argparse.ArgumentParser();a.add_argument('--inside-runtime',type=Path);a.add_argument('--binding-output',type=Path);a.add_argument('--maps',type=Path);a.add_argument('--fixture',type=Path);a.add_argument('--prepared',type=Path);a.add_argument('--baseline-adjudication',type=Path);a.add_argument('--output',type=Path);a.add_argument('--prior-rms-root',type=Path);a.add_argument('--first-hc-native-root',type=Path);a.add_argument('--prefix4-native-root',type=Path);a.add_argument('--model-identity',type=Path);a.add_argument('--bulk-build-root',type=Path);a.add_argument('--leased',action='store_true');args=a.parse_args()
 if args.inside_runtime:inside_runtime(args.inside_runtime,args.binding_output,args.maps);return 0
 require(args.fixture and args.prepared and args.output and args.prior_rms_root and args.first_hc_native_root and args.model_identity and args.bulk_build_root,'Explicit original fixture/current C137/new output/closed V8 required');from owned_hc_device_rs_experiment_v1 import prior_header as header,prior_binding as admit_prior
 header(args.prior_rms_root)
 binding=source_binding();fixture=fixture_binding(args.fixture);prior_proof=admit_prior(args.prior_rms_root);p,capture,guard,preserve,full4,baseline=modules();prepared,baseline_proof=baseline.finalized_binding(args.prepared,args.baseline_adjudication);require(Path(prepared['engine_receipt']).resolve().parent==p.SDK and prepared['cards']==[0],'Exact current SDK37/card0 C137 baseline required')
 if not args.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 for card in (0,1):require(os.path.samefile('/proc/self/fd/'+str(8+card),'/mnt/vm_8tb/b70/gpu.lock.'+str(card)),'Pair leases for compiled pre/post health required; leaf card0 only')
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Owned RMS parent terminated')))
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);write(out/'source-plan.snapshot.json',read(PLAN));(out/'qualifier.snapshot.py').write_bytes(Path(__file__).read_bytes());lock=read(HERE/'model-lock.json');shards=[ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];require(len(shards)==4,'Original four-shard roster required');started=time.time();report={'schema':1,'producer_pid':os.getpid(),'passed':False,'started_epoch':started,'source_binding':binding,'fixture_binding':fixture,'prior_rms_binding':prior_proof,'original_work_config':{'first_native_root':str(args.first_hc_native_root.resolve()),'prefix_native_root':str(args.prefix4_native_root.resolve()) if args.prefix4_native_root else None,'model_identity':str(args.model_identity.resolve()),'bulk_root':str(args.bulk_build_root.resolve())},'baseline_binding':baseline_proof,'errors':[],'workload_cards':[0],'health_cards':[0,1],'GPU_execution_requested':True,'actual_leaf_execution_observed':False,'full_model_math_qualified':False,'normal_model_graph_qualified':False,'device_intrinsics_qualified':False};names=[];phase_recipes={};active_commands=[];terminal=None;health_invoked=False
 def command(argv,label,timeout,phase=None):
  from native_rms_phase_supervisor_v3 import supervise
  path=out/(label+'.log');write(out/(label+'.command.json'),argv)
  callback=(lambda:cleanup_exact_phase(*phase)) if phase is not None else None
  row=supervise(argv,path,timeout,callback,active_commands);row['command_sha256']=sha(out/(label+'.command.json'));write(out/(label+'.receipt.json'),row);require(row['passed'],'Actual command deadline/EOF/owned phase failed '+label);return row
 def cleanup_exact_phase(name,image,recipe,kind):
  ids=subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True).strip()
  if not ids:return
  obj=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0];observed_container(obj,name,image,recipe);write(out/(kind+'-cleanup-original-inspection.json'),obj)
  if obj['State']['Running']:subprocess.run(['docker','stop','--time','15',name],check=True,capture_output=True,timeout=30)
  obj=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0];observed_container(obj,name,image,recipe);require(not obj['State']['Running'],'Exact owned phase still running');write(out/(kind+'-cleanup-terminal.json'),obj);subprocess.run(['docker','rm',name],check=True,capture_output=True);require(not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True).strip(),'Exact owned phase removal absent')
 def health(stage):
  nonlocal health_invoked
  health_invoked=True
  rows=[command([str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],stage+'-strict-health',210),command([str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'],stage+'-compiled-health',240)];return {'passed':True,'finished_epoch':time.time(),'rows':rows,'image':HEALTH}
 def journal(stage):
  row=command(['journalctl','-k','--since','@'+str(int(started)),'--no-pager'],stage+'-kernel',30);from native_rms_kernel_journal_v5 import reject_faults;reject_faults((out/(stage+'-kernel.log')).read_text());return row
 def terminal_owned(name,kind):
  obj=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0];observed_container(obj,name,p.IMAGE if kind=='compile' else RUNTIME_IMAGE,report['compile_command' if kind=='compile' else 'run_command']['command']);require(not obj['State']['Running'] and obj['State']['ExitCode']==0 and not obj['State']['OOMKilled'] and not obj['State'].get('Error'),'Actual owned RMS zero-exit terminal required');write(out/(kind+'-inspection.json'),obj);subprocess.run(['docker','rm',name],check=True,capture_output=True);require(not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True).strip(),'Owned RMS removal unobserved');return obj
 try:
  report['pre_pages']=preserve(shards[2],out,'pre');report['pre_full4']=full4(HERE/'model-lock.json',lock,shards,out/'pre-full4.json',time.time());report['pre_full4_after_pages']=preserve(shards[2],out,'pre-after-full4');require(report['pre_pages']['passed'] and report['pre_full4_after_pages']['passed'] and report['pre_full4']['passed'],'Original current pre-source/full4/pages prerequisite failed beforedevice');report['pre_health']=health('pre');from native_rms_device_ops_lifecycle_v8 import native_trace,docker_epochs,pre_publisher;report['pre_publisher_admission']=pre_publisher(out,report,lock,sha(HERE/'model-lock.json'),ROOT);report['pre_journal']=journal('pre');build=out/'build';build.mkdir();name='b70-rms37-build-'+str(os.getpid());names.append(name);argv=p.compile_argv();argv[argv.index('/leaf/native_rms_rsqrt37_gpu_v1.cpp')]='/leaf/owned_hc_rsqrt37_service_v1.cpp';report['compile_argv']=argv
  recipe=docker_recipe(name,p.IMAGE,[(p.SDK/'source','/sdk/source','ro'),(p.SDK/'build','/sdk/build','ro'),(HERE,'/leaf','ro'),(build,'/out','rw')],{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu'},SETVARS_PREFIX+'/opt/intel/oneapi/compiler/2026.1/bin/icpx --version; exec '+shlex.join(argv));phase_recipes[name]=(p.IMAGE,recipe,'compile');report['compile_command']=command(recipe,'compile',300,(name,p.IMAGE,recipe,'compile'));report['compile_terminal']=terminal_owned(name,'compile');helper=build/'native-rms-rsqrt37';require(helper.read_bytes()[:4]==b'\x7fELF','Fresh RMS helper ELF required');report['helper_sha256']=sha(helper);require(source_binding()==binding,'Source/SDK changed during fresh compile')
  name='b70-rms37-run-'+str(os.getpid());names.append(name);shell=SETVARS_PREFIX+'/opt/b70-c1-python/bin/python /harness/strata/flash-next/qualifier.py --inside-runtime /out/native-rms-rsqrt37 --binding-output /results/runtime-before.json; /out/native-rms-rsqrt37 --maximum-requests 512 --maps-prefix /out/own-reference; /opt/b70-c1-python/bin/python /harness/strata/flash-next/qualifier.py --inside-runtime /out/native-rms-rsqrt37 --binding-output /results/runtime-after.json --maps /out/own-reference.after';recipe=docker_recipe(name,RUNTIME_IMAGE,[(build,'/out','rw'),(args.fixture.resolve(),'/inputs','ro'),(out,'/results','rw'),(Path(__file__).resolve(),CONTAINER_RUNNER,'ro')],{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_UR_TRACE':'0','SYCL_CACHE_PERSISTENT':'0'},shell);recipe.insert(2,'-i');phase_recipes[name]=(RUNTIME_IMAGE,recipe,'runtime');from owned_hc_device_rs_bridge_v1 import run_phase
  from owned_hc_device_rs_work_v1 import OriginalWork
  context=OriginalWork(args.first_hc_native_root,args.model_identity,args.bulk_build_root,args.prefix4_native_root,out/'reference-work');write(out/'runtime.command.json',recipe)
  def start_admission(ready):
   obj=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0];observed_container(obj,name,RUNTIME_IMAGE,recipe);require(obj['State']['Running'],'Actual own helper READY requires running owned phase');write(out/'runtime-running-inspection.json',obj);return {'helper_source_sha256':sha(HERE/'owned_hc_rsqrt37_service_v1.cpp'),'operation':'sycl::rsqrt','captured_operands_used':False,'ULP_adjustment_or_lookup_used':False,'actual_owned_device_started':True,'synthetic_only':False,'actual_source_flags_and_recipe_qualified':True}
  row,phase=run_phase(recipe,out/'runtime.log',context,lambda:cleanup_exact_phase(name,RUNTIME_IMAGE,recipe,'runtime'),start_admission,active_commands);row['command_sha256']=sha(out/'runtime.command.json');write(out/'runtime.receipt.json',row);report['run_command']=row;report['reference_phase']=phase;require(row['passed'],'Actual owned own-reference EOF/phase failed');report['run_terminal']=terminal_owned(name,'runtime');terminal=time.time();report['GPU_terminal_epoch']=terminal;report['actual_leaf_execution_observed']=True;report['Docker_timestamp_admission']=docker_epochs(report)
  before=read(out/'runtime-before.json');after=read(out/'runtime-after.json');require(before['helper_sha256']==after['helper_sha256']==sha(helper) and all(after['libraries'].get(k)==v for k,v in before['libraries'].items()) and after['mapped_GPU_libraries_included'] is True,'Actual fresh ELF/current dynamic libraries changed');report['runtime_binding_before']=before;after['transient_unmapped_libraries_observed']=False;report['runtime_binding_after']=after;from owned_hc_device_rs_experiment_v1 import recollect;report['reference_ledger_admission']=recollect(out/'runtime.log',phase,phase['work'],out/'reference-work');report['first_gate']=phase['work']['first_gate'];report['prefix4_attempted']=phase['work']['prefix4_attempted']
 except BaseException as error:report['errors'].append(type(error).__name__+': '+str(error));report['failure_traceback']=traceback.format_exc()
 finally:
  for name in names:
   while subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True).strip():
    try:
     image,recipe,kind=phase_recipes[name];cleanup_exact_phase(name,image,recipe,kind);report['errors'].append('Owned phase required failure cleanup '+name)
    except BaseException as error:
     # Never remove a foreign image/recipe, and never drop leases while the
     # unresolved named phase remains. Operator may resolve it externally.
     text='cleanup retains leases: '+str(error)
     if text not in report['errors']:report['errors'].append(text);write(out/'report.json',report)
     time.sleep(1)
  terminal=terminal or time.time()
  try:
   require(health_invoked,'Posthealth/new4 scope unobserved: pre-source failure before anydeviceaction');require(all(proc.poll() is not None and not reader.is_alive() for proc,reader,label in active_commands),'Posthealth/source cannot outrun actual command/EOF retirement');require(all(not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+n+'$'],text=True).strip() for n in names),'Posthealth/source blocked by owned residual container');report['post_health']=health('post');report['post_journal']=journal('post');report['before_post_full4_pages']=preserve(shards[2],out,'post-before-full4');report['post_full4']=full4(HERE/'model-lock.json',lock,shards,out/'post-full4.json',max(terminal,report['post_health']['finished_epoch'],report['post_journal']['finished_epoch']));report['post_pages']=preserve(shards[2],out,'post-after-full4');require(report['pre_pages']['passed'] and report['before_post_full4_pages']['passed'] and report['post_pages']['passed'] and report['pre_full4']['passed'] and report['post_full4']['passed'],'Actual source publisher/pages failed');require(source_binding()==binding and fixture_binding(args.fixture)==fixture,'Original inputs/SDK/source changed');_,postbaseline=baseline.finalized_binding(args.prepared,args.baseline_adjudication);require(postbaseline==baseline_proof,'Current C137 baseline changed');require(admit_prior(args.prior_rms_root)==prior_proof,'Actual prior V8 changed during owned runtime')
  except BaseException as error:report['errors'].append('post-proof: '+str(error))
  report['finished_epoch']=time.time()
  if not report['errors']:
   try:report['chronology']=chronology(report);from native_rms_health_journal_binding_v3 import admit as health_admit;report['exact_health_journal_admission']=health_admit(report,ROOT,HEALTH);from native_rms_publisher_binding_v2 import publisher_binding;report['publisher_admission']=publisher_binding(out,report,lock,sha(HERE/'model-lock.json'),ROOT)
   except Exception as error:report['errors'].append('chronology: '+str(error))
  report['passed']=not report['errors'] and 'reference_phase' in report;report['artifact_sha256']={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='report.json'};write(out/'report.json',report);print(json.dumps({'passed':report['passed'],'full_model_math_qualified':False,'output':str(out)}))
 return int(not report['passed'])
if __name__=='__main__':raise SystemExit(main())
