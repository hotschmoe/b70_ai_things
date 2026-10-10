#!/usr/bin/env python3
"""Bounded queried-bundle admission: new ELF, no direct-launch/JIT history claim."""
import hashlib,json,math,re,shlex,stat,struct
from pathlib import Path
from native_rms_kernel_journal_v5 import reject_faults
ROOT=Path('/mnt/vm_8tb/github/b70_ai_things')
HERE=Path(__file__).resolve().parent
SOURCE_SHA='9712e1ccf9dbe2eece5c28db8b8347f632c9ac02e9446c18d29af22fadf0ab1d'
BINARY_SHA='c2c34be0ef0df97d2dad298c76be8d1be2a3001c3b02828fb899824013a7aa49'
ORIGINAL_REPORT_SHA='5e66575dcf49c66fd1118c9af14e118217fa672ea5ed42217e506cba1b110cc3'
ORIGINAL=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/owned-indexer-half-control-v2-run-v1')
INPUT=ORIGINAL.parent/'owned-indexer-half-inputs-v2-prepared-v1/raw.f32'
INPUT_SHA='de82f9d41a59b874c1973cd95d9a51091ccbcc3911b4fa01e51ee7e338f39226'
NAMES={'_ZTS16Half37Expression','_ZTS11Half37Store','_ZTS10Half37Load'}
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
EXPECTED_COMPILE_ARGV=['/opt/intel/oneapi/compiler/2026.1/bin/icpx', '-O3', '-DNDEBUG', '-std=c++20', '-fsycl', '-fsycl-default-sub-group-size=32', '-fsycl-device-code-split=per_kernel', '-fp-model=precise', '-DSTRATA_SYCL_Q8_HC_BUILT=1', '-DSTRATA_VERSION="0.1.41"', '-I/sdk/source/third_party/ggml', '-I/sdk/source/sycl/include', '-I/sdk/source/include', '/leaf/half37_native_bundle_diagnostic_v1.cpp', '/sdk/build/libstrata_kernels.a', '/sdk/build/libstrata_core.a', '/usr/lib/x86_64-linux-gnu/libze_loader.so', '-Xsycl-target-backend=spir64', '-cl-fp32-correctly-rounded-divide-sqrt', '-o', '/out/half37-native-bundle', '-lze_loader']
PREFIX='set -e; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; '
def require(ok,msg):
 if not ok:raise ValueError(msg)
def unique(pairs):
 out={}
 for k,v in pairs:require(k not in out,'Duplicate JSON key');out[k]=v
 return out
def data(path,cap=32<<20):
 path=Path(path)
 for p in [path,*path.parents]:require(not p.is_symlink(),'Symlink evidence refused')
 a=path.stat();require(stat.S_ISREG(a.st_mode) and 0<a.st_size<=cap,'Bounded regular evidence required');raw=path.read_bytes();b=path.stat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns) and len(raw)==a.st_size,'Current evidence changed');return raw
def sha(path):return hashlib.sha256(data(path)).hexdigest()
def read(path):return json.loads(data(path),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def terminal(state):
 require(state['Status']=='exited' and state['Error']=='' and all(state[x]is False for x in ('Running','Paused','Restarting','OOMKilled','Dead')) and type(state['ExitCode'])is int and state['ExitCode']==0 and type(state['Pid'])is int and state['Pid']==0,'Exact normal inspected terminal required')
def receipt(root,label):
 r=read(root/(label+'.receipt.json'));command=read(root/(label+'.command.json'));require(r['passed']is True and type(r['return_code'])is int and r['return_code']==0 and r['error']is None and r['command_error']is None and r['phase_cleanup_error']is None and r['eof']is True and r['reader_retired']is True,'Actual command/EOF retirement failed')
 require(r['command']==command and r['command_sha256']==sha(root/(label+'.command.json')) and r['sha256']==sha(root/(label+'.log')) and r['path']==str(root/(label+'.log')),'Original command/log receipt differs')
 times=[r[x] for x in ('started_command_epoch','started_epoch','completed_epoch','finished_epoch')];require(all(type(t)in (int,float) and math.isfinite(t) for t in times) and times==sorted(times),'Actual command chronology differs');return r

def runtime_recipe(name,root,compile_root):
 require(re.fullmatch(r'b70-half37-jit-[1-9][0-9]*',name),'Exact owned runtime name required')
 argv=['docker','run','--name',name,'--label','b70.half37.plan=52dbb91d9886e4685bd733b064166c202c6f8ad7bbbaa167a6a82a2f2fa3861b','--network','none','--device','/dev/dri','--group-add','991','--user','1000:1000','--memory','2g','--memory-swap','2g','--cpus','2','--pids-limit','256','--entrypoint','/bin/bash']
 for source,dest,mode in [(compile_root,'/helper','ro'),(INPUT.parent,'/inputs','ro'),(root,'/out','rw'),(root/'igc','/tmp/igc','rw')]:argv+=['-v',str(source)+':'+dest+':'+mode]
 for key,val in sorted({'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_CACHE_PERSISTENT':'0','SYCL_UR_TRACE':'0','SYCL_PROGRAM_COMPILE_OPTIONS':''}.items()):argv+=['-e',key+'='+val]
 shell=PREFIX+'exec /helper/half37-native-bundle --raw /inputs/raw.f32 --output /out/native-output --maps-prefix /out/half37'
 return argv+[IMAGE,'-c',shell]

def parse(log,bundle):
 require('HALF37_ERROR' not in log and log.count('HALF37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 compiler_lowering_qualified=0 full_model_math_qualified=0')==1,'Actual helper terminal markers differ')
 for i in range(3):require(log.count(f'HALF37_FRAME route={i} graph_replay={i} fields=4 own_input_restored=1 values=512')==1,'Actual route marker differs')
 kernels=re.findall(r'^HALF37_BUNDLE_KERNEL name=(\S+) handle=(0x[0-9a-f]+) module_uuid=([0-9a-f]{32}) kernel_uuid=([0-9a-f]{32}) executable_bundle_membership=1 actual_direct_launch_handle_observed=0$',log,re.M)
 require(len(kernels)==3 and {x[0] for x in kernels}==NAMES and len({x[1] for x in kernels})==3,'Exact executable kernel ID roster required')
 modules=re.findall(r'^HALF37_BUNDLE_MODULE index=(\d+) handle=(0x[0-9a-f]+) bytes=(\d+) names=(\d+)$',log,re.M);names=re.findall(r'^HALF37_BUNDLE_NAME module=(\d+) name=(\S+)$',log,re.M)
 result=re.findall(r'^HALF37_BUNDLE_RESULT modules=(\d+) bytes=(\d+) new_ELF_only=1 historical_JIT_observed=0 actual_direct_launch_module_proven=0$',log,re.M)
 require(len(result)==1 and 1<=len(modules)<=8 and int(result[0][0])==len(modules) and len({x[1] for x in modules})==len(modules),'Bounded distinct module roster differs')
 rows=[];total=0;assigned=[]
 for expected,(index,handle,size,count) in enumerate(modules):
  require(int(index)==expected and 0<int(count)<=512,'Module index/name bounds differ');selected=[name for i,name in names if int(i)==expected];require(len(selected)==int(count) and len(set(selected))==len(selected),'Complete module kernel names differ');assigned+=selected
  p=bundle/('module-'+str(expected)+'.bin');raw=data(p,16<<20);require(len(raw)==int(size) and len(raw)>=64 and raw[:7]==b'\x7fELF\x02\x01\x01' and struct.unpack_from('<H',raw,18)[0]==205,'Complete little-endian native IntelGT ELF required');total+=len(raw);require(total<=64<<20,'Total native byte quota exceeded');rows.append({'index':expected,'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'kernel_names':selected})
 require(set(assigned)==NAMES and len(assigned)==3 and total==int(result[0][1]) and {p.name for p in bundle.iterdir()}=={Path(x['path']).name for x in rows},'Exact three-role native file/name/byte roster differs')
 return {'modules':rows,'kernel_properties':[{'name':x[0],'handle':x[1],'module_uuid':x[2],'kernel_uuid':x[3]}for x in kernels],'executable_bundle_membership_observed':True,'actual_direct_launch_module_association_proven':False,'historical_JIT_code_observed':False}

def collect(root,compile_root):
 root=Path(root).absolute();compile_root=Path(compile_root).absolute();report=read(root/'report.json');before=sha(root/'report.json')
 require(report['passed']is True and report['error']is None and report['all_original_output_bytes_equal']is True and report['new_ELF_only']is True and report['original_direct_launch_module_association_proven']is False and report['original_historical_JIT_code_observed']is False,'Original new-ELF replay report scope differs')
 roster=report['artifact_sha256'];require(set(roster)=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='report.json'},'Original full artifact roster differs')
 for path,want in roster.items():require(sha(root/path)==want,'Original replay artifact changed')
 require(sha(HERE/'half37_native_bundle_diagnostic_v1.cpp')==SOURCE_SHA and sha(compile_root/'half37-native-bundle')==BINARY_SHA==report['helper_sha256'],'Current new source/helper bytes differ')
 compile_binding=read(compile_root/'compile-binding.json');require(compile_binding['passed']is True and compile_binding['source_sha256']==SOURCE_SHA and compile_binding['binary_sha256']==BINARY_SHA and compile_binding['command_sha256']==sha(compile_root/'command.json') and compile_binding['actual_kernel_executed']is False and compile_binding['original_JIT_proven']is False,'Actual compile association differs')
 compile_obj=read(compile_root/'inspection.json');terminal(compile_obj['State']);require(compile_obj['Image']=='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','Actual compiler image differs');c=read(compile_root/'receipt.json');require(c['passed']is True and c['return_code']==0 and type(c['return_code'])is int and c['error']is None and c['eof']is True and c['reader_retired']is True and c['sha256']==sha(compile_root/'compile.log'),'Actual compiler terminal/log differs')
 require(c['command']==read(compile_root/'command.json') and c['command_sha256']==sha(compile_root/'command.json') and c['command_error']is None and c['phase_cleanup_error']is None,'Actual compiler command/retirement differs')
 require(read(compile_root/'argv.json')==EXPECTED_COMPILE_ARGV and compile_obj['Config']['Entrypoint']==['/bin/bash'] and compile_obj['Config']['Cmd']==['-c',PREFIX+'exec '+shlex.join(EXPECTED_COMPILE_ARGV)],'Exact compiler source/flags/link recipe differs')
 runtime=receipt(root,'runtime');obj=read(root/'runtime.inspection.json');terminal(obj['State']);require(obj['Image']==IMAGE and obj['Name'][1:]==runtime['command'][runtime['command'].index('--name')+1] and runtime['command']==runtime_recipe(obj['Name'][1:],root,compile_root),'Actual full runtime image/owner/recipe differs')
 require(obj['Config']['Entrypoint']==['/bin/bash'] and obj['Config']['Cmd']==runtime['command'][-2:] and obj['Config']['User']=='1000:1000' and obj['HostConfig']['NetworkMode']=='none','Actual inspected runtime configuration differs')
 for key in ('LD_PRELOAD','UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS'):require(not any(x.startswith(key+'=')for x in obj['Config']['Env']),'No interposer/adapter route forcing admitted')
 health={label:receipt(root,label)for label in ('pre-strict','pre-pair','post-strict','post-pair')};strict=[str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH];pair=[str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180']
 for label,r in health.items():require(r['command']==(pair if label.endswith('pair')else strict),'Actual health recipe differs')
 require(health['pre-strict']['finished_epoch']<=health['pre-pair']['started_command_epoch']<=health['pre-pair']['finished_epoch']<=runtime['started_command_epoch']<=runtime['finished_epoch']<=health['post-strict']['started_command_epoch']<=health['post-strict']['finished_epoch']<=health['post-pair']['started_command_epoch'],'Actual health/runtime chronology differs')
 journal=receipt(root,'kernel');require(journal['command'][:3]==['journalctl','-k','--since'] and len(journal['command'])==5 and journal['command'][4]=='--no-pager' and journal['command'][3].startswith('@') and int(journal['command'][3][1:])<=health['pre-strict']['started_command_epoch'] and journal['started_command_epoch']>=health['post-pair']['finished_epoch'],'Actual original journal window/chronology differs');reject_faults(data(root/'kernel.log').decode())
 require(sha(ORIGINAL/'report.json')==ORIGINAL_REPORT_SHA and sha(INPUT)==INPUT_SHA==report['original_input_sha256'],'Original input/history identity differs');old=read(ORIGINAL/'report.json');require(old['passed']is True,'Original raw control not qualified')
 comparisons=[]
 for route in range(3):
  for field in ('expression.f32','materialized.f32','materialized.f16','input.f32'):
   relative=f'build/native-output/route{route}/{field}';original=ORIGINAL/relative;actual=root/f'native-output/route{route}/{field}';require(sha(original)==old['artifact_sha256'][relative] and data(actual)==data(original),'Original/current complete raw output bytes differ');comparisons.append({'route':route,'field':field,'bytes':len(data(actual)),'sha256':sha(actual),'bitwise_equal':True})
 require(data(root/'native-output/consumed-raw.f32')==data(INPUT),'Exact own raw consumed input differs')
 proof=parse(data(root/'runtime.log').decode(),root/'native-output/native-bundle')
 for path,want in roster.items():require(sha(root/path)==want,'Replay artifact changed during collection')
 require(sha(root/'report.json')==before,'Original report changed during collection')
 proof.update(report_sha256=before,actual_raw12_comparisons=comparisons,new_ELF_sha256=BINARY_SHA,source_sha256=SOURCE_SHA,actual_replay_runtime_and_health_joined=True,mapped_library_current_bytes_recollected=False,maps_before_sha256=sha(root/'half37.before'),maps_after_sha256=sha(root/'half37.after'),full_model_math_qualified=False,serving_latency_qualified=False)
 return proof
