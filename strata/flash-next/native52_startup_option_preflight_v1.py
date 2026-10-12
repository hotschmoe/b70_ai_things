"""Small source52 option-guard preflight; no model/device/ABI admission."""
import argparse,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROPOSAL=HERE/'native52-copy-engine-prefill-engine-build-proposal-v1.json';PROPOSAL_SHA='25c8b73f8fa0d92f0e7c562f232776844405dc62b7e3524d1584a92283f4630d'
GENERATE='sycl/src/program/generate.cpp';HEADER='sycl/include/strata/program/host_critical_path_trace.hpp'
GUARD='host_trace36::init(o.serve && o.batch==0 && o.pipeline_windows==0 && o.mtp.empty() && o.layer_split.empty() && o.split_device.empty() && o.suffix_draft==0 && o.lookup_chain==0)'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_binding(source_root):
 root=Path(source_root);require(root.is_absolute()and root.resolve()==root,'Canonical actual source root required');require(sha(PROPOSAL)==PROPOSAL_SHA,'Exact source52 proposal changed');plan=json.loads(PROPOSAL.read_bytes());bound={};bodies={}
 for name in(GENERATE,HEADER):
  p=root/name;require(p.is_file()and not p.is_symlink(),'Exact regular source file required');before=p.stat();raw=p.read_bytes();after=p.stat();sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns);digest=hashlib.sha256(raw).hexdigest();require(sig(before)==sig(after)and digest==plan['expected_patched_source_sha256'][name],'Exact native guard source changed '+name);bound[name]=digest;bodies[name]=raw.decode('ascii')
 require(bodies[GENERATE].count(GUARD)==1 and 'if(std::strcmp(flag,"1")!=0||!serial||std::getenv("STRATA_VERIFY_EAGER"))return false;'in bodies[HEADER]and 'if(!flag||std::strcmp(flag,"0")==0)return true;'in bodies[HEADER],'Expected actual source52 H36 branch changed')
 return {'source52_proposal_sha256':PROPOSAL_SHA,'files':bound,'source_option_guard_only':True,'SDK_ABI_or_model_admitted':False}

def check(recipe,source_root):
 binding=source_binding(source_root);argv=recipe['native_argv'];env=recipe['native_env'];require(type(argv)is list and all(type(x)is str for x in argv)and type(env)is dict,'Exact command/env shape required');require(argv.count('/build/strata')==1,'Single original native binary behind optional profiler wrapper required');args=argv[argv.index('/build/strata')+1:]
 def option(key,default='0'):
  require(args.count(key)<=1,'Duplicate relevant native option '+key)
  if key not in args:return default
  at=args.index(key);require(at+1<len(args)and not args[at+1].startswith('--'),'Missing relevant native option '+key);return args[at+1]
 def number(key,default='0'):
  require(key in args,'Current generated recipe must explicitly declare '+key)
  v=option(key,default);require(re.fullmatch(r'[0-9]+',v)is not None,'Exact generated decimal option required '+key);return int(v)
 h36=env.get('STRATA_CRITICAL_PATH_TRACE','0');require(h36 in('0','1'),'Actual H36 flag must be absent/0/1')
 serial='--serve'in args and number('--batch')==0 and number('--pipeline-windows')==0 and option('--mtp','')==''and option('--layer-split','')==''and option('--split-device','')==''and number('--suffix-draft')==0 and number('--lookup-chain')==0
 if h36=='1':
  require(serial and 'STRATA_VERIFY_EAGER'not in env,'Actual native H36 guard rejects these options BEFORE model startup');directory=env.get('STRATA_CRITICAL_PATH_TRACE_DIR','');digest=env.get('STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256','');require(type(directory)is str and directory.startswith('/')and len(directory.encode('ascii'))<4096 and re.fullmatch('[0-9a-f]{64}',digest)is not None,'Actual H36 DIR/binding declaration invalid')
 mirror=env.get('STRATA_VERIFY_MIRROR_STAGE_T1','0');require(mirror in('0','1'),'Actual mirror-T1 flag must be absent/0/1')
 if mirror=='1':require(number('--batch')==0 and option('--mtp','')==''and number('--spec','2')==2 and '--batch-mtp'not in args and number('--pipeline-windows')==0 and number('--suffix-draft')==0 and number('--lookup-chain')==0,'Actual native mirror-T1 startup guard rejects these options')
 return {'passed':True,'source_guard':binding,'H36_enabled':h36=='1','H36_serial_option_predicate':serial,'mirror_T1_enabled':mirror=='1','runtime_directory_allocation_and_other_startup_checks_pending':True,'GPU_touch_performed':False,'model_read_performed':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--construction',type=Path,required=True);p.add_argument('--expected-construction-sha256',required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists()and sha(a.construction)==a.expected_construction_sha256,'New output and exact existing construction receipt required');v=json.loads(a.construction.read_bytes());require(v['GPU_launch_performed']is False and v['pure_command_construction_passed']is True,'Original pure construction required');r={'schema':1,'original_construction_sha256':a.expected_construction_sha256,'arms':[check(row['recipe'],a.source_root)for row in v['commands']],'configuration_guard_passed':True,'actual_runtime_pass_claim':False};require(sha(a.construction)==a.expected_construction_sha256,'Construction changed');a.output.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n',encoding='ascii')
