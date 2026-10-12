"""Saved numeric-component recovery; never changes the failed producer report."""
import argparse,hashlib,json,re
from pathlib import Path
import run_original_mtp53_shifted_v2 as runtime
ROOT=Path(__file__).resolve().parents[2]
PLAN=Path(__file__).with_name('original-mtp53-shifted-recovery-source-plan-v1.json')
def sha(p):return runtime.sha(p)
def read(p):return json.loads(Path(p).read_bytes())
def need(v,s):runtime.require(v,s)
def finalized_binding(root,expected_report_sha256):
 root=Path(root);need(root.is_absolute()and root.resolve()==root,'Canonical saved root required')
 plan=read(PLAN)
 paths={ROOT/p:h for p,h in plan['files'].items()};paths.update({root/p:h for p,h in plan['saved_artifacts'].items()})
 for p,h in paths.items():need(p.is_file()and not p.is_symlink()and sha(p)==h,'Pinned original source/artifact changed '+str(p))
 before={str(p):(runtime.stat5(p),sha(p))for p in paths}
 try:
  report=read(root/'report.json');need(sha(root/'report.json')==expected_report_sha256,'Original failed report changed')
  need(report['passed']is False and report['errors']==['Retirement Exact fixture command/read-only original-only mounts required','Completion Actual normal original process/container retirement required'],'Only documented observer failure may recover')
  need(report['source_plan_sha256']==sha(runtime.PLAN.with_name('original-mtp53-shifted-runtime-source-plan-v1.json')),'Authentic V1 producer source required')
  need(report['forced_cleanup']is False and report['owned_removed']is True and report['original_foreground_EOF']is True and type(report['original_client_return_code'])is int and report['original_client_return_code']==0,'Normal original creator EOF required')
  command=read(root/'native.command.json');terminal=read(root/'coordinator-terminal-before-removal-v1.json');recovery=read(root/'coordinator-retirement-recovery-v1.json');client=read(root/'client.json')
  name=command[command.index('--name')+1];label=command[command.index('--label')+1];key,owner=label.split('=',1)
  need(terminal['Name']=='/'+name and terminal['Config']['Labels'][key]==owner and terminal['Config']['Image']==runtime.RUNTIME and terminal['Config']['Entrypoint']==['/bin/bash'] and terminal['Config']['Cmd']==command[-2:],'Exact original Docker owner/command required')
  binds=[command[i+1]for i,x in enumerate(command)if x=='-v'];need(runtime.exact_binds(terminal['HostConfig']['Binds'],binds)and len(binds)==2,'Exact sorted original mounts required')
  need(binds[1]==str(runtime.MTP)+':/original.gguf:ro'and binds[0].endswith('/output:/harness:ro'),'Original-only RO fixture mounts required')
  host=terminal['HostConfig'];need(host['Privileged']is False and host['NetworkMode']=='none'and not host.get('DeviceRequests')and not host.get('DeviceCgroupRules'),'Exact device domain required')
  need(sorted((d['PathOnHost'],d['PathInContainer'],d['CgroupPermissions'])for d in host['Devices'])==sorted((p,p,'rwm')for p in ['/dev/dri/card0','/dev/dri/renderD128','/dev/dri/card1','/dev/dri/renderD129']),'Exact paired device pins required')
  state=terminal['State'];need(all(state[k]is False for k in ['Running','Paused','Restarting','Dead','OOMKilled'])and type(state['ExitCode'])is int and state['ExitCode']==0 and type(state['Pid'])is int and state['Pid']==0 and state['Error']==''and type(terminal['RestartCount'])is int and terminal['RestartCount']==0,'Typed actual normal terminal required')
  need(recovery['passed']is True and recovery['failed_original_controller_preserved']is True and recovery['original_container_id']==terminal['Id'] and recovery['original_terminal_sha256']==sha(root/'coordinator-terminal-before-removal-v1.json')and recovery['original_creator_PID']==client['pid']and type(client['pid'])is int and client['session']==client['pid']and recovery['original_creator_exited']is True and type(recovery['rm_return_code'])is int and recovery['rm_return_code']==0,'Authentic coordinator removal/creator settlement required')
  need((root/'native.stdout').stat().st_size+(root/'native.stderr').stat().st_size<=4<<20 and (root/'native.stdout').read_text('ascii').count(runtime.MARKER)==1,'Exact numeric marker and bounded EOF output required')
  prior=report['started_epoch']
  for phase in ['pre-kernel','pre-health','pre-collective','post-health','post-collective','post-kernel']:
   result=read(root/(phase+'.result.json'));need(type(result['return_code'])is int and result['return_code']==0 and result['finished_epoch']>=result['started_epoch']>=prior,'Successful ordered original checks required '+phase)
   for stream in ['stdout','stderr']:need(sha(root/(phase+'.'+stream))==result[stream+'_sha256'],'Raw original check changed')
   prior=result['finished_epoch']
  need(read(root/'post-health.result.json')['started_epoch']>=recovery['epoch'] and report['finished_epoch']>=prior,'Recovery must precede actual closing checks')
  need(report['post_health_passed']is True and report['kernel_window_passed']is True and not re.search(r'Fault response|Faulted Address|Engine reset|Timedout job|segfault|GPU HANG|CAT error',(root/'post-kernel.stdout').read_text('ascii'),re.I),'Original post-health/kernel failed')
  for phase in ['pre','post']:
   identity=read(root/(phase+'-original-MTP.json'));need(identity['passed']is True and identity['sha256']==identity['expected_sha256']==runtime.MTP_SHA and identity['before']==identity['after'],'Actual whole original identity check required')
  identity=read(root/'post-original-MTP.json');need(runtime.stat5(runtime.MTP)==identity['after'],'Current original file metadata changed; no current payload rehash claimed')
  buildroot=Path(binds[0].split(':/harness:ro')[0]);build=read(buildroot/'receipt.json')
  need(sha(buildroot/'receipt.json')==report['build_receipt_sha256']and sha(buildroot/'original-mtp53-shifted-loaded')==report['elf_sha256']and build['passed']is True and build['CPU_contract_executed']is True and build['CPU_contract_passed']is True and type(build['CPU_contract_controls'])is int and build['CPU_contract_controls']==4,'Original actual build/CPU4 required')
  for name,h in build['outputs'].items():need(sha(buildroot/name)==h,'Original compiled output changed')
  for stream in ['stdout','stderr']:need(sha(buildroot/('checkpoint-cpu.'+stream))==build['CPU_contract_'+stream+'_sha256'],'Actual CPU evidence changed')
  return {'schema':1,'numeric_component_admitted':True,'original_producer_passed':False,'original_report_sha256':expected_report_sha256,'coordinator_recovery_sha256':sha(root/'coordinator-retirement-recovery-v1.json'),'source_plan_sha256':sha(PLAN),'synthetic_only':True,'MAIN_executed':False,'math_qualified':False,'API_qualified':False,'current_original_payload_rehashed':False,'saved_pre_post_original_payload_identity':True}
 finally:
  for p,h in paths.items():need((runtime.stat5(p),sha(p))==before[str(p)],'Closing saved evidence changed '+str(p))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(a.output.is_absolute()and a.output.resolve()==a.output and not a.output.exists()and a.root.resolve()not in a.output.parents,'New external output required');value=finalized_binding(a.root,a.expected_report_sha256);a.output.write_text(json.dumps(value,indent=2)+'\n',encoding='ascii');print(json.dumps(value))
