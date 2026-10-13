"""Protected root NoDri actual HTTP identity load; no API/cache qualification."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-mtp53-http-identity-preflight-source-plan-v1.json'
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
PYTHON='/opt/b70-c1-python/bin/python'
def need(v,s):
 if not v:raise ValueError(s)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def stat5(p):
 s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def absent(stderr,name):
 if type(stderr)is not bytes or len(stderr)>4096:return False
 try:text=stderr.decode('ascii').strip().lower()
 except UnicodeDecodeError:return False
 return any(text==prefix+'no such '+kind+': '+name.lower()for prefix in('','error: ','error response from daemon: ')for kind in('container','object'))
def validate_terminal(row,command):
 name=command[command.index('--name')+1];key,owner=command[command.index('--label')+1].split('=',1);cfg=row['Config'];host=row['HostConfig']
 need(row['Name']=='/'+name and cfg['Labels'].get(key)==owner and cfg['Image']==IMAGE and cfg['Entrypoint']==['/bin/bash']and cfg['Cmd']==command[-2:]and cfg['User']==command[command.index('--user')+1],'Exact original preflight owner/command required')
 mounts=[command[i+1]for i,x in enumerate(command)if x=='-v'];need(type(host['Binds'])is list and all(type(x)is str for x in host['Binds'])and sorted(host['Binds'])==sorted(mounts)and all(x.endswith(':ro')for x in mounts),'Exact readonly bind multiset required')
 need(host['Privileged']is False and host['NetworkMode']=='none'and not host.get('Devices')and not host.get('DeviceRequests')and not host.get('DeviceCgroupRules')and not host.get('GroupAdd')and not host.get('CapAdd'),'Exact NoDri unprivileged domain required')
 need(all(type(host[k])is int and host[k]==v for k,v in {'NanoCpus':2000000000,'Memory':2<<30,'MemorySwap':2<<30}.items())and type(row['RestartCount'])is int and row['RestartCount']==0,'Typed CPU2/RAM2 runtime bounds required')
 return row

def main(a):
 plan=read(PLAN)
 for p,h in plan['files'].items():need(sha(ROOT/p)==h,'Preflight source changed '+p)
 need(os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0'),'Root-coordinated card0 FD8 NoDri preflight required')
 root=a.prepared;need(root.is_absolute()and root.resolve()==root and sha(root/'receipt.json')==a.prepared_receipt_sha256,'Authentic prepared HTTP identity directory/receipt required')
 prepared=read(root/'receipt.json');need(prepared['source_plan_sha256']==sha(HERE/'original-mtp53-http-identity-source-plan-v1.json')and prepared['NoDri_required']is True and prepared['preflight_starts_native_engine']is False and prepared['preflight_EXECUTED']is False,'Prepared actual HTTP identity/load-only recipe required')
 need(sha(root/'manifest.json')==prepared['manifest_sha256']and sha(root/'config.json')==prepared['config_sha256'],'Original HTTP manifest/config bytes changed')
 manifest=read(root/'manifest.json');cfg=read(root/'config.json');need(cfg['model_name']=='hotschmoe-dd'and cfg['aliases']==[manifest['research_alias']]and manifest['primary_model_name']=='hotschmoe-dd'and cfg['artifact_identity_manifest']=='/http/manifest.json','Actual stable primary/alias/manifest contract required')
 original_command=prepared['preflight_command'];command=list(original_command)
 expected_script='if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; exec '+PYTHON+' /preflight.py --config /http/config.json --manifest-sha256 '+prepared['manifest_sha256']
 need(command[:2]==['docker','run']and command[-3:]==[IMAGE,'-lc',expected_script]and command[command.index('--entrypoint')+1]=='/bin/bash'and command[command.index('--network')+1]=='none'and command[command.index('--cpus')+1]=='2'and command[command.index('--memory')+1]==command[command.index('--memory-swap')+1]=='2g'and all(k not in command for k in ('--device','--device-cgroup-rule','--gpus','--privileged','--detach','-d')),'Exact prepared guarded setvars/qualified-venv foreground NoDri command required')
 need([command[i+1]for i,x in enumerate(command)if x=='-v']==[source+':'+dest+':ro'for source,dest in prepared['readonly_mounts'].items()],'Exact declared readonly mount roster required')
 # Only a fresh owned runtime name/label replace the preparation process identity.
 command[command.index('--name')+1]='b70-mtp-http-identity-run-'+str(os.getpid());command[command.index('--label')+1]='b70.mtp.http.identity='+str(os.getpid())
 inputs={root/'receipt.json':a.prepared_receipt_sha256,root/'manifest.json':prepared['manifest_sha256'],root/'config.json':prepared['config_sha256']}
 mounts={dest:Path(source)for source,dest in prepared['readonly_mounts'].items()}
 def host_path(container):
  cp=Path(container);need(cp.is_absolute()and '..'not in cp.parts,'Absolute nonescaping consumed container path required');choices=[dest for dest in mounts if cp==Path(dest)or Path(dest)in cp.parents];need(choices,'Consumed container file has no declared readonly backing');dest=max(choices,key=len);p=mounts[dest]/cp.relative_to(dest);need(p.resolve()==p and p.is_file()and not p.is_symlink(),'Canonical declared consumed host file required');return p
 for rel,want in manifest['runtime']['python_sources'].items():p=host_path('/src/'+rel);need(sha(p)==want,'Actual modified API/tokenizer Python source changed '+rel);inputs[p]=want
 need(set(manifest['runtime']['python_sources'])=={'serve/batch_request_identity.py','serve/server.py','serve/frontend.py','serve/artifact_identity.py','tools/strata_tokenizer.py','tools/gguf_reader.py','serve/original_mtp53_http_compound.py'},'Exact modified seven-source identity roster required')
 for rel,want in manifest['tokenizer_files'].items():p=host_path(cfg['tokenizer']+'/'+rel);need(sha(p)==want,'Actual original tokenizer/template export changed');inputs[p]=want
 exe=host_path(cfg['exe']);need(sha(exe)==manifest['runtime']['exe_sha256'],'Actual original custom native ELF changed');inputs[exe]=manifest['runtime']['exe_sha256']
 preflight=host_path('/preflight.py');need(preflight==HERE/'preflight_original_mtp53_http_identity_v1.py','Exact source-owned actual load_identity script required');inputs[preflight]=sha(preflight)
 original_GGUF=host_path(manifest['model']['tokenizer_gguf']);original_stat=stat5(original_GGUF)
 out=a.output;need(out.is_absolute()and out.resolve()==out and not out.exists(),'New canonical preflight result root required');out.mkdir()
 name=command[command.index('--name')+1];proc=None;terminal=None;removed=False;forced=False;attempted=False;signals=[]
 receipt={'schema':1,'passed':False,'started_epoch':time.time(),'errors':[],'source_plan_sha256':sha(PLAN),'HTTP_prepared_receipt_sha256':a.prepared_receipt_sha256,'HTTP_manifest_sha256':prepared['manifest_sha256'],'HTTP_config_sha256':prepared['config_sha256'],'original_prepared_command_sha256':hashlib.sha256(json.dumps(original_command,separators=(',',':')).encode('ascii')).hexdigest(),'runtime_image':IMAGE,'devices_exposed':False,'owned_removed':False,'original_EOF':False,'return_code':None,'GGUF_payload_read_by_root_NoDri_load_identity':True,'GGUF_payload_read_by_agent':False,'native_engine_started':False,'API_qualified':False,'cache_qualified':False,'GPU_executed':False,'command_path':str(out/'command.json'),'stdout_path':str(out/'stdout.txt'),'stderr_path':str(out/'stderr.txt'),'terminal_path':str(out/'terminal.json')}
 write(out/'command.json',command)
 def error(e):
  s=str(e)
  if s not in receipt['errors']and len(receipt['errors'])<64:receipt['errors'].append(s)
 def inspect():
  r=subprocess.run(['docker','inspect',name],capture_output=True,timeout=20)
  if r.returncode:need(absent(r.stderr,name),'Unknown original Docker state; ownership retained');return None
  rows=json.loads(r.stdout);need(type(rows)is list and len(rows)==1,'Exact single original Docker inspection required');return validate_terminal(rows[0],command)
 saved={n:signal.signal(n,lambda number,frame:signals.append(number))for n in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
 try:
  need(inspect()is None,'Preflight owned name must be absent before launch')
  with(out/'stdout.txt').open('wb')as so,(out/'stderr.txt').open('wb')as se:
   attempted=True;proc=subprocess.Popen(command,stdout=so,stderr=se,start_new_session=True);write(out/'client.json',{'pid':proc.pid,'session':os.getpgid(proc.pid),'started_epoch':time.time()});deadline=time.monotonic()+600
   while proc.poll()is None:
    if signals or time.monotonic()>deadline or (out/'stdout.txt').stat().st_size+(out/'stderr.txt').stat().st_size>4<<20:raise RuntimeError('Preflight signal/deadline/output failure')
    time.sleep(.1)
   need(proc.returncode==0,'Actual API artifact validator failed; raw output retained')
 except BaseException as e:error(type(e).__name__+': '+str(e))
 finally:
  if attempted:
   while True:
    try:
     row=inspect()
     if row and row['State']['Running']:
      forced=True;error('Forced original preflight cleanup');r=subprocess.run(['docker','stop','--time','10',row['Id']],capture_output=True,timeout=30);need(r.returncode==0,'Original stop unresolved');continue
     if proc is None:
      # An ambiguous attempted Popen has no successful EOF receipt. Keep
      # ownership until root settles it; never release on name absence alone.
      raise RuntimeError('Attempted launch has no returned creator; ownership retained')
     if proc.poll()is None:time.sleep(.1);continue
     if row:
      terminal=row
      try:write(out/'terminal.json',row)
      except BaseException as e:error('Terminal evidence IO '+str(e))
      r=subprocess.run(['docker','rm',row['Id']],capture_output=True,timeout=20);need(r.returncode==0,'Original removal unresolved')
      try:write(out/'removal.json',{'id':row['Id'],'return_code':r.returncode,'stdout':r.stdout.decode('ascii',errors='replace'),'stderr':r.stderr.decode('ascii',errors='replace'),'epoch':time.time()})
      except BaseException as e:error('Removal evidence IO '+str(e))
      continue
     proc.wait();removed=True;break
    except BaseException as e:
     error('Retirement '+type(e).__name__+': '+str(e))
     try:write(out/'retirement-pending.json',{'error':str(e),'name':name,'ownership_retained':True,'epoch':time.time()})
     except BaseException:pass
     time.sleep(1)
  if signals:error('Signals latched '+str(signals))
  try:
   need(proc is not None and proc.returncode==0 and terminal is not None and removed and not forced,'Original normal preflight EOF/terminal/removal required')
   state=terminal['State'];need(type(state['ExitCode'])is int and state['ExitCode']==0 and type(state['Pid'])is int and state['Pid']==0 and all(state[k]is False for k in ['Running','Paused','Restarting','Dead','OOMKilled'])and state['Error']=='','Typed normal NoDri terminal required')
   result=read(out/'stdout.txt');need(result['passed']is True and result['manifest_sha256']==prepared['manifest_sha256']and result['actual_load_identity_executed']is True and result['actual_validate_runtime_executed']is True and result['actual_engine_args_and_child_env_executed']is True and result['qualified_interpreter']==PYTHON and result['primary_model_name']=='hotschmoe-dd'and result['research_aliases']==[manifest['research_alias']]and result['original_first_GGUF_full_hash_executed']is True and result['export_vs_original_GGUF_metadata_checked']is True and result['native_engine_started']is False and result['GPU_exposed']is False and result['API_qualified']is False and result['cache_qualified']is False,'Actual modified load_identity/runtime/first GGUF/export/alias result required');need((out/'stderr.txt').read_bytes()==b'','Actual HTTP preflight stderr must remain clean');receipt['actual_HTTP_identity_result']=result
   need((out/'stdout.txt').stat().st_size+(out/'stderr.txt').stat().st_size<=4<<20,'Closing preflight output bound')
   receipt.update(owned_removed=True,original_EOF=True,return_code=proc.returncode,forced_cleanup=False)
  except BaseException as e:error('Completion '+str(e))
  for p,h in inputs.items():
   try:need(sha(p)==h,'Closing exact artifact dependency changed '+str(p))
   except BaseException as e:error('Binding closure '+str(e))
  for p,h in plan['files'].items():
   try:need(sha(ROOT/p)==h,'Closing preflight source changed')
   except BaseException as e:error('Source closure '+str(e))
  try:need(stat5(original_GGUF)==original_stat,'Original tokenizer GGUF changed across actual full-hash/metadata read');receipt['original_tokenizer_GGUF_stat5_before']=original_stat;receipt['original_tokenizer_GGUF_stat5_after']=stat5(original_GGUF)
  except BaseException as e:error('Original GGUF closure '+str(e))
  for number,handler in saved.items():signal.signal(number,handler)
  receipt['finished_epoch']=time.time();receipt['passed']=not receipt['errors']
  for key in ('command','stdout','stderr','terminal'):
   p=Path(receipt[key+'_path']);receipt[key+'_sha256']=sha(p)if p.is_file()else None
  write(out/'receipt.json',receipt)
 print(json.dumps({'passed':receipt['passed'],'errors':receipt['errors'],'receipt':str(out/'receipt.json')},ensure_ascii=True));return 0 if receipt['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--prepared-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);raise SystemExit(main(p.parse_args()))
