"""Owned Dockerexec recipe/receipt API; caller retains original native lease."""
import hashlib,json,subprocess,time
from pathlib import Path
import run_decode40_host_profile_v3 as lifecycle
import pti_mapped_injection_census_v1 as helper
require=helper.require
HERE=Path(__file__).resolve().parent

def command(container_id,phase,sha):
 require(phase in('after_READY','after_last_DONE_before_QUIT'),'Exact census phase required');return ['docker','exec',container_id,'/usr/bin/env','-i','PATH=/usr/bin:/bin','LANG=C','LC_ALL=C','PYTHONDONTWRITEBYTECODE=1','/opt/b70-c1-python/bin/python','/pti-census.py','--output','/results/'+phase,'--phase',phase,'--expected-sha256',sha]
def run(directory,name,native_command,image,binding,phase,sha):
 # Existing complete native recipe inspection plus exact original ID before/after.
 obj=lifecycle.inspect_owned(name,native_command,image,binding);require(obj['State']['Running']is True and type(obj['State']['Pid'])is int and obj['State']['Pid']>0,'Exact live original native container required');require(any(x==str(HERE/'pti_mapped_injection_census_v1.py')+':/pti-census.py:ro'for x in obj['HostConfig']['Binds']),'Exact readonly helper mount required');require(any(x.endswith(':/pti:ro')for x in obj['HostConfig']['Binds']),'Readonly actual fresh install mount required');cmd=command(obj['Id'],phase,sha);started=time.time();proc=None
 try:
  proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True);output,_=proc.communicate(timeout=90);require(proc.returncode==0,'Owned metadata census exec failed')
 finally:
  # Metadata CLI cannot survive exceptions/timeout and release original lease.
  if proc is not None:
   # Never kill a CLI then guess a late Dockerexec child was canceled.
   while proc.poll()is None:
    try:proc.communicate(timeout=10)
    except BaseException:continue
   proc.communicate();proc.wait()
 after=lifecycle.inspect_owned(name,native_command,image,binding);require(after['Id']==obj['Id']and after['State']['Running']is True and after['State']['Pid']==obj['State']['Pid'],'Original native owner changed during census');root=Path(directory)/phase;receipt=json.loads((root/'receipt.json').read_bytes());require(receipt['passed']is True and receipt['sha256']==sha and receipt['phase']==phase and receipt['process']['pid']==1,'Actual original census receipt differs');raw=(root/'mapped-injection.elf').read_bytes();require(hashlib.sha256(raw).hexdigest()==sha and len(raw)==receipt['bytes'],'Actual exported host backing-file artifact differs')
 return {'passed':True,'command':cmd,'container_id':obj['Id'],'host_native_pid':obj['State']['Pid'],'started_epoch':started,'finished_epoch':time.time(),'return_code':proc.returncode,'original_container_path':receipt['original_container_path'],'exported_host_path':str(root/'mapped-injection.elf'),'exported_host_sha256':sha,'census':receipt,'metadata_stdout_bytes':len(output),'process_memory_bytes_read':False}
