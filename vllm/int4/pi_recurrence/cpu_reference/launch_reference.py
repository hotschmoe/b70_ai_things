"""CPU-only owned Docker wrapper for an explicitly approved frozen reference plan."""
import argparse
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path


def command(argv,**kwargs):
    return subprocess.run(argv,text=True,capture_output=True,check=True,timeout=60,**kwargs).stdout


def is_missing_container(returncode, stderr):
    message = stderr.lower()
    return returncode != 0 and any(text in message for text in ('no such object:', 'no such container:'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=Path(a.directory).resolve();launch=json.loads((root/'launch.json').read_text())
    planpath=root/'inputs/plan.json'
    if hashlib.sha256(planpath.read_bytes()).hexdigest()!=launch['plan_sha256']:raise ValueError('Plan hash')
    for relative,sha in launch['frozen_files'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=sha:raise ValueError(('Frozen file',relative))
    plan=json.loads(planpath.read_text());image=plan['image']
    if not a.execute:
        print(json.dumps({'prepared':True,'image':image,'timeout':launch['timeout_seconds'],'no_execution':True}));return
    evidence=root/'execution';evidence.mkdir(exist_ok=False)
    output=Path(launch['host_output']);output.mkdir(parents=True,exist_ok=False)
    imageinfo=json.loads(command(['docker','image','inspect',image]))[0]
    if imageinfo['Id']!=image:raise ValueError('Runtime image identity')
    (evidence/'image-inspect.json').write_text(json.dumps(imageinfo,indent=2)+'\n')
    name='b70-cpu-reference-'+uuid.uuid4().hex[:12]
    cmd=['docker','create','--name',name,'--network','none','--cpus','8','--memory','12g',
      '--user','1000:1000','-e','OMP_NUM_THREADS=8','-e','MKL_NUM_THREADS=8',
      '-v',str(root/'source')+':/source:ro','-v',str(root/'inputs')+':/inputs:ro',
      '-v',launch['host_model']+':/model:ro','-v',str(output)+':/out',
      '--entrypoint','python',image,'/source/run_reference.py','--plan','/inputs/plan.json',
      '--plan-sha256',launch['plan_sha256'],'--run-full']
    cid=command(cmd).strip();rc=None;timed_out=False;cleanup=False
    try:
        inspect=json.loads(command(['docker','inspect',cid]))[0];hc=inspect['HostConfig']
        if inspect['Image']!=image or hc.get('Devices') or hc.get('DeviceRequests') or hc['NetworkMode']!='none' or hc['Memory']!=12*1024**3 or hc['NanoCpus']!=8*10**9:
            raise ValueError('Container isolation mismatch')
        (evidence/'container-inspect.json').write_text(json.dumps(inspect,indent=2)+'\n')
        with (evidence/'stdout.log').open('w') as log:
            try:rc=subprocess.run(['docker','start','--attach',cid],stdout=log,stderr=subprocess.STDOUT,timeout=launch['timeout_seconds']).returncode
            except subprocess.TimeoutExpired:timed_out=True
        if not timed_out:
            stopped=json.loads(command(['docker','inspect',cid]))[0]
            (evidence/'container-final-inspect.json').write_text(json.dumps(stopped,indent=2)+'\n')
            if stopped['State']['Running']:raise ValueError('Container still running after attach')
            rc=stopped['State']['ExitCode']
    finally:
        subprocess.run(['docker','rm','-f',cid],capture_output=True,text=True,timeout=60)
        check=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=60)
        cleanup=is_missing_container(check.returncode, check.stderr)
        (evidence/'lifecycle.json').write_text(json.dumps({'exit_code':rc,'timed_out':timed_out,'container_removed':cleanup,'container_id':cid,'finished':time.time()},indent=2)+'\n')
    result=output/'result/result.json'
    if rc!=0 or timed_out or not cleanup or not result.exists() or json.loads(result.read_text()).get('passed') is not True:
        raise SystemExit(1)
if __name__=='__main__':main()
