import json,os,subprocess,time,hashlib,shlex
from pathlib import Path
root=Path('/mnt/vm_8tb/github/b70_ai_things');base=Path('/mnt/vm_8tb/b70/build/original-mtp53-two-stage-compile-v2');recipe=json.loads((base/'compile-recipe.json').read_text());out=Path(recipe['mounts']['output_RW']);assert not (out/'receipt.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'started_epoch':time.time(),'devices_exposed':False,'model_payload_read':False,'recipe_sha256':sha(base/'compile-recipe.json'),'passed':False,'commands':[],'outputs':{}}
for fd,card in [(9,1)]:assert os.path.samefile('/proc/self/fd/'+str(fd),'/mnt/vm_8tb/b70/gpu.lock.'+str(card))
script='set -e\nif [ "${SETVARS_COMPLETED:-0}" != 1 ]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi\n'+'\n'.join(shlex.join(c['argv'])for c in recipe['compile_commands'])
name='b70-original-mtp53-compile-'+str(os.getpid());owner=str(os.getpid());label='b70.mtp53.compile'
argv=['docker','run','--name',name,'--label',label+'='+owner,'--network','none','--cpus','2','--memory','6g','--memory-swap','6g','--user',str(os.getuid())+':'+str(os.getgid()),'-v',recipe['mounts']['SDK_RO']+':/sdk:ro','-v',recipe['mounts']['overlay_RO']+':/overlay:ro','-v',str(out)+':/out:rw','--entrypoint','/bin/bash',recipe['compiler_image'],'-lc',script]
report['commands'].append(argv)
try:
 with(out/'compile.stdout').open('w')as stdout,(out/'compile.stderr').open('w')as stderr:r=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=300)
 report['compile_rc']=r.returncode;assert r.returncode==0
 for c in recipe['compile_commands']:report['outputs'][c['name']]=sha(out/c['name'])
 report['passed']=True
finally:
 q=subprocess.run(['docker','inspect',name],capture_output=True)
 if q.returncode==0:
  row=json.loads(q.stdout)[0];assert row['Config']['Labels'][label]==owner and not row['HostConfig']['Devices']
  if row['State']['Running']:subprocess.run(['docker','stop','--time','20',name],check=True,capture_output=True);row=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
  assert not row['State']['Running'];(out/'terminal.json').write_text(json.dumps(row,indent=2)+'\n',encoding='ascii');subprocess.run(['docker','rm',name],check=True,capture_output=True)
 report['finished_epoch']=time.time();(out/'controller.snapshot.py').write_bytes(Path(__file__).read_bytes());(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='ascii');print(json.dumps(report))
assert report['passed']
