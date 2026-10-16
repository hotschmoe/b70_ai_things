"""Root-only finite S112 v2 controller; inherited quietCPU/card0 lease, no force cleanup."""
from pathlib import Path
import argparse,grp,hashlib,json,os,re,subprocess,sys,time
P=Path(__file__).resolve().parent;R=P.parent;REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
IMAGE='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
HEALTH='b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78'
p=argparse.ArgumentParser()
for k in ['out','prep','exe','audit']:p.add_argument('--'+k,type=Path,required=True)
for k in ['sha256','audit-sha256','manifest-sha256']:p.add_argument('--'+k,required=True)
p.add_argument('--reuse-s107-prehealth',action='store_true');a=p.parse_args();O=a.out.resolve();prep=a.prep.resolve();exe=a.exe.resolve()
def sha(f):
 with f.open('rb')as s:return hashlib.file_digest(s,'sha256').hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2)+'\n',encoding='ascii')
def command(label,cmd):
 save(label+'-command.json',cmd)
 with(O/(label+'.stdout')).open('w')as out,(O/(label+'.stderr')).open('w')as err:r=subprocess.run(cmd,cwd=REPO,stdout=out,stderr=err)
 (O/(label+'.exit')).write_text(str(r.returncode)+'\n');return r.returncode
def identity():
 for n,h in json.loads((P/'harness-pins.json').read_text()).items():assert sha(P/n)==h,n
 assert prep.name=='greedy-top1-s112-oracle-prep-v2'and exe==prep/'build/greedy_top1_s112'
 assert os.getuid()==1000 and os.environ.get('ZE_AFFINITY_MASK')=='0'and os.environ.get('ONEAPI_DEVICE_SELECTOR')=='level_zero:*'
 assert sha(prep/'manifest.json')==a.manifest_sha256
 m=json.loads((prep/'manifest.json').read_text());files=m['files'];assert isinstance(files,dict)and'oracle.cpp'in files and'private_argmax.hpp'in files
 pins={}
 for n,v in files.items():
  f=prep/n;assert f.resolve().is_relative_to(prep)and f.stat().st_size==v['bytes']and sha(f)==v['sha256'],n;pins[n]=v
 assert sha(exe)==a.sha256 and exe.is_file()
 assert sha(a.audit)==a.audit_sha256
 receipt=json.loads(a.audit.read_text());assert receipt['GPU_GO']is True
 bound={receipt[k]for k in ['binary_sha256','new_binary','main_sha256']if k in receipt};assert bound=={a.sha256},'Root audit must explicitly bind this executable'
 assert json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]['Id']==IMAGE
 s=exe.stat();return {'manifest_sha256':a.manifest_sha256,'runner_pins_sha256':sha(P/'harness-pins.json'),'source_pins':pins,'exe':str(exe),'binary_sha256':a.sha256,'binary_stat':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns],'root_audit':str(a.audit),'root_audit_sha256':a.audit_sha256,'image':IMAGE,'requested_physical_card':0,'requested_PCI':'0000:0b:00.0','device_pin':'ZE_AFFINITY_MASK=0;ONEAPI_DEVICE_SELECTOR=level_zero:*','scope':'Requested physical/card pin grounded in same-host health mapping; actual leaf IntelB70 device name checked by parser, no extra GPU identity process.'}
def checked_identity(prefix):
 try:
  value=identity();save(prefix+'.json',value);(O/(prefix+'.exit')).write_text('0\n');return value
 except Exception as e:
  save(prefix+'-error.json',{'type':type(e).__name__,'detail':str(e)});(O/(prefix+'.exit')).write_text('2\n');return None
# No GPU, Docker launch or model workload occurs during source preparation.
save('runner-argv.json',sys.argv)
before=checked_identity('identity')
if before is None:raise SystemExit(2)
name='strata-greedy-top1-s112-v2';assert not(O/'attempt.cid').exists()
existing=subprocess.check_output(['docker','ps','-aq','--filter','name=^'+name+'$']).strip();assert not existing,'Preserve existing named container'
for sub in ['tmp','cache']:(O/sub).mkdir()
if a.reuse_s107_prehealth:
 assert os.environ.get('B70_S112_PREHEALTH_REVALIDATED')=='1','Root must explicitly revalidate no intervening GPU/stack work; never autofill'
 pins=json.loads((P/'prehealth-pins.json').read_text())
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==pins['boot_id']
 for rec in pins['files']:
  src=Path(rec['path']);assert sha(src)==rec['sha256'];(O/('reused-'+src.name)).write_bytes(src.read_bytes())
 health=json.loads((R/'q6-a4-main-s107-prep/mode1/post-health.json').read_text());assert all(v['exit']==0 for v in health.values())
 summary=json.loads((R/'q6-a4-main-s107-prep/root-result-summary.json').read_text());assert summary['status']=='PASS_MATCHED_DISCRIMINATOR'and all(summary['checks'].values())
 stop=json.loads((R/'q6-a4-main-s107-prep/mode1/normal-stop.json').read_text());assert stop['lease_wrapper_exit']==0 and stop['container_removed']
 removal=json.loads((R/'q6-a4-main-s107-prep/mode1/container-removal.json').read_text());assert removal['full_cid_absent']and removal['name_absent']
 save('prehealth-reuse.json',{'root_explicit_revalidation_env':True,'source':'S107 mode1 full card+compiled posthealth','pins':pins,'scope':'Same-host/current stack/no-intervening coordinated GPU work must be revalidated by root immediately before this run; copied health is historical, not a fresh invocation.'});pre=0
else:pre=command('pre-health',[str(REPO/'bin/xpu-health'),'--card','0','--img',HEALTH,'--timeout','400'])
(O/'pre-health.exit').write_text(str(pre)+'\n');assert pre==0
cmd=['docker','create','--name',name,'--cidfile',str(O/'attempt.cid'),'--network','none','--pull','never','--device','/dev/dri','--group-add',str(grp.getgrnam('render').gr_gid),'-u','1000:'+str(os.getgid()),'-e','ZE_AFFINITY_MASK=0','-e','ONEAPI_DEVICE_SELECTOR=level_zero:*','-e','SYCL_CACHE_PERSISTENT=0','-e','SYCL_CACHE_DIR=/out/cache','-e','XDG_CACHE_HOME=/out/cache','-e','TMPDIR=/out/tmp','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-v',str(prep)+':/src:ro','-v',str(exe.parent)+':/probe:ro','-v',str(O)+':/out','--entrypoint','bash',IMAGE,'-c','source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -euo pipefail; exec /probe/greedy_top1_s112 --timing']
create=command('create',cmd);assert create==0
cid=(O/'attempt.cid').read_text().strip();assert re.fullmatch('[0-9a-f]{64}',cid)
def inspect_owned():
 r=subprocess.run(['docker','inspect',cid],capture_output=True,text=True);save('latest-inspect-command.json',{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 if r.returncode:return None
 try:
  ci=json.loads(r.stdout)[0];assert ci['Id']==cid and ci['Name']=='/'+name and ci['Image']==IMAGE and ci['Config']['User']=='1000:'+str(os.getgid());return ci
 except (ValueError,KeyError,IndexError,AssertionError)as e:
  save('inspect-identity-error.json',{'type':type(e).__name__,'detail':str(e)});return None
ci=inspect_owned();assert ci and str(exe.parent)+':/probe:ro'in ci['HostConfig']['Binds'];save('container-created.json',ci)
# Non--rm preserves authoritative final OOM/ExitCode for stopped owned container.
save('docker-start-command.json',['docker','start','-a',cid])
with(O/'native.log').open('w')as out,(O/'native.stderr').open('w')as err:
 start=time.monotonic();r=subprocess.run(['docker','start','-a',cid],stdout=out,stderr=err)
start_rc=r.returncode;(O/'docker-start.exit').write_text(str(start_rc)+'\n');save('leaf-wall.json',{'elapsed_s':time.monotonic()-start})
# Unexpected hang stays in docker start under the SAME leases for root diagnosis.
# If client ended while owned workload remains live/unknown, never overlap health.
while True:
 ci=inspect_owned()
 if ci is not None and not ci['State']['Running']:break
 print('S112 HOLD: owned container live or state unknown; leases retained; root diagnosis, no force stop/reset/health',flush=True);time.sleep(5)
save('container-final.json',ci);native=int(ci['State']['ExitCode']);(O/'native.exit').write_text(str(native)+'\n');oom=bool(ci['State']['OOMKilled'])
rm=command('normal-rm',['docker','rm',cid]);assert rm==0
while True:
 r=subprocess.run(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}}'],capture_output=True,text=True)
 save('container-enumeration.json',{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});rows=[x.split()for x in r.stdout.splitlines()]
 absent=r.returncode==0 and all(len(x)==2 and x[0]!=cid and x[1]!=name for x in rows)
 if absent:break
 print('S112 HOLD: CID/name absence unconfirmed; leases retained, root diagnosis; no health',flush=True);time.sleep(5)
save('container-removal.json',{'full_cid':cid,'name':name,'full_cid_absent':True,'name_absent':True,'docker_start_terminal':True,'normal_rm_exit':rm,'final_native_exit':native,'final_OOMKilled':oom,'no_force_stop':True})
post=command('post-health',[str(REPO/'bin/xpu-health'),'--card','0','--img',HEALTH,'--timeout','400'])
analysis=2
if start_rc==native==0 and not oom:analysis=command('analysis',[sys.executable,str(P/'analyze.py'),str(O)])
(O/'analysis.exit').write_text(str(analysis)+'\n')
after=checked_identity('post-identity');same=before==after
if not same:(O/'post-identity.exit').write_text('2\n')
passed=start_rc==native==pre==post==analysis==0 and not oom and same
save('closure.json',{'status':'PASS_PRIVATE_STRICT_DIAGNOSTIC_ONLY'if passed else'FAIL_PRESERVE_NO_RETRY','docker_start_exit':start_rc,'native_exit':native,'pre_health_exit':pre,'post_health_exit':post,'analysis_exit':analysis,'identity_equal':same,'final_OOMKilled':oom,'full_cid_absent':True,'name_absent':True,'normal_rm_exit':rm,'processes':1,'retries':0,'force_stop_or_reset':False,'speed_or_serving_qualified':False})
print(json.dumps({'status':'PASS'if passed else'FAIL','post_health':post,'analysis':analysis}),flush=True);raise SystemExit(0 if passed else 2)
