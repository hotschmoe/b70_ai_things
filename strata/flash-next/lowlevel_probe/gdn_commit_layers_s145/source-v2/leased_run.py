"""Root reviewed single finite card0 process; no force/reset/retry."""
from pathlib import Path
import grp,hashlib,json,os,re,subprocess,sys,time
P=Path(__file__).resolve().parent;REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
IMAGE='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
HEALTH='b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78'
NAME='strata-gdn-layers-s145-private';EXE=P/'build/gdn_layers_s145'
assert len(sys.argv)==3,'NEW_OUTPUT_DIR EXPECTED_EXE_SHA256';O=Path(sys.argv[1]).resolve();expected=sys.argv[2]
assert re.fullmatch('[0-9a-f]{64}',expected)and not O.exists();O.mkdir()
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n',encoding='ascii')
def cmd(label,args):
 save(label+'-command.json',args)
 with(O/(label+'.stdout')).open('w')as out,(O/(label+'.stderr')).open('w')as err:r=subprocess.run(args,cwd=REPO,stdout=out,stderr=err)
 (O/(label+'.exit')).write_text(str(r.returncode)+'\n');return r.returncode
assert os.getuid()==1000 and os.environ.get('ZE_AFFINITY_MASK')=='0'and os.environ.get('ONEAPI_DEVICE_SELECTOR')=='level_zero:*'
assert sha(EXE)==expected and(P/'build/build.exit').read_text().strip()=='0'and(P/'build/tee.exit').read_text().strip()=='0'
manifest=json.loads((P/'manifest.json').read_text())
for n,row in manifest['files'].items():assert(P/n).stat().st_size==row['bytes']and sha(P/n)==row['sha256'],n
assert json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]['Id']==IMAGE
assert not subprocess.check_output(['docker','ps','-aq','--filter','name=^'+NAME+'$']).strip(),'preserve existing container'
save('identity-before.json',{'exe':str(EXE),'sha256':expected,'image':IMAGE,'manifest_sha256':sha(P/'manifest.json'),'PCI':'0000:0b:00.0','card':0,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
assert cmd('pre-health',[str(REPO/'bin/xpu-health'),'--card','0','--img',HEALTH,'--timeout','400'])==0
for n in['tmp','cache']:(O/n).mkdir()
entry='source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -euo pipefail; ldd /probe/gdn_layers_s145 > /out/runtime-ldd.txt; exec /probe/gdn_layers_s145 /out/leaf 0000:0b:00.0 --timing'
create=['docker','create','--name',NAME,'--cidfile',str(O/'attempt.cid'),'--network','none','--pull','never','--workdir','/out','--device','/dev/dri','--group-add',str(grp.getgrnam('render').gr_gid),'-u','1000:'+str(os.getgid()),'-e','ZE_AFFINITY_MASK=0','-e','ONEAPI_DEVICE_SELECTOR=level_zero:*','-e','STRATA_GDN_SPLIT=0','-e','STRATA_GDN_COMMIT_SPLIT=1','-e','STRATA_DBG_GDN=0','-e','SYCL_CACHE_PERSISTENT=0','-e','SYCL_CACHE_DIR=/out/cache','-e','XDG_CACHE_HOME=/out/cache','-e','TMPDIR=/out/tmp','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-v',str(EXE.parent)+':/probe:ro','-v',str(O)+':/out','--entrypoint','bash',IMAGE,'-c',entry]
def hold_root(reason,detail=None):
 save('lifecycle-HOLD.json',{'status':'HOLD_SAME_LEASE_ROOT_REQUIRED','reason':reason,'detail':detail,'no_force_reset_retry_or_health':True})
 print('HOLD '+reason+'; SAME CPU/GPU leases retained; root intervention required',flush=True)
 while True:time.sleep(5)
# From the first create attempt until confirmed full CID/NAME absence, every
# uncertain lifecycle error holds THIS process/lease instead of escaping.
try:
 create_rc=cmd('create',create)
 if create_rc!=0:hold_root('create status nonzero',{'exit':create_rc})
 if not(O/'attempt.cid').is_file():hold_root('create missing CID file')
 cid=(O/'attempt.cid').read_text().strip()
 if not re.fullmatch('[0-9a-f]{64}',cid):hold_root('create invalid CID')
 def inspect():
  r=subprocess.run(['docker','inspect',cid],capture_output=True,text=True);save('latest-inspect.json',{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
  if r.returncode:hold_root('owned container inspect failed',{'exit':r.returncode})
  ci=json.loads(r.stdout)[0]
  if not(ci['Id']==cid and ci['Name']=='/'+NAME and ci['Image']==IMAGE):hold_root('owned container inspect identity mismatch')
  return ci
 ci=inspect()
 if ci['Config']['User']!='1000:'+str(os.getgid()):hold_root('created container user mismatch')
 if str(EXE.parent)+':/probe:ro'not in ci['HostConfig']['Binds']:hold_root('created executable mount mismatch')
 save('created.json',ci)
 t=time.monotonic();start=cmd('leaf',['docker','start','-a',cid]);save('elapsed.json',{'seconds':time.monotonic()-t})
 ci=inspect()
 if ci['State']['Running']:hold_root('owned container still running after start terminal')
 save('final-state.json',ci);native=ci['State']['ExitCode'];normal=start==native==0 and not ci['State']['OOMKilled']and not ci['State'].get('Dead')and not ci['State'].get('Error')
 wait=cmd('normal-wait',['docker','wait',cid])
 if wait!=0 or(O/'normal-wait.stdout').read_text().strip()!=str(native):hold_root('normal wait unconfirmed',{'exit':wait,'expected_native_exit':native})
 rm=cmd('normal-rm',['docker','rm',cid])
 if rm!=0:hold_root('normal remove unconfirmed',{'exit':rm})
 r=subprocess.run(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}}'],capture_output=True,text=True);save('enumeration.json',{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 if r.returncode!=0:hold_root('post-remove enumeration failed',{'exit':r.returncode})
 rows=[x.split()for x in r.stdout.splitlines()]
 if not all(len(x)==2 and x[0]!=cid and x[1]!=NAME for x in rows):hold_root('full CID/NAME absence unconfirmed')
 save('removal.json',{'CID':cid,'name':NAME,'CID_absent':True,'name_absent':True,'normal_wait':True,'normal_rm':True,'native_exit':native,'OOMKilled':ci['State']['OOMKilled'],'no_force_reset_retry':True})
except Exception as error:
 hold_root('lifecycle exception',{'type':type(error).__name__,'message':str(error)})
if not normal:
 save('closure.json',{'status':'HOLD_ROOT_REVIEW_NO_AUTOMATIC_HEALTH_OR_RETRY','native_exit':native});raise SystemExit(2)
assert cmd('analysis',[sys.executable,str(P/'analyze.py'),str(O)])==0
post=cmd('post-health',[str(REPO/'bin/xpu-health'),'--card','0','--img',HEALTH,'--timeout','400'])
assert sha(EXE)==expected
save('closure.json',{'status':'PASS_PRIVATE_STRICT_AND_NORMAL_HEALTH'if post==0 else'HOLD_POST_HEALTH','pre_health_exit':0,'post_health_exit':post,'native_exit':native,'identity_equal':True,'CID_absent':True,'name_absent':True,'normal_wait_rm':True,'analysis':'comparison.json','retries':0,'serving_or_speed_qualified':False})
raise SystemExit(0 if post==0 else 2)
