"""ROOT-only finite C6; quiet CPU/card0 lease inherited; no forced recovery."""
from pathlib import Path
import argparse,grp,hashlib,json,os,re,subprocess,sys,time
P=Path(__file__).resolve().parent;R=P.parent;REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
IMAGE='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2';HEALTH='b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78'
PCI='0000:0b:00.0';NAME='strata-q5-head-c6-runtime-ones-private'
p=argparse.ArgumentParser()
for n in['out','source','source-manifest','root-go']:p.add_argument('--'+n,type=Path,required=True)
for n in['exe-sha256','root-go-sha256','manifest-sha256']:p.add_argument('--'+n,required=True)
a=p.parse_args();O=a.out.resolve();S=a.source.resolve();EXE=S/'build/q5_runtime_ones_s126';EXESHA=a.exe_sha256
SOURCE_MANIFEST_SHA='5960e16c650b5efa8c08e2971ea51febf8df9dfaae1515ed2088b3ae1e585910'
assert S.parent==R and S.name=='q5-head-c6-runtime-ones-private-prep','explicit reviewed C6 source only'
def sha(f):
 with f.open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2)+'\n',encoding='ascii')
def command(label,cmd):
 save(label+'-command.json',cmd)
 with(O/(label+'.stdout')).open('w')as out,(O/(label+'.stderr')).open('w')as err:result=subprocess.run(cmd,cwd=REPO,stdout=out,stderr=err)
 (O/(label+'.exit')).write_text(str(result.returncode)+'\n');return result.returncode
fix=json.loads((S/'fixture-pins.json').read_text());SHARD=Path(fix['preferred_runtime_NVMe_shard']['path']);FIXTURES=Path(fix['fixtures'][0]['path']).parent
assert all(Path(x['path']).parent==FIXTURES for x in fix['fixtures'])
def stat(f):
 s=f.stat();return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def identity(check_slice):
 for n,h in json.loads((P/'harness-pins.json').read_text()).items():assert sha(P/n)==h,n
 assert os.getuid()==1000 and os.environ.get('ZE_AFFINITY_MASK')=='0'and os.environ.get('ONEAPI_DEVICE_SELECTOR')=='level_zero:*'
 assert a.manifest_sha256==SOURCE_MANIFEST_SHA and sha(a.source_manifest)==a.manifest_sha256
 manifest=json.loads(a.source_manifest.read_text());assert isinstance(manifest['files'],dict)
 for name,row in manifest['files'].items():
  f=S/name;assert f.resolve().is_relative_to(S)and f.stat().st_size==row['bytes']and sha(f)==row['sha256']
 assert EXE.is_file()and sha(EXE)==EXESHA
 assert (S/'build/build.exit').read_text().strip()=='0'and (S/'build/tee.exit').read_text().strip()=='0'and json.loads((S/'build/image.json').read_text())[0]['Id']==IMAGE
 assert sha(a.root_go)==a.root_go_sha256;receipt=json.loads(a.root_go.read_text())
 assert receipt['RUN_GO']is True and receipt['scope']=='Q5_HEAD_C6_RUNTIME_ONES_PRIVATE' and receipt['binary_sha256']==EXESHA
 assert receipt['source_manifest_sha256']==SOURCE_MANIFEST_SHA and receipt['source']==str(S)
 review=R/'q5-head-c6-runtime-ones-private-source-review.txt'
 assert receipt['independent_source_review_sha256']==sha(review)=='40acf13bb8c1112cd83e8d1aef7e8a9fb34b1f336a528246a11e3a19a3929484'
 assert receipt['optional_cpu_compiled_certification']=='DEFERRED_BY_USER'
 assert json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]['Id']==IMAGE
 assert sha(Path(fix['model_lock']))==fix['model_lock_sha256']
 assert stat(SHARD)==fix['preferred_runtime_NVMe_shard']['stat'],'NVMe shard stat changed since source binding'
 assert re.fullmatch(r'[A-Za-z0-9_.-]+',SHARD.name)
 if check_slice:
  h=hashlib.sha256();remaining=fix['head_slice']['bytes']
  with SHARD.open('rb')as f:
   f.seek(fix['head_slice']['offset'])
   while remaining:
    b=f.read(min(8*1024*1024,remaining));assert b,'short selected slice';h.update(b);remaining-=len(b)
  assert h.hexdigest()==fix['head_slice']['sha256_from_actual_S52'],'selected437MB slice SHA mismatch'
 for row in fix['fixtures']:
  f=Path(row['path']);assert f.stat().st_size==row['bytes']and sha(f)==row['sha256_from_retained_S52_comparison']
 # Read-only source ordering proof for the numeric-failure whitelist, not new math.
 cpp=(S/'q5_runtime_ones_leaf.cpp').read_text();raw=cpp[cpp.index('static void raw_gate'):cpp.index('static void raw_extremes')]
 assert raw.index('for(size_t k=count*4;')<raw.index('for(size_t k=0;k<count;'),'validation tail must precede INT64 gate'
 assert cpp.count('w.readonly(weights.data())')==1 and cpp.count('state->weight_payload_check(')==2
 return {'source':str(S),'source_manifest':str(a.source_manifest),'manifest_sha256':a.manifest_sha256,'binary_sha256':EXESHA,'binary_stat':stat(EXE),'build_sha256_record_sha256':sha(S/'build/sha256.txt'),'compiler_version':(S/'build/compiler-version.txt').read_text(),'root_RUN_GO':str(a.root_go),'root_RUN_GO_sha256':a.root_go_sha256,'optional_cpu_compiled_certification':'DEFERRED_BY_USER','independent_source_review_sha256':sha(review),'image':IMAGE,'requested_card':0,'required_actual_native_PCI':PCI,'selectors':{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:*'},'model_lock_sha256':fix['model_lock_sha256'],'selected_head_slice':fix['head_slice'],'shard_path':str(SHARD),'shard_stat':stat(SHARD),'fixtures':fix['fixtures'],'validation':'selected slice hashed once before; leaf FNV actual load and twice full device payload checks; post same stat; no whole model hash','boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
def checked_identity(label,check_slice):
 try:x=identity(check_slice);save(label+'.json',x);(O/(label+'.exit')).write_text('0\n');return x
 except Exception as e:save(label+'-error.json',{'type':type(e).__name__,'detail':str(e)});(O/(label+'.exit')).write_text('2\n');return None
save('runner-argv.json',sys.argv);before=checked_identity('identity',True)
if before is None:raise SystemExit(2)
assert not(O/'attempt.cid').exists();assert not subprocess.check_output(['docker','ps','-aq','--filter','name=^'+NAME+'$']).strip(),'preserve existing named container'
for n in['tmp','cache']:(O/n).mkdir()
pre=command('pre-health',[str(REPO/'bin/xpu-health'),'--card','0','--img',HEALTH,'--timeout','400'])
assert pre==0,'fresh card0 prehealth failed; no leaf start'
entry=r"""source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
ldd /probe/q5_runtime_ones_s126 > /out/runtime-ldd.txt
printf '%s\n' "${LD_LIBRARY_PATH-}" > /out/runtime-LD_LIBRARY_PATH.txt
python3 - <<'PYLIB'
from pathlib import Path
import hashlib,json,os,re
out=Path('/out');rows=[]
for line in (out/'runtime-ldd.txt').read_text().splitlines():
 m=re.search(r'=> (/\S+)',line)or re.match(r'\s*(/\S+)',line)
 if not m:continue
 p=Path(m[1]).resolve();st=p.stat();row={'ldd_line':line,'resolved_path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
 if any(k in line for k in ['libsycl.so','libstdc++.so','libze_loader.so']):
  with p.open('rb')as stream:row['sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
 rows.append(row)
(out/'prospective-library-identities.json').write_text(json.dumps({'scope':'ldd-resolved prospective dependencies only; not actual leaf proc maps/dlopen proof','libraries':rows},indent=2)+'\n')
keys=['ZE_AFFINITY_MASK','ONEAPI_DEVICE_SELECTOR','SYCL_CACHE_PERSISTENT','SYCL_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','OMP_NUM_THREADS','MKL_NUM_THREADS','LD_LIBRARY_PATH']
(out/'effective-entry-env.json').write_text(json.dumps({'environment':{k:os.environ.get(k)for k in keys},'native_exec_unsets':['LD_DEBUG'],'LD_DEBUG':'ABSENT'},indent=2)+'\n')
PYLIB
exec env -u LD_DEBUG /probe/q5_runtime_ones_s126 /out/leaf /weights/SHARD_NAME /fixtures 01010101
"""
entry=entry.replace('SHARD_NAME',SHARD.name)
cmd=['docker','create','--name',NAME,'--cidfile',str(O/'attempt.cid'),'--network','none','--workdir','/out','--pull','never','--device','/dev/dri','--group-add',str(grp.getgrnam('render').gr_gid),'-u','1000:'+str(os.getgid()),'-e','ZE_AFFINITY_MASK=0','-e','ONEAPI_DEVICE_SELECTOR=level_zero:*','-e','SYCL_CACHE_PERSISTENT=0','-e','SYCL_CACHE_DIR=/out/cache','-e','XDG_CACHE_HOME=/out/cache','-e','TMPDIR=/out/tmp','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','STRATA_MMVQ_WIDE_K=1','-e','STRATA_MMVQ_WIDE_K_RPW=1','-e','STRATA_HEAD_Q5K_RPW2=0','-v',str(S)+':/src:ro','-v',str(EXE.parent)+':/probe:ro','-v',str(SHARD.parent)+':/weights:ro','-v',str(FIXTURES)+':/fixtures:ro','-v',str(O)+':/out','--entrypoint','bash',IMAGE,'-c',entry]
create=command('create',cmd)
if create!=0:save('recovery-required.json',{'reason':'docker create failed; no automatic start/cleanup/health','GPU_health_after_attempt':False});raise SystemExit(2)
cid=(O/'attempt.cid').read_text().strip();assert re.fullmatch('[0-9a-f]{64}',cid)
def inspect_owned():
 r=subprocess.run(['docker','inspect',cid],capture_output=True,text=True);save('latest-inspect-command.json',{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 if r.returncode:return None
 try:
  ci=json.loads(r.stdout)[0];assert ci['Id']==cid and ci['Name']=='/'+NAME and ci['Image']==IMAGE and ci['Config']['User']=='1000:'+str(os.getgid());return ci
 except (ValueError,KeyError,IndexError,AssertionError)as e:save('inspect-identity-error.json',{'type':type(e).__name__,'detail':str(e)});return None
ci=inspect_owned();assert ci and str(EXE.parent)+':/probe:ro'in ci['HostConfig']['Binds'];save('container-created.json',ci)
save('docker-start-command.json',['docker','start','-a',cid])
with(O/'leaf.log').open('w')as out,(O/'leaf.stderr').open('w')as err:
 start=time.monotonic();r=subprocess.run(['docker','start','-a',cid],stdout=out,stderr=err)
start_rc=r.returncode;(O/'docker-start.exit').write_text(str(start_rc)+'\n');save('leaf-wall.json',{'elapsed_s':time.monotonic()-start})
# A hang remains on same docker start/leases for root. Unknown/live state never overlaps health.
while True:
 ci=inspect_owned()
 if ci is not None and not ci['State']['Running']:break
 print('C6 HOLD owned container live/state unknown; same leases retained, no force/reset/health',flush=True);time.sleep(5)
save('container-final.json',ci);native=int(ci['State']['ExitCode']);oom=bool(ci['State']['OOMKilled']);(O/'leaf.exit').write_text(str(native)+'\n')
wait=command('normal-wait',['docker','wait',cid]);waittext=(O/'normal-wait.stdout').read_text().strip();waitcode=int(waittext)if re.fullmatch(r'\d+',waittext)else None
stderr=(O/'leaf.stderr').read_text()
failures=re.findall(r'^S126 STRICT_FAILURE: (.*)$',stderr,re.M)
raw_numeric={'extremes-ones exact INT64 raw16 gate','payload-negative-control exact INT64 raw16 gate','canonical-a exact INT64 raw16 gate','canonical-b exact INT64 raw16 gate'}
# C6 retains V3 validation tail is checked before raw INT64 comparison. Every
# full-output comparison follows snapshot guards/tails/input/Q8 checks.
full_numeric=lambda m:m=='STRICT saved S52 RPW1 public byte anchor'or bool(re.fullmatch(r'STRICT head (?:eager-(?:original|candidate)-[ab]|repeat-(?:original|candidate)-[ab]|restore-eager-(?:public|original|candidate)|graph-(?:public|original|candidate)-[ab]|graph-repeat-(?:public|original|candidate)-[ab]|batch20-(?:original|candidate)-[ab]|restore-graph-(?:public|original|candidate)|restore-batch20-(?:original|candidate)|warm-[0-2]-(?:original|candidate)|timed-r[01]-s[0-3]-(?:original|candidate)|final) full output bytes',m))
cleanup_bad=any(t.lower()in stderr.lower()for t in['HOLD','drain failed','capture balance','cleanup failed'])
numeric_failure=native==5 and len(failures)==1 and(failures[0]in raw_numeric or full_numeric(failures[0]))and not cleanup_bad
normal_state=start_rc==native and wait==0 and waitcode==native and not oom and not ci['State'].get('Dead',False)and not ci['State'].get('Error','')
rm=command('normal-rm',['docker','rm',cid]);assert rm==0
while True:
 r=subprocess.run(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}}'],capture_output=True,text=True);save('container-enumeration.json',{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});rows=[x.split()for x in r.stdout.splitlines()]
 if r.returncode==0 and all(len(x)==2 and x[0]!=cid and x[1]!=NAME for x in rows):break
 print('C6 HOLD CID/NAME absence unconfirmed; same leases retained, no health',flush=True);time.sleep(5)
removal={'full_cid':cid,'name':NAME,'full_cid_absent':True,'name_absent':True,'docker_start_terminal':True,'normal_wait_exit':wait,'wait_native_exit':waitcode,'normal_rm_exit':rm,'final_native_exit':native,'final_OOMKilled':oom,'no_force_stop':True};save('container-removal.json',removal)
if not normal_state or not(native==0 or numeric_failure):
 save('recovery-required.json',{'status':'ROOT_RECOVERY_REQUIRED_NO_BLIND_GPU_HEALTH','normal_state':normal_state,'native_exit':native,'numeric_failure_whitelisted':numeric_failure,'strict_failure_messages':failures,'removal':removal,'automatic_health_or_reset':False});raise SystemExit(2)
# Capture/validate all completed native/DOT diagnostics before any posthealth.
analysis=2
if native==0:
 analysis=command('analysis',[sys.executable,str(P/'analyze.py'),str(O)])
 if analysis!=0:
  save('recovery-required.json',{'status':'ROOT_DIAGNOSTIC_REVIEW_REQUIRED_NO_BLIND_HEALTH','analysis_exit':analysis,'normal_removed_state':True,'automatic_health_or_reset':False,'timing_summary':None})
  raise SystemExit(2)
# Single-card completed, whitelisted numerical mismatch may receive normal card0 posthealth.
post=command('post-health',[str(REPO/'bin/xpu-health'),'--card','0','--img',HEALTH,'--timeout','400'])
if post!=0:save('recovery-required.json',{'status':'ROOT_HEALTH_RECOVERY_REQUIRED','post_health_exit':post,'automatic_retry_or_reset':False})
(O/'analysis.exit').write_text(str(analysis)+'\n');after=checked_identity('post-identity',False);same=before==after
if not same:(O/'post-identity.exit').write_text('2\n')
passed=start_rc==native==pre==post==analysis==0 and not oom and same
save('closure.json',{'status':'PASS_PRIVATE_C6_STRICT_TIMING_DIAGNOSTIC_ONLY'if passed else'FAIL_PRESERVE_NULL_TIMINGS_NO_RETRY','docker_start_exit':start_rc,'native_exit':native,'normal_wait_exit':wait,'pre_health_exit':pre,'post_health_exit':post,'analysis_exit':analysis,'identity_equal':same,'final_OOMKilled':oom,'full_cid_absent':True,'name_absent':True,'normal_rm_exit':rm,'normal_numeric_failure':numeric_failure,'processes':1,'retries':0,'force_stop_or_reset':False,'timing_summary':'comparison.json'if passed else None,'speed_or_serving_qualified':False})
print(json.dumps({'status':'PASS'if passed else'FAIL','post_health':post,'analysis':analysis}),flush=True);raise SystemExit(0 if passed else 2)
