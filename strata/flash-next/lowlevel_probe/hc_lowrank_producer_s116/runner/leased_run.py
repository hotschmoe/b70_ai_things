"""ROOT-only finite S116; quietCPU/card1 lease inherited; no forced recovery."""
from pathlib import Path
import argparse,grp,hashlib,json,os,re,subprocess,sys,time
P=Path(__file__).resolve().parent;R=P.parent;REPO=Path('/mnt/vm_8tb/github/b70_ai_things');S=R/'hc-producer-s116-oracle-prep';EXE=S/'build/hc_producer_s116'
IMAGE='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2';HEALTH='b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78'
EXESHA='33e390536480f83f169a63e1f9d2f2e06d8e341279a956c84a29a255951c46d3';PCI='0000:44:00.0';NAME='strata-hc-producer-s116-private'
p=argparse.ArgumentParser()
for n in['out','audit','reuse-prehealth-proof']:p.add_argument('--'+n,type=Path,required=n!='reuse-prehealth-proof')
for n in['audit-sha256','manifest-sha256']:p.add_argument('--'+n,required=True)
p.add_argument('--reuse-prehealth-sha256');a=p.parse_args();O=a.out.resolve()
def sha(f):
 with f.open('rb')as s:return hashlib.file_digest(s,'sha256').hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2)+'\n',encoding='ascii')
def command(label,cmd):
 save(label+'-command.json',cmd)
 with(O/(label+'.stdout')).open('w')as out,(O/(label+'.stderr')).open('w')as err:r=subprocess.run(cmd,cwd=REPO,stdout=out,stderr=err)
 (O/(label+'.exit')).write_text(str(r.returncode)+'\n');return r.returncode
def identity(check_anchor_hashes):
 for n,h in json.loads((P/'harness-pins.json').read_text()).items():assert sha(P/n)==h,n
 assert os.getuid()==1000 and os.environ.get('ZE_AFFINITY_MASK')=='1'and os.environ.get('ONEAPI_DEVICE_SELECTOR')=='level_zero:*'
 assert sha(S/'MANIFEST.json')==a.manifest_sha256
 manifest=json.loads((S/'MANIFEST.json').read_text());assert isinstance(manifest['files'],list)and len(manifest['files'])==14
 for row in manifest['files']:
  f=S/row['path'];assert f.resolve().is_relative_to(S)and f.stat().st_size==row['bytes']and sha(f)==row['sha256']
 assert EXE.stat().st_size==822704 and sha(EXE)==EXESHA
 assert (S/'build/build.exit').read_text().strip()=='0'and json.loads((S/'build/image.json').read_text())[0]['Id']==IMAGE
 assert sha(a.audit)==a.audit_sha256
 receipt=json.loads(a.audit.read_text());assert receipt['GPU_GO']is True
 bound={receipt[k]for k in['binary_sha256','exe_sha256','main_sha256']if k in receipt};assert bound=={EXESHA},'Root GPU GO must bind exact executable'
 assert json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]['Id']==IMAGE
 spec=json.loads((S/'s39-anchors.json').read_text());assert spec['count']==len(spec['anchors'])==36 and spec['snapshot_bytes']==747776;anchors={}
 for n,h in spec['anchors'].items():
  f=Path(spec['anchor_directory'])/n;st=f.stat();assert st.st_size==747776
  if check_anchor_hashes:assert sha(f)==h,n
  anchors[n]={'expected_sha256':h,'stat':[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]}
 st=EXE.stat()
 return {'manifest_sha256':a.manifest_sha256,'binary_sha256':EXESHA,'binary_stat':[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns],'build_compiler_version':(S/'build/compiler-version.txt').read_text(),'root_GPU_GO':str(a.audit),'root_GPU_GO_sha256':a.audit_sha256,'image':IMAGE,'requested_card':1,'required_actual_native_PCI':PCI,'selectors':{'ZE_AFFINITY_MASK':'1','ONEAPI_DEVICE_SELECTOR':'level_zero:*'},'anchors':anchors,'anchor_policy':'Hash once before GPU; exact source/native anchor-byte checks before timing; post host stat continuity only, no repeated27MB hash.'}
def checked_identity(label,check_anchor_hashes):
 try:x=identity(check_anchor_hashes);save(label+'.json',x);(O/(label+'.exit')).write_text('0\n');return x
 except Exception as e:save(label+'-error.json',{'type':type(e).__name__,'detail':str(e)});(O/(label+'.exit')).write_text('2\n');return None
save('runner-argv.json',sys.argv);before=checked_identity('identity',True)
if before is None:raise SystemExit(2)
assert not(O/'attempt.cid').exists();assert not subprocess.check_output(['docker','ps','-aq','--filter','name=^'+NAME+'$']).strip(),'preserve existing named container'
for n in['tmp','cache']:(O/n).mkdir()
if a.reuse_prehealth_proof:
 assert os.environ.get('B70_S116_PREHEALTH_REVALIDATED')=='1','Root must revalidate same boot/stack/no intervening GPU work immediately before run'
 assert a.reuse_prehealth_sha256 and sha(a.reuse_prehealth_proof)==a.reuse_prehealth_sha256
 proof=json.loads(a.reuse_prehealth_proof.read_text());assert proof['status']=='ROOT_REUSE_S114_FULL_POSTHEALTH_FOR_S116_CARD1'
 assert proof['no_intervening_GPU_or_stack_change']is True and proof['source_all_exit0']is True and proof['required_probe_card']==1
 assert proof['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert len(proof['files'])>=3
 for i,row in enumerate(proof['files']):
  f=Path(row['path']);assert f.resolve().is_relative_to(R)and sha(f)==row['sha256'];(O/(str(i)+'-reused-'+f.name)).write_bytes(f.read_bytes())
 health=json.loads(Path(proof['post_health_path']).read_text());assert set(health)=={'card','collective'}and all(v['exit']==0 for v in health.values())
 assert any(Path(v['path'])==Path(proof['post_health_path'])for v in proof['files'])
 save('prehealth-reuse.json',{'proof':proof,'proof_sha256':a.reuse_prehealth_sha256,'root_env_revalidation':True,'scope':'Historical full S114 card+compiled health; root current no-intervening-work assertion, not fresh health.'});pre=0
else:
 assert a.reuse_prehealth_sha256 is None
 pre=command('pre-health',[str(REPO/'bin/xpu-health'),'--card','1','--img',HEALTH,'--timeout','400'])
(O/'pre-health.exit').write_text(str(pre)+'\n');assert pre==0
anchor_dir=json.loads((S/'s39-anchors.json').read_text())['anchor_directory']
# ldd/loader diagnostics are not an extra GPU identity process; actual PCI checked in leaf.
entry=r"""source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
ldd /probe/hc_producer_s116 > /out/runtime-ldd.txt
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
(out/'effective-entry-env.json').write_text(json.dumps({'environment':{k:os.environ.get(k)for k in keys},'native_exec_unsets':['STRATA_HC_Q8','STRATA_QFUSE'],'LD_DEBUG':'ABSENT'},indent=2)+'\n')
PYLIB
exec env -u STRATA_HC_Q8 -u STRATA_QFUSE /probe/hc_producer_s116 /out/oracle --pci 0000:44:00.0 --timing
"""
cmd=['docker','create','--name',NAME,'--cidfile',str(O/'attempt.cid'),'--network','none','--workdir','/out','--pull','never','--device','/dev/dri','--group-add',str(grp.getgrnam('render').gr_gid),'-u','1000:'+str(os.getgid()),'-e','ZE_AFFINITY_MASK=1','-e','ONEAPI_DEVICE_SELECTOR=level_zero:*','-e','SYCL_CACHE_PERSISTENT=0','-e','SYCL_CACHE_DIR=/out/cache','-e','XDG_CACHE_HOME=/out/cache','-e','TMPDIR=/out/tmp','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-v',str(S)+':/src:ro','-v',str(EXE.parent)+':/probe:ro','-v',anchor_dir+':'+anchor_dir+':ro','-v',str(O)+':/out','--entrypoint','bash',IMAGE,'-c',entry]
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
with(O/'native.log').open('w')as out,(O/'native.stderr').open('w')as err:
 start=time.monotonic();r=subprocess.run(['docker','start','-a',cid],stdout=out,stderr=err)
start_rc=r.returncode;(O/'docker-start.exit').write_text(str(start_rc)+'\n');save('leaf-wall.json',{'elapsed_s':time.monotonic()-start})
# A hang remains on same docker start/leases for root. Unknown/live state never overlaps health.
while True:
 ci=inspect_owned()
 if ci is not None and not ci['State']['Running']:break
 print('S116 HOLD owned container live/state unknown; same leases retained, no force/reset/health',flush=True);time.sleep(5)
save('container-final.json',ci);native=int(ci['State']['ExitCode']);oom=bool(ci['State']['OOMKilled']);(O/'native.exit').write_text(str(native)+'\n')
wait=command('normal-wait',['docker','wait',cid]);waittext=(O/'normal-wait.stdout').read_text().strip();waitcode=int(waittext)if re.fullmatch(r'\d+',waittext)else None
stderr=(O/'native.stderr').read_text();holds=re.findall(r'^S116 HOLD: (.*); no retry/reset/tolerance$',stderr,re.M)
# Source throws all std::exceptions as5, so never equate exit5 with safe numeric failure.
numeric={'R bytes','lo bytes','rs bytes','inject bytes','mixed bytes','xn bytes','raw part bytes','raw part2 bytes','prefix R','prefix rs','prefix xn','prefix inject','prefix raw part','prefix raw part2','prefix mixed poison','candidate pre-UP lo vs control post-UP lo','historical S39 anchor bytes','finite R','finite rs','finite xn','finite part','finite inject','finite lo','finite mixed','finite part2'}
cleanup_bad=any(t in stderr for t in['HOLD capture balance failed','HOLD capture drain failed','HOLD final drain failed'])
numeric_failure=native==5 and len(holds)==1 and holds[0]in numeric and not cleanup_bad
normal_state=start_rc==native and wait==0 and waitcode==native and not oom and not ci['State'].get('Dead',False)and not ci['State'].get('Error','')
rm=command('normal-rm',['docker','rm',cid]);assert rm==0
while True:
 r=subprocess.run(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}}'],capture_output=True,text=True);save('container-enumeration.json',{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});rows=[x.split()for x in r.stdout.splitlines()]
 if r.returncode==0 and all(len(x)==2 and x[0]!=cid and x[1]!=NAME for x in rows):break
 print('S116 HOLD CID/NAME absence unconfirmed; same leases retained, no health',flush=True);time.sleep(5)
removal={'full_cid':cid,'name':NAME,'full_cid_absent':True,'name_absent':True,'docker_start_terminal':True,'normal_wait_exit':wait,'wait_native_exit':waitcode,'normal_rm_exit':rm,'final_native_exit':native,'final_OOMKilled':oom,'no_force_stop':True};save('container-removal.json',removal)
if not normal_state or not(native==0 or numeric_failure):
 save('recovery-required.json',{'status':'ROOT_RECOVERY_REQUIRED_NO_BLIND_GPU_HEALTH','normal_state':normal_state,'native_exit':native,'numeric_failure_whitelisted':numeric_failure,'holds':holds,'removal':removal,'automatic_health_or_reset':False});raise SystemExit(2)
# Single-card completed numeric comparison failure may still receive normal card1 posthealth.
post=command('post-health',[str(REPO/'bin/xpu-health'),'--card','1','--img',HEALTH,'--timeout','400'])
if post!=0:save('recovery-required.json',{'status':'ROOT_HEALTH_RECOVERY_REQUIRED','post_health_exit':post,'automatic_retry_or_reset':False})
analysis=2
if native==0 and post==0:analysis=command('analysis',[sys.executable,str(P/'analyze.py'),str(O)])
(O/'analysis.exit').write_text(str(analysis)+'\n');after=checked_identity('post-identity',False);same=before==after
if not same:(O/'post-identity.exit').write_text('2\n')
passed=start_rc==native==pre==post==analysis==0 and not oom and same
save('closure.json',{'status':'PASS_PRIVATE_S116_STRICT_TIMING_DIAGNOSTIC_ONLY'if passed else'FAIL_PRESERVE_NULL_TIMINGS_NO_RETRY','docker_start_exit':start_rc,'native_exit':native,'normal_wait_exit':wait,'pre_health_exit':pre,'post_health_exit':post,'analysis_exit':analysis,'identity_equal':same,'final_OOMKilled':oom,'full_cid_absent':True,'name_absent':True,'normal_rm_exit':rm,'normal_numeric_failure':numeric_failure,'processes':1,'retries':0,'force_stop_or_reset':False,'timing_summary':'analysis.json'if passed else None,'speed_or_serving_qualified':False})
print(json.dumps({'status':'PASS'if passed else'FAIL','post_health':post,'analysis':analysis}),flush=True);raise SystemExit(0 if passed else 2)
