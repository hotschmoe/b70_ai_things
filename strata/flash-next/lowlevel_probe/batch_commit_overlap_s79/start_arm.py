#!/usr/bin/env python3
"""Root executes under cpu-run for startup only. systemd unit retains pair lease."""
import argparse,hashlib,json,re,subprocess,time,urllib.request
from pathlib import Path
D=Path(__file__).resolve().parents[1];REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
p=argparse.ArgumentParser();p.add_argument('arm',choices=['0','1']);a=p.parse_args();out=D/f'mode{a.arm}'
assert not (out/'start-result.json').exists(),'Do not restart a prior arm'
inputs=json.loads((D/'serving-inputs.json').read_text());config=out/'config.json';c=json.loads(config.read_text())
arm=inputs['arms'][a.arm]
assert c['exe']==arm['exe']
with Path(c['exe']).open('rb') as binary:assert hashlib.file_digest(binary,'sha256').hexdigest()==arm['sha256']
assert all(k not in c['env'] for k in ['STRATA_HOST_BOUNDARY','STRATA_VERIFY_PROFILE','STRATA_PREFILL_HOST_BOUNDARY','STRATA_PREFILL_TIMING'])
assert c['env']['STRATA_TRACE']=='1' and c['env']['STRATA_BATCH_MTP_PLE_PREFETCH']=='0'
assert c['env']['STRATA_BATCH_COMMIT_OVERLAP']==a.arm and c['env']['STRATA_HEAD_Q5K_RPW2']=='0' and c['env']['STRATA_MMVQ_WIDE_K_RPW']=='1' and c['env']['STRATA_MMVQ_WIDE_K']=='1'
assert inputs['same_binary'] and inputs['arms']['0']==inputs['arms']['1'] and inputs['HC_state']=='COMMON_LO_ONCE_S51'
assert c['args'][c['args'].index('--native')+1]==inputs['native']['path']
assert c['args'][c['args'].index('--ple-gguf')+1]==inputs['native']['sibling_path']
assert c['args'][c['args'].index('--ple-io')+1]=='direct','S79 requires explicit Direct PLE'
container=inputs['containers'][a.arm]
assert not subprocess.check_output(['docker','ps','-aq','--filter',f'name=^{container}$']).strip(),'Existing S79 container must be handled by owner'
try:
 with urllib.request.urlopen('http://127.0.0.1:18085/health',timeout=2) as f:existing=json.load(f)
except OSError:pass
else:raise SystemExit('Port18085 already answers; preserve existing service, do not launch over it')
unit=f'strata-batch-commit-s79-mode{a.arm}'
# systemd manager creates the service; no cpu-run fd is transferred through DBus.
# Explicitly close 7 again in the detached service before gpu-run acquires 8/9.
cmd=['systemd-run','--user',f'--unit={unit}','--collect',f'--setenv=B70_S79_CONTAINER={container}',f'--property=WorkingDirectory={REPO}',
     f'--property=StandardOutput=append:{out}/server.log',f'--property=StandardError=append:{out}/server.log',
     '/bin/bash','-c','exec 7>&-; exec /bin/bash "$1" "$2"','bash',str(D/'harness-prep/launch_stage.sh'),str(config)]
t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,close_fds=True)
(out/'start-result.json').write_text(json.dumps({'command':cmd,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'unit':unit,'container':container,'epoch_ns':time.time_ns()},indent=2)+'\n',encoding='ascii');assert r.returncode==0,r.stderr
last=None
while time.monotonic()-t<1200:
 try:
  with urllib.request.urlopen('http://127.0.0.1:18085/health',timeout=3) as f:health=json.load(f)
  if health.get('loaded') is True:
   ci=json.loads(subprocess.check_output(['docker','inspect',container]))[0]
   assert str(config) in ci['Config']['Cmd'] and ci['State']['Running'] and not ci['State']['OOMKilled']
   assert health['model']=='hotschmoe-dd' and health['max_context']==200000
   log=(out/'engine.log').read_text();assert arm['exe'] in log.splitlines()[0]
   assert inputs['native']['path'] in log.splitlines()[0] and inputs['native']['sibling_path'] in log.splitlines()[0]
   assert re.search(r'--ple-io direct(?: |$)',log.splitlines()[0]),'Native command does not explicitly select Direct PLE'
   m=re.search(r'STRATA_BATCH_MTP_PLE_PREFETCH requested=(\d+) mode=(ON|OFF) slots=(\d+) wired=(\d+) table_open=(\d+) slot_ple_ready=(\d+) depth=(\d+) vector_budget=(\d+) cap=(\d+)',log)
   assert m and m.groups()==('0','OFF','2','0','1','0','2','6','16'), 'Startup activation/partial-state contract failed'
   assert 'this card holds 149 experts per layer, card 1 the other 363' in log
   assert re.search(r'card 1 has .* its 17424 experts',log)
   marker=f'strata batch commit overlap: enabled={a.arm} batch_mtp=1 (default OFF)'
   assert log.count(marker)==1,'S79 batch overlap startup marker missing/duplicated'
   assert '--batch-mtp: up to 2 proposals' in log
   assert json.loads(subprocess.check_output(['docker','image','inspect',ci['Image']]))[0]['Id']==inputs['image']
   for name,value in [('container-ready.json',ci),('health-ready.json',health)]:
    (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
   print(json.dumps({'arm':a.arm,'loaded':True,'elapsed_s':time.monotonic()-t,'CPU_lease_can_release':True}));break
  status=str(health)
 except (OSError,TimeoutError) as e:status=type(e).__name__
 # Observe an already-ended detached lease promptly; preserve handle/logs and never kill/restart here.
 server_log=out/'server.log'
 if server_log.exists():
  ended=re.search(r'gpu-run: .* done in \d+s \(exit (\d+)\)',server_log.read_text())
  if ended:raise SystemExit(f'Detached GPU lease exited {ended[1]} before readiness; preserve this arm and inspect its logs/unit')
 if status!=last:print(status,flush=True);last=status
 time.sleep(2)
else:raise SystemExit('Readiness observation ended; preserve and inspect same service, do not restart.')
