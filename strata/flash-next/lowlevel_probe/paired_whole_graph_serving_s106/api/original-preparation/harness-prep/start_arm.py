#!/usr/bin/env python3
"""Root executes under cpu-run for startup only. systemd unit retains pair lease."""
import argparse,hashlib,json,re,subprocess,time,urllib.request
from pathlib import Path
D=Path(__file__).resolve().parents[1];REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
p=argparse.ArgumentParser();p.add_argument('arm',choices=['0','1']);a=p.parse_args();out=D/f'mode{a.arm}'
assert not (out/'start-result.json').exists(),'Do not restart a prior arm'
inputs=json.loads((D/'serving-inputs.json').read_text());config=out/'config.json';c=json.loads(config.read_text())
arm=inputs['arms'][a.arm]
assert c['exe']==str(out/'trace-engine.sh')
with Path(arm['exe']).open('rb') as binary:assert hashlib.file_digest(binary,'sha256').hexdigest()==arm['sha256']
assert all(k not in c['env'] for k in ['STRATA_HOST_BOUNDARY','STRATA_VERIFY_PROFILE','STRATA_PREFILL_HOST_BOUNDARY','STRATA_PREFILL_TIMING','STRATA_PREFILL_SYNC','STRATA_VERIFY_TRACE','STRATA_VERIFY_EAGER'])
assert c['env']['STRATA_TRACE']=='1' and c['env']['STRATA_BATCH_MTP_PLE_PREFETCH']=='0'
assert 'STRATA_BATCH_COMMIT_OVERLAP' not in c['env']
assert c['env']['STRATA_BATCH_MTP_PAIR_GRAPH']==a.arm and c['env']['STRATA_IQ2XS_MULTI']=='0' and c['env']['STRATA_HEAD_Q5K_RPW2']=='0' and c['env']['STRATA_MMVQ_WIDE_K_RPW']=='1' and c['env']['STRATA_MMVQ_WIDE_K']=='1'
assert all(c['env'][k]==v for k,v in [('STRATA_MMVQ_WIDE','1'),('STRATA_MMVQ_WIDE_MIN','0'),('STRATA_MMVQ_WIDE_MAXC','8'),('STRATA_Q6_PACKED','0')])
assert c['env']['STRATA_MMVQ_A2']=='1' and c['env']['STRATA_MMVQ_WIDE_RPW']=='1' and c['env']['STRATA_MMVQ_PF']=='0'
assert inputs['same_binary'] and inputs['arms']['0']==inputs['arms']['1'] and inputs['HC_state']=='COMMON_LO_ONCE_S51'
assert c['args'][c['args'].index('--native')+1]==inputs['native']['path']
assert c['args'][c['args'].index('--ple-gguf')+1]==inputs['native']['sibling_path']
assert c['args'][c['args'].index('--ple-io')+1]=='direct','S106 requires explicit Direct PLE'
assert all(c['env'][k]==v for k,v in [('STRATA_Q6_CENTER','0'),('STRATA_QSA_PREFILL_TILED','0'),('STRATA_ATTN_VALUE_PIPELINE','0'),('STRATA_PROMPT_ATTN_XMX','0'),('STRATA_KV_ROT','0'),('STRATA_BATCH_PREFILL_QUANTUM','0'),('STRATA_PREFILL_FIRST','0'),('STRATA_EXPERT_SPLIT','0'),('STRATA_GU_LANES','8')])
for key,value in inputs['fixed_work_args'].items():
 assert c['args'].count(key)==1 and c['args'][c['args'].index(key)+1]==value
for path,row in json.loads((D/'compiled-tool-pins.json').read_text()).items():
 with Path(path).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
container=inputs['containers'][a.arm]
(out/'trace').mkdir(exist_ok=False)
assert not subprocess.check_output(['docker','ps','-aq','--filter',f'name=^{container}$']).strip(),'Existing S106 container must be handled by owner'
try:
 with urllib.request.urlopen('http://127.0.0.1:18085/health',timeout=2) as f:existing=json.load(f)
except OSError:pass
else:raise SystemExit('Port18085 already answers; preserve existing service, do not launch over it')
unit=f'strata-pair-graph-s106api-mode{a.arm}'
# systemd manager creates the service; no cpu-run fd is transferred through DBus.
# Explicitly close 7 again in the detached service before gpu-run acquires 8/9.
cmd=['systemd-run','--user',f'--unit={unit}','--collect',f'--setenv=B70_S106API_CONTAINER={container}',f'--property=WorkingDirectory={REPO}',
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
   assert (out/'attempt.cid').read_text().strip()==ci['Id'],'Created CID differs from inspected owned container'
   assert health['model']=='hotschmoe-dd' and health['max_context']==200000
   log=(out/'engine.log').read_text();assert c['exe'] in log.splitlines()[0]
   assert inputs['native']['path'] in log.splitlines()[0] and inputs['native']['sibling_path'] in log.splitlines()[0]
   assert re.search(r'--ple-io direct(?: |$)',log.splitlines()[0]),'Native command does not explicitly select Direct PLE'
   m=re.search(r'STRATA_BATCH_MTP_PLE_PREFETCH requested=(\d+) mode=(ON|OFF) slots=(\d+) wired=(\d+) table_open=(\d+) slot_ple_ready=(\d+) depth=(\d+) vector_budget=(\d+) cap=(\d+)',log)
   assert m and m.groups()==('0','OFF','2','0','1','0','2','6','16'), 'Startup activation/partial-state contract failed'
   assert 'this card holds 149 experts per layer, card 1 the other 363' in log
   assert re.search(r'card 1 has .* its 17424 experts',log)
   assert '--batch-mtp: up to 2 proposals' in log
   assert json.loads(subprocess.check_output(['docker','image','inspect',ci['Image']]))[0]['Id']==inputs['image']
   probe=r"""
import os,json,hashlib,sys
exe,expected,envjson=sys.argv[1:];env=json.loads(envjson);rows=[]
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 try:
  actual=os.readlink('/proc/'+pid+'/exe')
  if actual!=exe:continue
  with open('/proc/'+pid+'/exe','rb')as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  effective=dict(x.split('=',1)for x in open('/proc/'+pid+'/environ','rb').read().decode().split('\0')if'='in x)
  assert h==expected
  assert all(effective.get(k)==v for k,v in env.items())
  assert all(k not in effective for k in ['STRATA_BATCH_COMMIT_OVERLAP','STRATA_HOST_BOUNDARY','STRATA_VERIFY_PROFILE','STRATA_PREFILL_HOST_BOUNDARY','STRATA_PREFILL_TIMING','STRATA_PREFILL_SYNC','STRATA_VERIFY_TRACE','STRATA_VERIFY_EAGER'])
  assert 'unitrace'in effective.get('LD_PRELOAD','').lower()
  assert effective.get('UNITRACE_StartPaused')=='1'
  assert effective.get('UNITRACE_HostTiming')=='1'
  assert effective.get('UNITRACE_Session')=='s106api'+env['STRATA_BATCH_MTP_PAIR_GRAPH']
  assert effective.get('UNITRACE_FollowChildProcess')=='0'
  assert all(effective.get(k,'')in['','0']for k in ['UNITRACE_ChromeCallLogging','UNITRACE_CallLogging','UNITRACE_DeviceTiming','UNITRACE_DeviceTimeline','UNITRACE_KernelSubmission','UNITRACE_ChromeKernelLogging','UNITRACE_ChromeDeviceLogging','UNITRACE_OpenCLTracing','UNITRACE_ChromeSyclRuntimeLogging','UNITRACE_ChromeUrLogging','UNITRACE_ChromeIttLogging','UNITRACE_ChromeMpiLogging','UNITRACE_ChromeOmpLogging','UNITRACE_MetricQuery','UNITRACE_KernelMetrics','UNITRACE_SystemTime'])
  argv=open('/proc/'+pid+'/cmdline','rb').read().decode().rstrip('\0').split('\0')
  for key,value in {'--suffix-draft':'0','--lookup-chain':'0','--spec':'4','--mtp-max-t':'4','--spec-min-p':'0'}.items():
   assert argv.count(key)==1 and argv[argv.index(key)+1]==value,'Effective fixed-work argv differs'
  maps=open('/proc/'+pid+'/maps').read()
  libraries={};unitrace_libs={}
  for line in maps.splitlines():
   fields=line.split(maxsplit=5)
   if len(fields)==6 and '/libstdc++.so.' in fields[5]:
    path=fields[5];assert not path.endswith(' (deleted)')
    if path not in libraries:
     with open(path,'rb')as lib:lh=hashlib.file_digest(lib,'sha256').hexdigest()
     libraries[path]={'realpath':os.path.realpath(path),'sha256':lh}
   if len(fields)==6 and '/libunitrace_tool.so' in fields[5]:
    path=fields[5];assert not path.endswith(' (deleted)')
    if path not in unitrace_libs:
     with open(path,'rb')as lib:lh=hashlib.file_digest(lib,'sha256').hexdigest()
     assert os.path.realpath(path)=='/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/libunitrace_tool.so'
     assert lh=='66ec9ee2dfd14866162761ce45c62317e91b5e65b9686cb2954c6c21fda58bc6'
     unitrace_libs[path]={'realpath':os.path.realpath(path),'sha256':lh}
  assert len(libraries)==1,'Native libstdc++ provenance ambiguous'
  assert len(unitrace_libs)==1,'Native unitrace library provenance ambiguous'
  loader={k:effective.get(k)for k in ['LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT']}
  assert not loader['LD_AUDIT'],'Unexpected loader audit changes measurement'
  rows.append({'pid':int(pid),'argv':argv,'exe':actual,'sha256':h,'effective_config_env':{k:effective[k]for k in env},'profiling_mode':'HOST_API_SUMMARY_ONLY_START_PAUSED_NO_KERNEL_TRACING','unitrace_environment':{k:v for k,v in effective.items()if k.startswith('UNITRACE_')},'loader_environment':loader,'proc_maps':maps,'proc_maps_sha256':hashlib.sha256(maps.encode()).hexdigest(),'loaded_libstdcxx':libraries,'loaded_unitrace':unitrace_libs,'provenance_scope':'Actual native before warm/timing; does not establish historical S95 loaded-library identity or authorize a clockjoin.'})
 except FileNotFoundError:continue
assert len(rows)==1,rows
print(json.dumps(rows))
"""
   observed=json.loads(subprocess.check_output(['docker','exec',ci['Id'],'/mnt/vm_8tb/b70/build/strata-ab/venv/bin/python','-c',probe,arm['exe'],arm['sha256'],json.dumps(c['env'])],text=True))
   (out/'native-process-identity.json').write_text(json.dumps(observed,indent=2)+'\n',encoding='ascii')
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
