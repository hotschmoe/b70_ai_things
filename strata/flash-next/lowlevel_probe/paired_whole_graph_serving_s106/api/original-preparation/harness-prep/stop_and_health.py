#!/usr/bin/env python3
"""SIGTERM server -> existing engine.close QUIT; capture final report before health."""
import argparse,hashlib,json,re,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parents[1];REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
sys.path.insert(0,str(REPO/'strata/flash-next'));from bench_serving import get
p=argparse.ArgumentParser();p.add_argument('arm',choices=['0','1']);a=p.parse_args();out=D/f'mode{a.arm}'
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
functional_file=out/'functional/verdict.json'
try:functional_status=json.loads(functional_file.read_text()).get('status')
except (OSError,ValueError):functional_status='NOT_COMPLETED_FAILED_OR_INVALID'
# Cleanup remains available after a failed client/functional gate. Never issue new requests here.
profile=json.loads((D/'serving-inputs.json').read_text());container=profile['containers'][a.arm]
readyfile=out/'container-ready.json'
if readyfile.exists():ready=json.loads(readyfile.read_text());cid=ready['Id']
else:cid=(out/'attempt.cid').read_text().strip()
assert re.fullmatch('[0-9a-f]{64}',cid),'No valid owned CID; preserve attempt/root diagnoses'
inspect_cmd=['docker','inspect',cid]
inspected=subprocess.run(inspect_cmd,capture_output=True,text=True)
save(out/'docker-prestop-inspect.json',{'command':inspect_cmd,'exit':inspected.returncode,'stdout':inspected.stdout,'stderr':inspected.stderr})
ci=json.loads(inspected.stdout)[0] if inspected.returncode==0 else None
if ci is not None:
 assert ci['Id']==cid and ci['Name'].lstrip('/')==container and str(out/'config.json')in ci['Config']['Cmd']
 if readyfile.exists():assert ci['Id']==ready['Id']
 state=ci['State'];prestate={'Running':state['Running'],'OOMKilled':state['OOMKilled'],'inspect_succeeded':True}
 save(out/'docker-prestop.json',ci)
else:
 prestate={'Running':None,'OOMKilled':None,'inspect_succeeded':False}
 save(out/'docker-prestop.json',{'owned_cid':cid,'expected_name':container,'inspect_missing_or_failed':True,'inspect_exit':inspected.returncode,'stderr':inspected.stderr})
prehealthy=prestate['Running']is True and prestate['OOMKilled']is False
save(out/'prestop-state.json',prestate)
try:health=get('http://127.0.0.1:18085','/health')
except Exception as e:health={'observation_failed':type(e).__name__,'detail':str(e),'cleanup_still_required':True}
save(out/'health-prestop.json',health)
# On failed collection/startup, best-effort stop the named profiler in this
# validated owned CID without blocking normal TERM; no flush generation here.
try:stopped=json.loads((out/'unitrace-stop.json').read_text()).get('exit')==0
except (OSError,ValueError):stopped=False
if not stopped and ci is not None and prestate['Running']is True and not(out/'unitrace-cleanup-stop.json').exists():
 cmd=['docker','exec',cid,'/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace','--stop',profile['sessions'][a.arm]]
 try:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);rec={'command':cmd,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 except Exception as e:
  captured=lambda x:x.decode('utf8','replace')if isinstance(x,bytes)else x
  rec={'command':cmd,'exit':None,'exception':type(e).__name__,'detail':str(e),'stdout':captured(getattr(e,'stdout',None)),'stderr':captured(getattr(e,'stderr',None))}
 save(out/'unitrace-cleanup-stop.json',rec)
# A missing/stopped owned container gets no TERM. Authoritative CID/name-gone
# and wrapper-terminal checks still run; that path cannot qualify as healthy.
if ci is not None and prestate['Running']is True:
 quit_cmd=['docker','kill','--signal=TERM',cid]
 r=subprocess.run(quit_cmd,capture_output=True,text=True)
 quit_record={'command':quit_cmd,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'skipped':False}
else:
 quit_record={'command':None,'exit':None,'skipped':True,'reason':'owned container absent/uninspectable or not running'}
save(out/'quit-command.json',quit_record)
entries=[];gone=False;wrapper_exits=[];ps_exit=None
for _ in range(900):
 server=(out/'server.log').read_text()if(out/'server.log').exists()else''
 wrapper_exits=[int(x)for x in re.findall(r'gpu-run: .* done in \d+s \(exit (\d+)\)',server)]
 ps=subprocess.run(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}}'],capture_output=True,text=True);ps_exit=ps.returncode
 if ps_exit==0:
  entries=[x.split()for x in ps.stdout.splitlines()]
  gone=all(x[0]!=cid and x[1]!=container for x in entries)
 else:gone=False
 if gone and wrapper_exits:break
 time.sleep(.1)
removal={'docker_ps_exit':ps_exit,'docker_ps_stderr':ps.stderr,'entries':entries,'created_cid':cid,'expected_name':container,'full_cid_absent':ps_exit==0 and all(x[0]!=cid for x in entries),'name_absent':ps_exit==0 and all(x[1]!=container for x in entries),'epoch_ns':time.time_ns()}
save(out/'container-removal.json',removal)
wrapper=wrapper_exits[0]if len(wrapper_exits)==1 else None
normal=gone and prehealthy and quit_record['exit']==0 and wrapper==0
normal_record={'status':'PASS_NORMAL_TEARDOWN'if normal else'FAIL_ROOT_RECOVERY_LADDER_REQUIRED','signal':'TERM to server -> QUIT to native engine'if not quit_record['skipped']else'none: container absent/stopped','container_removed':gone,'lease_wrapper_exit':wrapper,'all_wrapper_terminal_exits':wrapper_exits,'prestop_state':prestate,'container_missing_or_inspect_failed_before_stop':ci is None,'diagnostic_before_any_forced_container_stop':True,'no_force_stop_used':True,'functional_status':functional_status,'epoch_ns':time.time_ns()}
save(out/'normal-stop.json',normal_record)
if not normal:
 # Do not blindly enter TP collectives/reset after an abnormal TP>1 exit.
 # Root follows the recovery ladder; no requests/force/retry occur here.
 save(out/'recovery-required.json',{'status':'ROOT_RECOVERY_LADDER_REQUIRED','normal_stop':normal_record,'removal':removal,'automatic_post_health_or_collectives_started':False,'automatic_reset':False})
 raise SystemExit('Abnormal/unknown teardown recorded; root recovery ladder required before any further GPU/collective work')
raw=(out/'engine.log').read_bytes();(out/'engine-final.log').write_bytes(raw)
text=raw.decode('utf8');matches=re.findall(r'^strata serve final: (ple io: .*prefetch_gathers=(\d+) prefetch_tokens=(\d+))$',text,re.M)
diag={'present_once':len(matches)==1,'raw':matches,'log_sha256':hashlib.sha256(raw).hexdigest(),'scope':'process cumulative, includes paused warm and profiled admission/pair; no flush request; not per-window'}
if len(matches)==1:
 diag.update(prefetch_gathers=int(matches[0][1]),prefetch_tokens=int(matches[0][2]),requests=int(re.search(r'^ple io: (\d+) rows',matches[0][0])[1]))
save(out/'aggregate-proof.json',diag)
records={}
for label,cmd in [('card',['bin/gpu-run','bin/xpu-health','--img','b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78','--timeout','400']),('collective',['bin/gpu-run','bin/xpu-collective-health','--timeout','400'])]:
 t=time.monotonic()
 with (out/(label+'-health.log')).open('w') as f:r=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
 records[label]={'command':cmd,'exit':r.returncode,'elapsed_s':time.monotonic()-t};save(out/'post-health.json',records)
 assert r.returncode==0,'Root follows health/recovery ladder; no automatic retry'
assert diag['present_once'],'Normal-QUIT aggregate proof absent/duplicated'
