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
container=json.loads((D/'serving-inputs.json').read_text())['containers'][a.arm]
ci=json.loads(subprocess.check_output(['docker','inspect',container]))[0]
ready=json.loads((out/'container-ready.json').read_text())
assert (out/'attempt.cid').read_text().strip()==ci['Id'],'Only the CID created by this arm may be stopped'
assert ci['Id']==ready['Id'] and str(out/'config.json') in ci['Config']['Cmd'] and ci['State']['Running'] and not ci['State']['OOMKilled']
save(out/'docker-prestop.json',ci)
try:health=get('http://127.0.0.1:18085','/health')
except Exception as e:health={'observation_failed':type(e).__name__,'detail':str(e),'cleanup_still_required':True}
save(out/'health-prestop.json',health)
# Python server handles TERM as Ctrl+C and invokes engine.close(), sending QUIT.
# Do not docker-stop/kill while waiting for the engine's normal final diagnostic.
r=subprocess.run(['docker','kill','--signal=TERM',ci['Id']],capture_output=True,text=True)
save(out/'quit-command.json',{'command':['docker','kill','--signal=TERM',ci['Id']],'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0
for _ in range(900):
 server=(out/'server.log').read_text();entries=[x.split() for x in subprocess.check_output(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}}'],text=True).splitlines()];gone=all(x[0]!=ci['Id'] and x[1]!=container for x in entries)
 if gone and re.search(r'gpu-run: .* done in \d+s \(exit 0\)',server):break
 time.sleep(.1)
else:raise SystemExit('Normal exit missing: preserve evidence/service and root diagnoses; no automatic force kill/restart')
# Container --rm completion is the normal cleanup; all logs are bind-mounted.
save(out/'container-removal.json',{'docker_ps_exit':0,'entries':entries,'created_cid':ci['Id'],'expected_name':container,'full_cid_absent':all(x[0]!=ci['Id']for x in entries),'name_absent':all(x[1]!=container for x in entries),'epoch_ns':time.time_ns()})
raw=(out/'engine.log').read_bytes();(out/'engine-final.log').write_bytes(raw)
text=raw.decode('utf8');matches=re.findall(r'^strata serve final: (ple io: .*prefetch_gathers=(\d+) prefetch_tokens=(\d+))$',text,re.M)
diag={'present_once':len(matches)==1,'raw':matches,'log_sha256':hashlib.sha256(raw).hexdigest(),'scope':'process cumulative: includes prefill/admission/functionals, not coding-only'}
if len(matches)==1:
 diag.update(prefetch_gathers=int(matches[0][1]),prefetch_tokens=int(matches[0][2]),requests=int(re.search(r'^ple io: (\d+) rows',matches[0][0])[1]))
save(out/'aggregate-proof.json',diag)
save(out/'normal-stop.json',{'signal':'TERM to server -> QUIT to native engine','container_removed':True,'lease_wrapper_exit':0,'diagnostic_before_any_forced_container_stop':True,'no_force_stop_used':True,'functional_status':functional_status,'epoch_ns':time.time_ns()})
records={}
for label,cmd in [('card',['bin/gpu-run','bin/xpu-health','--img','b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78','--timeout','400']),('collective',['bin/gpu-run','bin/xpu-collective-health','--timeout','400'])]:
 t=time.monotonic()
 with (out/(label+'-health.log')).open('w') as f:r=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
 records[label]={'command':cmd,'exit':r.returncode,'elapsed_s':time.monotonic()-t};save(out/'post-health.json',records)
 assert r.returncode==0,'Root follows health/recovery ladder; no automatic retry'
assert diag['present_once'],'Normal-QUIT aggregate proof absent/duplicated'
