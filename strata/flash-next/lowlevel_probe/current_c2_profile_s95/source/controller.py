#!/usr/bin/env python3
"""Root-only phases: start (CPU lease), collect (quiet CPU), stop (quiet CPU).
The detached launch owns both GPU leases for the complete server lifetime.
No phase retries/restarts a process or launches another profile request.
"""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,sys,time,urllib.request
P=Path(__file__).resolve().parent;OUT=P/'run';REPO=Path('/mnt/vm_8tb/github/b70_ai_things');B=Path('/mnt/vm_8tb/b70/build/strata-ab')
sys.path.insert(0,str(REPO/'strata/flash-next'));from bench_serving import get,send
I=json.loads((P/'inputs.json').read_text());C=json.loads((P/'config.json').read_text());BASE='http://127.0.0.1:18085';NAME=I['container'];UNIT='strata-current-c2-profile-s95';PROF=str(B/'pti-gpu/tools/unitrace/build/unitrace')

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,v):
 with(OUT/name).open('x',encoding='ascii')as f:json.dump(v,f,indent=2,ensure_ascii=True);f.write('\n');f.flush()
def check_source():
 for row in(P/'expected-source.sha256').read_text().splitlines():
  h,n=row.split(maxsplit=1);assert sha(P/n)==h,n
 assert C['env']==I['config_env'] and C['env']['STRATA_QSA_PREFILL_TILED']=='0'and C['env']['STRATA_ATTN_VALUE_PIPELINE']=='0'and C['env']['STRATA_HOST_BOUNDARY']=='1'and 'STRATA_VERIFY_PROFILE'not in C['env'] and C['env']['STRATA_Q6_CENTER']=='0'
 assert C['env']['STRATA_PROMPT_ATTN_XMX']=='0'and C['env']['STRATA_KV_ROT']=='0'
 assert C['exe']==str(P/'trace-engine.sh')
def identity():
 m=get(BASE,'/v1/models');ids=[x['id']for x in m['data']]
 assert ids[0]=='hotschmoe-dd' and 'qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata'in ids
 assert ids[1]in(REPO/'evals/configs/models.yaml').read_text();return m

def start():
 # Root must check/reuse valid full pair prehealth before this explicit phase.
 OUT.mkdir();(OUT/'trace').mkdir();check_source()
 assert sha(I['binary'])==I['binary_sha256']
 save('profiler-identity.json',{str(f):{'sha256':sha(f),'bytes':f.stat().st_size}for f in [Path(PROF),Path(PROF).parent/'libunitrace_tool.so']})
 assert not subprocess.check_output(['docker','ps','-aq','--filter',f'name=^{NAME}$']).strip()
 try:get(BASE,'/health')
 except OSError:pass
 else:raise RuntimeError('Port18085 already answers; preserve existing service')
 cmd=['systemd-run','--user',f'--unit={UNIT}','--collect',f'--property=WorkingDirectory={REPO}',f'--property=StandardOutput=append:{OUT}/server.log',f'--property=StandardError=append:{OUT}/server.log','/bin/bash','-c','exec 7>&-; exec /bin/bash "$1" "$2"','bash',str(P/'launch.sh'),str(P/'config.json')]
 t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,close_fds=True);save('start-result.json',{'command':cmd,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'unit':UNIT});assert r.returncode==0
 while time.monotonic()-t<1200:
  try:
   h=get(BASE,'/health')
   if h.get('loaded')is True:
    ci=json.loads(subprocess.check_output(['docker','inspect',NAME]))[0];assert str(P/'config.json')in ci['Config']['Cmd'] and ci['State']['Running'] and not ci['State']['OOMKilled'];assert ci['Image']==I['image']
    assert h['model']=='hotschmoe-dd' and h['max_context']==200000
    log=(OUT/'engine.log').read_text();assert str(P/'trace-engine.sh')in log.splitlines()[0]
    assert 'this card holds 149 experts per layer, card 1 the other 363'in log and re.search(r'card 1 has .* its 17424 experts',log)
    assert 'pre-filled 7152 of 7152 slots'in log
    assert '--batch-mtp: up to 2 proposals'in log and '--ple-io direct'in log.splitlines()[0]
    assert 'strata QSA prefill tiled:'not in log,'No request or eligible QSA selection before collection'
    # This excludes only known logged capture completions, not all possible graph work.
    assert not re.search(r'^strata verify: captured (?:the \d+-token window \(upload no error, sync no error\)|the batch window over slots [0-9,]+)$',log,re.M),'Known native capture completion logged before collection'
    probe=r'''
import os,json,hashlib,sys
exe,expected,envjson=sys.argv[1:];env=json.loads(envjson);rows=[]
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 try:
  if os.readlink('/proc/'+pid+'/exe')!=exe:continue
  with open('/proc/'+pid+'/exe','rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
  e=dict(x.split('=',1)for x in open('/proc/'+pid+'/environ','rb').read().decode().split('\0')if'='in x)
  assert all(e.get(k)==v for k,v in env.items())
  assert e.get('STRATA_PROMPT_ATTN_XMX')=='0'and e.get('STRATA_KV_ROT')=='0'
  assert 'unitrace'in e.get('LD_PRELOAD','').lower()
  assert 'STRATA_VERIFY_PROFILE'not in e and 'STRATA_PREFILL_TIMING'not in e and 'STRATA_PREFILL_SYNC'not in e and 'STRATA_VERIFY_EAGER'not in e and 'STRATA_VERIFY_TRACE'not in e
  assert e.get('UNITRACE_StartPaused')=='1' and e.get('UNITRACE_SystemTime','0')in['','0']
  rows.append({'pid':int(pid),'exe':exe,'sha256':expected,'effective_config_env':{k:e[k]for k in env},'LD_PRELOAD':e.get('LD_PRELOAD'),'UNITRACE_StartPaused':e.get('UNITRACE_StartPaused'),'UNITRACE_SystemTime':e.get('UNITRACE_SystemTime')})
 except FileNotFoundError:continue
assert len(rows)==1,rows
print(json.dumps(rows))
'''
    native=json.loads(subprocess.check_output(['docker','exec',ci['Id'],str(B/'venv/bin/python'),'-c',probe,I['binary'],I['binary_sha256'],json.dumps(C['env'])],text=True))
    save('native-process-identity.json',native);save('container-ready.json',ci);save('health-ready.json',h);save('models.json',identity());save('loaded.json',{'status':'READY_NO_REQUESTS','elapsed_s':time.monotonic()-t,'CPU_lease_can_release':True});return
  except (OSError,TimeoutError):pass
  if(OUT/'server.log').exists()and re.search(r'gpu-run: .* done in \d+s \(exit \d+\)',(OUT/'server.log').read_text()):raise RuntimeError('Detached lease ended before readiness; preserve same attempt')
  time.sleep(2)
 raise RuntimeError('Readiness observation expired; preserve same service/lease, no restart')

def control(action,label=None):
 ci=json.loads((OUT/'container-ready.json').read_text());cmd=['docker','exec',ci['Id'],PROF,'--'+action,I['session']]
 try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 except subprocess.TimeoutExpired as e:
  decoded=lambda x:x.decode('utf8','replace')if isinstance(x,bytes)else x
  save('unitrace-'+(label or action)+'.json',{'command':cmd,'exit':None,'timeout_s':30,'exception':type(e).__name__,'stdout':decoded(e.stdout),'stderr':decoded(e.stderr)})
  raise RuntimeError('Profiler control timed out; preserve same attempt, cleanup still required')from e
 save('unitrace-'+(label or action)+'.json',{'command':cmd,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0

def ring_snapshot():
 # Reuse actual S32 parser; importing this guarded module does not run analysis.
 import importlib.util
 path=P.parent/'host-boundary-s32/analysis/analyze.py'
 spec=importlib.util.spec_from_file_location('s32_ring',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 offset=json.loads((OUT/'pair/window-start.json').read_text())['byte_offset']
 for _ in range(300):
  slots=get(BASE,'/slots');raw=(OUT/'engine.log').read_bytes()[offset:];text=raw.decode('utf8');errors=[];dumps=module.parse(text.splitlines(),errors)
  records=[r for d in dumps for r in d['records']]
  complete=not errors and records and all(0<d['declared_records']<=8192 and d['dropped']==0 for d in dumps) and records[-1]['tag']=='window_end'
  if len(slots)==2 and all(x['is_processing']is False for x in slots) and complete:
   (OUT/'pair-engine-complete.log').write_bytes(raw)
   save('slots-idle.json',slots);save('ring-closure.json',{'status':'COMPLETE_DECLARED_COUNTS_DROPPED0_AND_FINAL_WINDOW_END','log_sha256':hashlib.sha256(raw).hexdigest(),'dump_headers':[{k:v for k,v in d.items()if k!='records'}for d in dumps],'actual_records':len(records),'window0_records':sum(r['w']==0 for r in records),'all_cold_solo_tail_records_retained':True,'no_footer_exists':True,'scope':'Idle observation AND complete declared dump; snapshot before optional flush.'});return
  time.sleep(.1)
 raise RuntimeError('Idle plus complete ring dump unproved; no regeneration/retry')

def trace_closed_metadata():
 # Only bounded tail/metadata checks here. The ONE full JSON parse is after stop.
 fs=list((OUT/'trace').glob('*.json'))
 if len(fs)!=1:return None
 f=fs[0];a=f.stat()
 if a.st_size<=0:return None
 with f.open('rb')as stream:
  stream.seek(max(0,a.st_size-512));tail=stream.read().rstrip()
 if not tail.endswith((b']',b'}')):return None
 time.sleep(.5);b=f.stat()
 if(a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns):return None
 return {'path':str(f),'bytes':b.st_size,'mtime_ns':b.st_mtime_ns,'bounded_tail_and_stable_metadata_only':True,'JSON_validity_unproved':True}

def collect():
 check_source();assert(OUT/'loaded.json').exists()and not(OUT/'collection-start.json').exists()
 for row in I['saved_request_files']:assert sha(row['path'])==row['sha256']
 save('collection-models.json',identity())
 for script,name in [('calibrate.py','timebase-before.json'),('clock_bridge.py','clock-bridge-before.json')]:subprocess.run([sys.executable,str(P/script),'--output',str(OUT/name)],check=True)
 save('collection-start.json',{'epoch_us':time.time_ns()/1000,'request_source_sha256':[x['sha256']for x in I['saved_request_files']],'capture':'Injected from start; no generation warmup; resume before first GEN/capture','request':'One ordered107prompt/64output EACH coding pair','native_steady_to_trace_join_allowed':False})
 begin=time.time_ns()/1000;end=begin;failure=None;controls=[]
 try:
  control('resume');begin=time.time_ns()/1000
  with(OUT/'pair-client.log').open('x')as f:r=subprocess.run([sys.executable,str(P/'pair.py')],stdout=f,stderr=subprocess.STDOUT)
  save('pair-client-exit.json',{'exit':r.returncode});assert r.returncode==0,'Pair failed: no extra request'
  ring_snapshot();end=time.time_ns()/1000
 except Exception as e:failure=e;save('collection-failure.json',{'type':type(e).__name__,'detail':str(e),'no_retry':True})
 finally:
  end=max(end,time.time_ns()/1000)
  # Stop is attempted even if resume/pause/pair failed; cleanup remains independent.
  for action in ['pause','stop']:
   try:control(action)
   except Exception as e:controls.append({'action':action,'error_type':type(e).__name__,'detail':str(e)})
  for script,name in [('calibrate.py','timebase-after.json'),('clock_bridge.py','clock-bridge-after.json')]:
   try:subprocess.run([sys.executable,str(P/script),'--output',str(OUT/name)],check=True)
   except Exception as e:controls.append({'clock':name,'error_type':type(e).__name__,'detail':str(e)})
  save('window.json',{'start_us':begin,'end_us':end,'scope':'One instrumented pair plus completion observation, before pause/stop/optional flush; all capture/solo/tail retained','native_steady_to_trace_join_allowed':False,'control_errors':controls})
 if failure is not None:raise failure
 assert not controls,'Stop/calibration failed; no extra generation'
 trace=None
 for _ in range(10):
  trace=trace_closed_metadata()
  if trace:break
  time.sleep(.2)
 if trace:save('flush-decision.json',{'needed':False,'reason':'One trace has closing tail and stable metadata after explicit stop; JSON validation deferred'})
 else:
  save('flush-decision.json',{'needed':True,'reason':'Explicit stop not yet observed as closed stable trace; ONE existing unprofiled native flush only','excluded_from_pair_window_and_native_counters':True})
  flush=send(BASE,'hotschmoe-dd',[{'role':'user','content':'[post trace canary] Return only the integer: 17 * 19.'}],32);save('post-stop-flush.json',flush);assert flush['text'].strip()=='323'
  for _ in range(60):
   trace=trace_closed_metadata()
   if trace:break
   time.sleep(.5)
 assert trace,'No single closed stable trace: preserve attempt, no repeat flush'
 save('collection.json',{'status':'PAIR_PREFIX_USAGE_NATIVE_WORK_RING_AND_STOP_PASS_TRACE_UNPARSED','trace':trace,'GPU_dispatch_unproved_until_analysis':True,'no_raw_trace_parsed':True,'instrumented_speed_qualified':False})


def stop():
 subprocess.run([sys.executable,str(P/'stop.py')],check=True)
 check_source();assert sha(I['binary'])==I['binary_sha256']
 tools=json.loads((OUT/'profiler-identity.json').read_text());assert all(sha(f)==v['sha256']for f,v in tools.items())
 save('closure.json',{'status':'NORMAL_TEARDOWN_POSTHEALTH_PASS','source_pins_rechecked':True,'trace_JSON_validated':False,'serving_or_speed_qualified':False})

def analyze():
 check_source();assert json.loads((OUT/'closure.json').read_text())['status']=='NORMAL_TEARDOWN_POSTHEALTH_PASS'
 c=json.loads((OUT/'collection.json').read_text());trace=Path(c['trace']['path']);assert trace.stat().st_size==c['trace']['bytes']
 w=json.loads((OUT/'window.json').read_text());analysis=OUT/'analysis';analysis.mkdir()
 cmd=[sys.executable,str(P/'analyze_trace.py'),str(trace),'--timebase',str(OUT/'timebase-before.json'),'--start-us',str(w['start_us']),'--end-us',str(w['end_us']),'--output',str(analysis/'gpu')]
 with(analysis/'gpu-parser.log').open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 save('raw-parser-exit.json',{'command':cmd,'exit':r.returncode,'one_full_raw_JSON_parse_only':True});assert r.returncode==0
 cmd=[sys.executable,str(P/'cpu_markers.py'),str(OUT/'pair-engine-complete.log'),'--output',str(analysis/'cpu')]
 with(analysis/'cpu-parser.log').open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 save('cpu-parser-exit.json',{'command':cmd,'exit':r.returncode});assert r.returncode==0
 save('analysis-limits.json',{'status':'SEPARATE_CPU_GPU_DIAGNOSTICS_REQUIRE_ROOT_REVIEW','native_steady_to_MONOTONIC_identity':'UNVERIFIED','native_to_trace_timestamp_join_allowed':False,'ordinal_reconciliation_only':True,'no_speed_qualification':True,'inadequate_replay_cycles':'INCONCLUSIVE_NO_RERUN','limitations':['No physical idle/bandwidth/traffic or removable-gap claim.','No cross-card clock join or additive CPU/GPU/engine budgets.','Kernel name/grid may not distinguish shared projections from other weights.','Wait invocations are not necessarily forced barriers; wait return is not independently measured GPU completion.','CPU first-observed role keys and actual L0 create/finalize retain all cold cycles; root reconciles ordinal evidence before choosing replay census.']})


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['start','collect','stop','analyze']);args=ap.parse_args();globals()[args.phase]()
