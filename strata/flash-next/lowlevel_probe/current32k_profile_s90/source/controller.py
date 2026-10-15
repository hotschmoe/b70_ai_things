#!/usr/bin/env python3
"""Root-only phases: start (CPU lease), collect (quiet CPU), stop (quiet CPU).
The detached launch owns both GPU leases for the complete server lifetime.
No phase retries/restarts a process or launches another profile request.
"""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,sys,time,urllib.request
P=Path(__file__).resolve().parent;OUT=P/'run';REPO=Path('/mnt/vm_8tb/github/b70_ai_things');B=Path('/mnt/vm_8tb/b70/build/strata-ab')
sys.path.insert(0,str(REPO/'strata/flash-next'));from bench_serving import get,send
I=json.loads((P/'inputs.json').read_text());C=json.loads((P/'config.json').read_text());BASE='http://127.0.0.1:18085';NAME=I['container'];UNIT='strata-current32k-profile-s90';PROF=str(B/'pti-gpu/tools/unitrace/build/unitrace')

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,v):
 with(OUT/name).open('x',encoding='ascii')as f:json.dump(v,f,indent=2,ensure_ascii=True);f.write('\n');f.flush()
def check_source():
 for row in(P/'expected-source.sha256').read_text().splitlines():
  h,n=row.split(maxsplit=1);assert sha(P/n)==h,n
 assert C['env']==I['config_env'] and C['env']['STRATA_QSA_PREFILL_TILED']=='1' and C['env']['STRATA_Q6_CENTER']=='0'
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
  assert 'unitrace'in e.get('LD_PRELOAD','').lower()
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
 ci=json.loads((OUT/'container-ready.json').read_text());cmd=['docker','exec',ci['Id'],PROF,'--'+action,I['session']];r=subprocess.run(cmd,capture_output=True,text=True);save('unitrace-'+(label or action)+'.json',{'command':cmd,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0

def collect():
 check_source();assert(OUT/'loaded.json').exists()and not(OUT/'collection-start.json').exists()
 assert sha(I['saved_request_file'])==I['saved_request_file_sha256'];ref=json.loads(Path(I['saved_request_file']).read_text());body=dict(ref['request']);body['max_tokens']=32
 assert body['model']=='hotschmoe-dd'and len(body['messages'])==1
 save('collection-models.json',identity());subprocess.run([sys.executable,str(P/'calibrate.py'),'--output',str(OUT/'timebase-before.json')],check=True)
 save('collection-start.json',{'epoch_us':time.time_ns()/1000,'request_source_sha256':I['saved_request_file_sha256'],'override':{'max_tokens':32},'capture':'Profiler injected from process start; no earlier generation; resumed before first request'})
 failure=None;row=None;begin=time.time_ns()/1000;resumed=False
 try:
  control('resume');resumed=True;begin=time.time_ns()/1000
  row=send(BASE,body['model'],body['messages'],32);save('p32768-cap32.json',row)
 finally:
  end=time.time_ns()/1000
  if resumed:
   # Explicit stop must remain available even when send/assertions fail; no extra profile request.
   try:control('pause')
   finally:control('stop')
  subprocess.run([sys.executable,str(P/'calibrate.py'),'--output',str(OUT/'timebase-after.json')],check=True)
  save('window.json',{'start_us':begin,'end_us':end,'scope':'One instrumented full request includes capture/prefill/32decode; no clean latency claim; no cross-card clock join'})
 assert row is not None
 # _Exit skips profiler destructors. One existing unprofiled arithmetic request makes
 # native observe the explicit stop and flush pending reports before normal QUIT.
 flush=send(BASE,'hotschmoe-dd',[{'role':'user','content':'[post trace canary] Return only the integer: 17 * 19.'}],32);save('post-stop-flush.json',flush);assert flush['text'].strip()=='323'
 expected=dict(body);assert row['request']==expected
 u=row['usage'];t=row['timings'];assert u['prompt_tokens']==32728 and u['completion_tokens']==32 and u['prompt_tokens_details']['cached_tokens']<=64
 assert t['prompt_n']+t['cache_n']==32728 and t['cache_n']==u['prompt_tokens_details']['cached_tokens']and t['predicted_n']==32
 assert row['text'] and ref['text'].startswith(row['text']),'Bounded greedy prefix differs from saved S38 reference'
 native_re=re.compile(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms .*?, (\d+) generated in ([\d.]+) ms .*?, drafts accepted (\d+) of (\d+), (\d+) checkpoints',re.M)
 for _ in range(300):
  engine=(OUT/'engine.log').read_text();native=[m for m in native_re.finditer(engine)if m[1]=='32728'and m[5]=='32']
  if len(native)==1:break
  time.sleep(.1)
 else:raise RuntimeError('Native full-request summary missing/duplicated; preserve same attempt')
 assert int(native[0][2])==t['cache_n'] and int(native[0][3])==t['prompt_n']
 save('native-request-work.json',{'summary':native[0].group(0),'groups':native[0].groups(),'scope':'One32728prompt/32completion summary; flush request excluded by shape'})
 marker=re.findall(r'^strata QSA prefill tiled: enabled=(\d+) nq=256 idx=4x128 R4 stride=50002 active=4097\.\.50001 first_active=(\d+) tile=8 key_wg=256 SG32$',(OUT/'engine.log').read_text(),re.M)
 assert marker==[('1','4097')],marker
 for _ in range(60):
  fs=list((OUT/'trace').glob('*.json'))
  if fs and all(f.stat().st_size>0 for f in fs):break
  time.sleep(1)
 else:raise RuntimeError('No nonempty stopped trace; preserve attempt, no rerun')
 save('collection.json',{'status':'REQUEST_USAGE_PREFIX_MARKER_AND_STOP_PASS_TRACE_UNPARSED','files':[{'path':str(f),'bytes':f.stat().st_size}for f in fs],'QSA_marker':marker,'GPU_dispatch_unproved_until_trace_analysis':True,'no_raw_trace_parsed':True,'instrumented_speed_qualified':False})

def stop():
 # Available independently after failed collect. Do not issue new requests here.
 ready_file=OUT/'container-ready.json'
 if ready_file.exists():
  ready=json.loads(ready_file.read_text());cid=ready['Id']
 else:
  cid=(OUT/'attempt.cid').read_text().strip()
 assert re.fullmatch('[0-9a-f]{64}',cid),'No valid owned full CID; preserve attempt and root diagnoses'
 ci=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
 assert ci['Id']==cid and ci['Name'].lstrip('/')==NAME and str(P/'config.json')in ci['Config']['Cmd']
 if ready_file.exists():assert ci['Id']==ready['Id']
 # Startup may fail before ready metadata, or collection before explicit stop.
 # Stop only the named profiler inside this validated owned CID. Failure here
 # must not prevent normal TERM -> native QUIT; never send another generation.
 stopped=False
 stop_file=OUT/'unitrace-stop.json'
 if stop_file.exists():stopped=json.loads(stop_file.read_text()).get('exit')==0
 if not stopped and not(OUT/'unitrace-cleanup-stop.json').exists():
  cmd=['docker','exec',cid,PROF,'--stop',I['session']]
  try:
   cleanup=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
   cleanup_record={'command':cmd,'exit':cleanup.returncode,'stdout':cleanup.stdout,'stderr':cleanup.stderr}
  except Exception as e:
   captured=lambda x:x.decode('utf8','replace')if isinstance(x,bytes)else x
   cleanup_record={'command':cmd,'exit':None,'exception':type(e).__name__,'detail':str(e),'stdout':captured(getattr(e,'stdout',None)),'stderr':captured(getattr(e,'stderr',None)),'normal_TERM_still_required':True}
  save('unitrace-cleanup-stop.json',cleanup_record)
 r=subprocess.run(['docker','kill','--signal=TERM',ci['Id']],capture_output=True,text=True);save('quit-command.json',{'command':['docker','kill','--signal=TERM',ci['Id']],'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0
 for _ in range(900):
  entries=[x.split()for x in subprocess.check_output(['docker','ps','-a','--no-trunc','--format','{{.ID}} {{.Names}}'],text=True).splitlines()];gone=all(x[0]!=ci['Id']and x[1]!=NAME for x in entries)
  log=(OUT/'server.log').read_text()
  if gone and re.search(r'gpu-run: .* done in \d+s \(exit 0\)',log):break
  time.sleep(.1)
 else:raise RuntimeError('Normal exit/removal unconfirmed; root diagnoses, no force/retry/health overlap')
 save('normal-stop.json',{'signal':'TERM to server -> QUIT','full_cid':ci['Id'],'name':NAME,'full_cid_absent':True,'name_absent':True,'lease_wrapper_exit':0,'entries':entries})
 records={}
 for name,cmd in [('card',['bin/gpu-run','bin/xpu-health','--img','b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78','--timeout','400']),('collective',['bin/gpu-run','bin/xpu-collective-health','--timeout','400'])]:
  with(OUT/(name+'-health.log')).open('x')as f:r=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
  records[name]={'command':cmd,'exit':r.returncode}
  # Preserve partial health if a check fails, without a repeat.
  (OUT/'post-health.json').write_text(json.dumps(records,indent=2)+'\n');assert r.returncode==0
 check_source();assert sha(I['binary'])==I['binary_sha256']
 tools=json.loads((OUT/'profiler-identity.json').read_text());assert all(sha(f)==v['sha256']for f,v in tools.items())
 save('closure.json',{'status':'NORMAL_TEARDOWN_POSTHEALTH_PASS','collection_status':json.loads((OUT/'collection.json').read_text())['status']if(OUT/'collection.json').exists()else'COLLECTION_FAILED_INSPECT_RAW','post_health':records,'source_pins_rechecked':True,'trace_JSON_validated':False,'serving_or_speed_qualified':False});check_source()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['start','collect','stop']);args=ap.parse_args();globals()[args.phase]()
