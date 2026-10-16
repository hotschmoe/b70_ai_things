#!/usr/bin/env python3
"""Root-only phases: start (CPU lease), collect (quiet CPU), stop (quiet CPU).
The detached launch owns both GPU leases for the complete server lifetime.
No phase retries/restarts a process or launches another profile request.
"""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,sys,time,urllib.request
P=Path(__file__).resolve().parent;OUT=P/'run';REPO=Path('/mnt/vm_8tb/github/b70_ai_things');B=Path('/mnt/vm_8tb/b70/build/strata-ab')
sys.path.insert(0,str(REPO/'strata/flash-next'));from bench_serving import get,send
I=json.loads((P/'inputs.json').read_text());C=json.loads((P/'config.json').read_text());BASE='http://127.0.0.1:18085';NAME=I['container'];UNIT='strata-unprofiled-ring-s142'

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
 OUT.mkdir();check_source()
 assert sha(I['binary'])==I['binary_sha256']
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
  assert not e.get('LD_PRELOAD') and not any(k.startswith('UNITRACE_')for k in e)
  assert not any(k in e for k in ['SYCL_UR_TRACE','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'])
  assert 'STRATA_VERIFY_PROFILE'not in e and 'STRATA_PREFILL_TIMING'not in e and 'STRATA_PREFILL_SYNC'not in e and 'STRATA_VERIFY_EAGER'not in e and 'STRATA_VERIFY_TRACE'not in e
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

def collect():
 check_source();assert(OUT/'loaded.json').exists()and not(OUT/'collection-start.json').exists()
 for row in I['saved_request_files']:assert sha(row['path'])==row['sha256']
 save('collection-models.json',identity())
 save('collection-start.json',{'epoch_us':time.time_ns()/1000,'request_source_sha256':[x['sha256']for x in I['saved_request_files']],'capture':'No generation warmup, same S133binary/flags; no unitrace or URlogger injection','request':'One ordered107prompt/64output EACH coding pair','CPU_clock_only':True})
 with(OUT/'pair-client.log').open('x')as f:r=subprocess.run([sys.executable,str(P/'pair.py')],stdout=f,stderr=subprocess.STDOUT)
 save('pair-client-exit.json',{'exit':r.returncode});assert r.returncode==0,'Pair failed: no extra request'
 ring_snapshot()
 ring=json.loads((OUT/'ring-closure.json').read_text());ref=json.loads((P.parent/'current-mtp-profile-s136/run/ring-closure.json').read_text())
 assert ring['actual_records']==ref['actual_records']==820 and ring['window0_records']==ref['window0_records']==0
 save('collection.json',{'status':'ONE_UNPROFILED_PAIR_NUMERIC_NATIVE_WORK_RING_PASS','CPU_clock_only':True,'no_raw_trace_created_or_parsed':True,'no_unitrace_controls':True,'no_generation_flush':True,'serving_or_speed_qualified':False})


def stop():
 subprocess.run([sys.executable,str(P/'stop.py')],check=True)
 check_source();assert sha(I['binary'])==I['binary_sha256']
 save('closure.json',{'status':'NORMAL_TEARDOWN_POSTHEALTH_PASS','source_pins_rechecked':True,'trace_JSON_validated':False,'serving_or_speed_qualified':False})

def analyze():
 check_source();assert json.loads((OUT/'closure.json').read_text())['status']=='NORMAL_TEARDOWN_POSTHEALTH_PASS'
 analysis=OUT/'analysis';analysis.mkdir()
 cmd=[sys.executable,str(P/'cpu_markers.py'),str(OUT/'pair-engine-complete.log'),'--output',str(analysis/'cpu')]
 with(analysis/'cpu-parser.log').open('x')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 save('cpu-parser-exit.json',{'command':cmd,'exit':r.returncode});assert r.returncode==0
 old=json.loads((P.parent/'current-mtp-profile-s136/run/analysis/cpu/analysis.json').read_text());new=json.loads((analysis/'cpu/analysis.json').read_text())
 assert old['validation']['actual_records']==new['validation']['actual_records']==820 and new['validation']['status']=='COMPLETE_VALIDATED'
 save('CPU-comparison.json',{'status':'CPU_RING_DIAGNOSTIC_ONLY','instrumented_reference':'S136 with unitrace','candidate':'S142 without unitrace; same S133binary/flags/request','old_replay_windows':old['replay_pair_windows'],'new_replay_windows':new['replay_pair_windows'],'old_span_stats_ms':old['replay_same_window_span_stats_ms'],'new_span_stats_ms':new['replay_same_window_span_stats_ms'],'old_cross_window_stats_ms':old['replay_same_dump_cross_window_span_stats_ms'],'new_cross_window_stats_ms':new['replay_same_dump_cross_window_span_stats_ms'],'not_clean_serving_speed_qualification':True,'no_CPU_GPU_clock_join':True,'no_causal_GPUgap_or_idle_claim':True,'profiler_perturbation_candidate_not_single_API_attribution':True})


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['start','collect','stop','analyze']);args=ap.parse_args();globals()[args.phase]()
