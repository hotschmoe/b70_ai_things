#!/usr/bin/env python3
"""Explicit finite API-probe phases; no automatic retry or extra generation."""
import argparse,hashlib,json,re,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
sys.path.insert(0,str(REPO/'strata/flash-next'))
from bench_serving import get
BASE='http://127.0.0.1:18085'
PROF='/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace'
def read(p):return json.loads(p.read_text())
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(out,name,v):
 with(out/name).open('x',encoding='ascii')as f:json.dump(v,f,indent=2,ensure_ascii=True);f.write('\n')
def source():
 for line in(P/'expected-source.sha256').read_text().splitlines():
  h,n=line.split(maxsplit=1);assert sha(P/n)==h,n
 inputs=read(P/'serving-inputs.json')
 assert inputs['arms']['0']==inputs['arms']['1']and inputs['arms']['0']['sha256']=='e3ca8ac25c1f1e6042792906be3e7faa55e14a1cddaeece176f15c52f6b071fd'
 for a in ['0','1']:
  assert sha(P/f'mode{a}/config.json')==inputs['generated'][a]['config_sha256']
  assert sha(P/f'mode{a}/trace-engine.sh')==inputs['generated'][a]['wrapper_sha256']
 for row in inputs['saved_request_files']:assert sha(row['path'])==row['sha256']
 return inputs
def subprocess_phase(arm,script,*args):
 # Existing phase scripts retain source config, identity, lease and cleanup gates.
 subprocess.run([sys.executable,str(P/'harness-prep'/script),arm,*args],check=True)
def idle(out,label):
 for _ in range(300):
  slots=get(BASE,'/slots')
  if len(slots)==2 and all(s['is_processing']is False for s in slots):
   save(out,label,{'slots':slots,'epoch_ns':time.time_ns(),'scope':'HTTP slots quiescent; not standalone hardware completion proof'});return
  time.sleep(.1)
 raise RuntimeError('Slots did not become idle; no extra request/retry')
def control(out,inputs,arm,action):
 ci=read(out/'container-ready.json');cmd=['docker','exec',ci['Id'],PROF,'--'+action,inputs['sessions'][arm]]
 try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 except subprocess.TimeoutExpired as e:
  decode=lambda x:x.decode('utf8','replace')if isinstance(x,bytes)else x
  save(out,'unitrace-'+action+'.json',{'command':cmd,'exit':None,'timeout_s':30,'stdout':decode(e.stdout),'stderr':decode(e.stderr)})
  raise RuntimeError('Profiler control timed out; preserve attempt; normal cleanup still required')from e
 save(out,'unitrace-'+action+'.json',{'command':cmd,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'epoch_ns':time.time_ns()})
 assert r.returncode==0,'Profiler control failed; no profile request retry'
def counters(out,offset):
 raw=(out/'engine.log').read_bytes();assert len(raw)>=offset
 segment=raw[offset:];assert segment.endswith(b'\n')
 assert hashlib.sha256(raw[:offset]).hexdigest()==read(out/'collection-start.json')['engine_prefix_sha256']
 assert read(out/'coding/profile/window-start.json')['byte_offset']==offset,'HOLD engine writes between collection and client offsets'
 text=segment.decode('utf8');pair=read(out/'coding/profile/pair.json')
 windows=re.findall(r'^strata batch: (\d+) windows, avg ([0-9.]+) rows,',text,re.M)
 pairs=re.findall(r'^strata batch pair head: (\d+) windows drafted as a pair, (\d+) deferred but solo(?: .*|)$',text,re.M)
 assert len(windows)==len(pairs)==1,'HOLD unexpected/missing/duplicate native summary population'
 assert [[*x]for x in windows]==pair['native_window_rows']
 assert int(windows[0][0])==pair['native_windows']and int(pairs[0][0])==pair['paired']and int(pairs[0][1])==pair['solo']
 assert pair['paired']>0
 assert not re.search(r'ERR batch MTP|DEVICE_LOST|capture.*(?:failed|produced no graph)',text,re.I)
 return raw,segment,{'windows':pair['native_windows'],'paired':pair['paired'],'solo':pair['solo'],'rounded_rows':pair['native_window_rows'],'logged_new_captures':pair['logged_capture_layouts']}
def start(arm,out,inputs):subprocess_phase(arm,'start_arm.py')
def warm(arm,out,inputs):
 assert(out/'health-ready.json').exists()
 save(out,'warm-start.json',{'epoch_ns':time.time_ns(),'profiler':'injected/start-paused; no resume issued','output_per_stream':256})
 assert not(out/'unitrace-resume.json').exists()
 subprocess_phase(arm,'client.py','warm');idle(out,'warm-idle.json')
 save(out,'warm-closure.json',{'status':'PASS_ONE_PAUSED_107_256_C2_PAIR','pair':read(out/'coding/warm/pair.json'),'all_replay_keys_observed':False})
def collect(arm,out,inputs):
 assert read(out/'warm-closure.json')['status']=='PASS_ONE_PAUSED_107_256_C2_PAIR'
 idle(out,'before-resume-idle.json')
 raw=(out/'engine.log').read_bytes();offset=len(raw)
 save(out,'collection-start.json',{'epoch_ns':time.time_ns(),'engine_byte_offset':offset,'engine_prefix_sha256':hashlib.sha256(raw).hexdigest(),'scope':'ONE fresh ordered107/64 C2 pair plus idle/native-counter closure; all admission/capture/solo/tail retained','speed_qualified':False})
 failure=None;errors=[]
 try:
  control(out,inputs,arm,'resume')
  with(out/'profile-client.log').open('x')as f:
   r=subprocess.run([sys.executable,str(P/'harness-prep/client.py'),arm,'profile'],stdout=f,stderr=subprocess.STDOUT)
  save(out,'profile-client-exit.json',{'exit':r.returncode});assert r.returncode==0,'Profile pair failed; no extra request'
  idle(out,'profile-idle.json')
  first,segment,counts=counters(out,offset);time.sleep(.1)
  second,segment2,counts2=counters(out,offset)
  assert first==second and segment==segment2 and counts==counts2,'Native byte-offset closure changed; HOLD without retry'
  (out/'profile-engine-complete.log').write_bytes(segment)
  save(out,'native-counter-closure.json',{'status':'PASS_IDLE_AND_EXACT_NATIVE_SUMMARY_CLOSURE','start_byte_offset':offset,'end_byte_offset':len(second),'segment_bytes':len(segment),'segment_sha256':hashlib.sha256(segment).hexdigest(),'counts':counts,'unobserved':'per-call max_steps and all replay keys; physical hardware idle not independently observed'})
 except Exception as e:
  failure=e;save(out,'collection-failure.json',{'type':type(e).__name__,'detail':str(e),'no_retry_or_extra_request':True})
 finally:
  # Even a failed resume/client/pause still attempts the ONE session stop.
  for action in ['pause','stop']:
   try:control(out,inputs,arm,action)
   except Exception as e:errors.append({'action':action,'type':type(e).__name__,'detail':str(e)})
  save(out,'collection.json',{'status':'HOLD_COLLECTION_FAILED'if failure or errors else'PASS_PAIR_NATIVE_CLOSURE_AND_CONTROLS_SUMMARY_UNPARSED','control_errors':errors,'epoch_ns':time.time_ns(),'summary_closure':'Deferred until normal QUIT and owned process/container gone; NO canary/flush request','instrumented_speed_qualified':False})
 if failure:raise failure
 assert not errors,'Control failure; normal cleanup required, no extra request'
def stop(arm,out,inputs):
 # Available after failed start/warm/collect; S95 cleanup handles absent readiness.
 subprocess_phase(arm,'stop_and_health.py')
 assert read(out/'normal-stop.json')['status']=='PASS_NORMAL_TEARDOWN'
 assert sha(inputs['arms'][arm]['exe'])==inputs['arms'][arm]['sha256']
 for path,row in read(P/'compiled-tool-pins.json').items():assert sha(path)==row['sha256']
 # Read metadata only. ONE bounded small host-summary parse belongs to analyze.py.
 summaries=list((out/'trace').glob('timing*.csv'))
 assert len(summaries)==1 and set((out/'trace').iterdir())==set(summaries),'HOLD missing/ambiguous/unexpected host-summary output'
 summary=summaries[0]
 assert 0<summary.stat().st_size<=2_000_000,'HOLD unexpected host-summary size; no broader capture'
 save(out,'closure.json',{'status':'PASS_NORMAL_QUIT_FULL_HEALTH_SUMMARY_UNPARSED','summary':str(summary),'summary_bytes':summary.stat().st_size,'source_tool_binary_pins_rechecked':True,'speed_qualified':False})

def analyze(arm,out,inputs):
 subprocess.run([sys.executable,str(P/'analyze.py'),arm],check=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['0','1']);ap.add_argument('phase',choices=['start','warm','collect','stop','analyze']);a=ap.parse_args();inputs=source();out=P/f'mode{a.arm}'
 globals()[a.phase](a.arm,out,inputs)
