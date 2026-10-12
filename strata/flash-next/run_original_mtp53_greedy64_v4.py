"""Root-run two-process original MTP greedy64 diagnostic. No READY/client protocol."""
import argparse,hashlib,json,os,re,shlex,signal,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/vm_8tb/github/b70_ai_things');HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-publisher-mtp53-greedy64-runner-proposal-v4.json'
FIXTURE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-normal-cli-v1-run-v1/tokenizer-fixtures.json')
BASE=Path('/tmp/b70-original-mtp53-main-v5-card0-state-preserving-blueprint.json')
IDENTITY=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source52-clean-serial-cache-latency-v1-run-v1/pre-full4.json')
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def stat5(p):s=Path(p).stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def read(p):return json.loads(Path(p).read_bytes())
def controls(text,prompt_count):
 tokens=[];windows=[];summary=[]
 for line in text.splitlines():
  if line.startswith(('MTP53_GREEDY_TOKEN ','MTP53_GREEDY_WINDOW ','MTP53_GREEDY_PASS ')):
   v=dict(x.split('=',1)for x in line.split()[1:])
   if line.startswith('MTP53_GREEDY_TOKEN '):tokens.append(v)
   elif line.startswith('MTP53_GREEDY_WINDOW '):windows.append(v)
   else:summary.append(v)
 if len(summary)!=1 or not tokens or any('READY 'in line for line in text.splitlines()):raise ValueError('Unique greedy completion/EOF required, no READY')
 s=summary[0];ids=[int(v['id'])for v in tokens];eos=[i for i,v in enumerate(ids)if v in (248044,248046)]
 if [int(v['ordinal'])for v in tokens]!=list(range(1,len(tokens)+1)) or any(v<0 or v>=248320 for v in ids):raise ValueError('Actual token sequence differs')
 if any(int(v['eos'])!=int(int(v['id'])in (248044,248046))for v in tokens):raise ValueError('EOS marker differs')
 if eos and eos!=[len(ids)-1]:raise ValueError('EOS must actually terminate')
 if not eos and len(ids)!=64:raise ValueError('No padded/short non-EOS completion')
 if int(s['generated'])!=len(ids) or int(s['useful'])!=len(ids)-len(eos) or int(s['natural_eos'])!=int(bool(eos)) or int(s['budget'])!=64 or int(s['prompt_tokens'])!=prompt_count:raise ValueError('Actual completion counters differ')
 position=prompt_count;emitted=1;accepted=rejected=tail=0
 for n,w in enumerate(windows,1):
  remain=64-emitted;keep=int(w['committed']);a=int(w['accepted']);mode=s['mode']
  if int(w['window'])!=n or int(w['remaining_before'])!=remain or a not in (0,1) or keep not in (1,2) or keep>remain or int(w['input'])!=ids[emitted-1]:raise ValueError('Actual committed-window roster differs')
  if emitted+keep>len(ids):raise ValueError('Commit exceeds observed emitted picks')
  first_eos=ids[emitted]in (248044,248046)
  want=2 if mode=='original_mtp_on'and a and not first_eos else 1
  if keep!=want or (mode=='target_off'and a) or (remain==1 and a):raise ValueError('Actual acceptance/EOS/budget commit differs')
  if mode=='original_mtp_on':
   if remain==1:tail+=1
   elif a:accepted+=1
   else:rejected+=1
   if a and int(w['guess'])!=ids[emitted]:raise ValueError('Accepted draft differs from target pick')
  position+=keep;emitted+=keep
  if int(w['position'])!=position:raise ValueError('Commit position differs')
 if emitted!=len(ids) or int(s['windows'])!=len(windows) or int(s['accepted_windows'])!=accepted or int(s['rejected_windows'])!=rejected or int(s['single_tail_windows'])!=tail or int(s['draft_cells'])!=(position if s['mode']=='original_mtp_on'else 0):raise ValueError('Final actual state/count differs')
 times=[int(v['host_monotonic_ns'])for v in tokens]
 if any(b<a for a,b in zip(times,times[1:])):raise ValueError('Token timestamps reversed')
 return {'ids':ids,'natural_eos':bool(eos),'useful':len(ids)-len(eos),'tokens':tokens,'windows':windows,'summary':s,'token_gap_ns':[b-a for a,b in zip(times,times[1:])]}
def absence_classification(stderr,name):
 # Docker errors are case-variable; absence must still name THIS original container exactly.
 if not isinstance(stderr,bytes) or len(stderr)>4096:return 'unknown'
 try:text=stderr.decode('ascii').strip().lower()
 except UnicodeDecodeError:return 'unknown'
 expected=name.lower()
 for prefix in ('error: ','error response from daemon: ',''):
  for kind in ('object','container'):
   if text==prefix+'no such '+kind+': '+expected:return 'original_absent'
 return 'unknown'
def phase_rows(text,actual):
 import math,collections
 rows=[];scope=[];counters=[]
 for line in text.splitlines():
  if line.startswith(('MTP53_HOST_PHASE ','MTP53_HOST_PHASE_SCOPE ','MTP53_VERIFIER_DECODE_COUNTERS ')):
   v=dict(x.split('=',1)for x in line.split()[1:])
   if line.startswith('MTP53_HOST_PHASE '):rows.append(v)
   elif line.startswith('MTP53_HOST_PHASE_SCOPE '):scope.append(v)
   else:counters.append(v)
 if len(scope)!=1 or len(counters)!=1 or len(rows)>2048 or int(scope[0]['rows'])!=len(rows) or [int(v['ordinal'])for v in rows]!=list(range(1,len(rows)+1)):raise ValueError('Actual bounded phase row roster differs')
 if any(scope[0][k]!='0'for k in ('additional_GPU_waits','additional_GPU_queries','timers_nested')):raise ValueError('Phase scope differs')
 counts=collections.Counter();totals=collections.defaultdict(float)
 for v in rows:
  ms=float(v['wall_ms'])
  if not math.isfinite(ms) or ms<0:raise ValueError('Actual host phase time invalid')
  key=(v['scope'],v['phase']);counts[key]+=1;totals[key]+=ms
 prompt=int(actual['summary']['prompt_tokens']);windows=len(actual['windows']);committed=sum(int(v['committed'])for v in actual['windows']);on=actual['summary']['mode']=='original_mtp_on'
 expected={('serial_prompt','target_run'):prompt,('serial_prompt','target_commit_wait'):prompt,('decode','target_run'):windows,('decode','target_commit_wait'):windows,('decode','final_existing_drain'):1}
 if on:
  for phase in ('draft_prepare_submit','draft_record_wait','draft_settle_wait'):expected[('serial_prompt',phase)]=prompt;expected[('decode',phase)]=committed
  for phase in ('own_head_read_argmax','draft_prior_drain'):expected[('decode',phase)]=windows
 if {k:v for k,v in counts.items()if v}!={k:v for k,v in expected.items()if v}:raise ValueError('Phase counts differ from actual target/draft windows')
 for key in ('wait_ms','pool_ms','host_ms','commit_ms'):
  value=float(counters[0][key])
  if not math.isfinite(value) or value<0:raise ValueError('Actual verifier counter invalid')
 return {'rows':rows,'scope':scope[0],'verifier_counters':counters[0],'host_phase_totals_ms':{a+'/'+b:v for (a,b),v in totals.items()},'causal_GPU_suboperation_attribution':False,'resolved_residency_equal':False}
def main(a):
 if not os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0'):raise ValueError('Card0 lease FD8 required')
 plan=read(PLAN)
 for p,h in plan['bindings'].items():
  if sha(p)!=h:raise ValueError('Runner exact source/metadata changed '+p)
 recipe=read(a.compile_root/'compile-recipe.json');receipt=a.compile_root/'output/receipt.json';elf=a.compile_root/'output/original-mtp53-greedy64'
 if sha(receipt)!=a.compile_receipt_sha256 or sha(elf)!=a.elf_sha256:raise ValueError('Genuine compile receipt/ELF changed')
 r=read(receipt)
 if r.get('passed')is not True or r.get('devices_exposed')is not False or r.get('model_payload_read')is not False or r['recipe_sha256']!=sha(a.compile_root/'compile-recipe.json') or recipe['main_proposal_sha256']!=sha(HERE/'original-publisher-mtp53-greedy64-proposal-v3.json'):raise ValueError('Actual compile provenance differs')
 cfg=read(BASE);fixture=read(FIXTURE)['fixtures'][0];ids=fixture['ids']
 if len(ids)!=54 or any(type(v)is not int or not 0<=v<248320 for v in ids):raise ValueError('Authentic54 water-cycle tokenizer fixture required')
 out=a.output
 if not out.is_absolute()or out.resolve()!=out or out.exists():raise ValueError('New canonical output required')
 out.mkdir();write(out/'tokenizer-fixture.json',fixture);write(out/'compile-recipe.snapshot.json',recipe)
 report={'started_epoch':time.time(),'passed':False,'errors':[],'source_plan_sha256':sha(PLAN),'producer_pid':os.getpid(),'compile_receipt_sha256':sha(receipt),'elf_sha256':sha(elf),'arm_results':[],'common_full5_scan_count':0,'full64_equivalence':False,'draft_numerical_qualified':False,'speed_qualified':False,'population_stability_qualified':False,'prefill_speed_qualified':False}
 stop=[]
 def signal_stop(n,f):stop.append(n)
 for n in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(n,signal_stop)
 def command(argv,prefix,timeout=180):
  write(out/(prefix+'.command.json'),argv)
  with(out/(prefix+'.stdout')).open('wb')as so,(out/(prefix+'.stderr')).open('wb')as se:p=subprocess.run(argv,stdout=so,stderr=se,timeout=timeout)
  if p.returncode:raise ValueError(prefix+' rc '+str(p.returncode))
 def identity(phase):
  manifest=read(IDENTITY);rows=[]
  if manifest['passed']is not True or len(manifest['rows'])!=4:raise ValueError('Original locked four-shard identity required')
  mtp=Path('/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-flash-next/unsloth-ud-q4-k-xl/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf')
  for old in manifest['rows']+[{'path':str(mtp),'expected_sha256':'8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231'}]:
   p=Path(old['path']);before=stat5(p);h=sha(p);after=stat5(p);rows.append({'path':str(p),'before':before,'after':after,'sha256':h,'expected_sha256':old['expected_sha256'],'passed':before==after and h==old['expected_sha256']})
  write(out/(phase+'-full5.json'),rows);report['common_full5_scan_count']+=1
  if not all(v['passed']for v in rows):raise ValueError('Original full5 identity differs')
 def arm(mode):
  d=out/('OFF'if mode==0 else'ON');d.mkdir();started=time.time();name='b70-mtp53-greedy-'+str(os.getpid())+'-'+str(mode);label='b70.mtp53.greedy';owner=str(os.getpid())+'-'+str(mode);proc=None;errors=[];terminal=None;removed=False
  env=cfg['native_env'].copy();env.update({'STRATA_ONE_TOKEN_COMMIT':'0','STRATA_VERIFY_MIRROR_STAGE_T1':'0','STRATA_ORIGINAL_MTP53':'1','STRATA_ORIGINAL_MTP53_GREEDY':str(mode),'STRATA_ORIGINAL_MTP53_PHASE_TIMING':'1'if a.phase_timing else'0','ZE_AFFINITY_MASK':'0'})
  native=cfg['native_argv'].copy();native[0]='/diagnostic/original-mtp53-greedy64';native+=['--tokens',','.join(map(str,ids)),'--max-new','64'];mounts=cfg['read_only_mounts'].copy()
  for src,dst in list(mounts.items()):
   if dst=='/diagnostic':del mounts[src]
  mounts[str(a.compile_root/'output')]='/diagnostic'
  argv=['docker','run','--name',name,'--label',label+'='+owner,'--network','none','--cpus','16','--memory','105g','--memory-swap','105g','--user',str(os.getuid())+':'+str(os.getgid()),'--device','/dev/dri/card0','--device','/dev/dri/renderD128','--workdir',cfg['workdir']]
  for gid in sorted({os.stat('/dev/dri/card0').st_gid,os.stat('/dev/dri/renderD128').st_gid}):argv+=['--group-add',str(gid)]
  for src,dst in mounts.items():argv+=['-v',src+':'+dst+':ro']
  argv+=['-v',str(d)+':/results:rw']
  for k,v in env.items():argv+=['-e',k+'='+v]
  argv+=['--entrypoint','/bin/bash',cfg['image'],'-lc','if [ "${SETVARS_COMPLETED:-0}" != 1 ]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; exec '+shlex.join(native)]
  write(d/'native.command.json',argv);write(d/'recipe.json',{'native_argv':native,'native_env':env,'mounts':mounts,'fixture_sha256':sha(FIXTURE),'mode':mode})
  def inspect():
   r=subprocess.run(['docker','inspect',name],capture_output=True,timeout=30)
   if r.returncode:
    if absence_classification(r.stderr,name)=='original_absent':return None
    raise ValueError('Docker inspection unknown '+r.stderr.decode(errors='replace')[:500])
   row=json.loads(r.stdout)[0]
   if row['Config']['Labels'].get(label)!=owner or row['Config']['Image']!=cfg['image']:raise ValueError('Original container identity differs')
   return row
  try:
   command([str(ROOT/'bin/xpu-health'),'--card','0','--img',HEALTH],d.name+'-pre-health')
   with(d/'native.stdout').open('wb')as so,(d/'native.stderr').open('wb')as se:
    proc=subprocess.Popen(argv,stdout=so,stderr=se,start_new_session=True);write(d/'client.json',{'pid':proc.pid,'started_epoch':time.time()});deadline=time.monotonic()+1800
    while proc.poll()is None:
     size=(d/'native.stdout').stat().st_size+(d/'native.stderr').stat().st_size
     if stop or time.monotonic()>deadline or size>4<<20:
      errors.append('Controlled interruption/timeout/output bound');row=inspect()
      if row and row['State']['Running']:subprocess.run(['docker','stop','--time','20',row['Id']],capture_output=True,timeout=40)
      break
     time.sleep(.5)
    if not errors and proc.returncode!=0:raise ValueError('Native rc '+str(proc.returncode))
  except BaseException as e:errors.append(type(e).__name__+': '+str(e))
  finally:
   # Retain the lease and original process until exact terminal+absence+EOF, including ambiguous create failures.
   while True:
    try:
     row=inspect()
     if row and row['State']['Running']:
      errors.append('Owned forced cleanup');subprocess.run(['docker','stop','--time','20',row['Id']],capture_output=True,timeout=40);continue
     if proc and proc.poll()is None:time.sleep(.5);continue
     if row:
      terminal=row;write(d/'terminal.json',row);subprocess.run(['docker','rm',row['Id']],check=True,capture_output=True,timeout=30);continue
     removed=True
     if proc:proc.wait();report_client={'pid':proc.pid,'return_code':proc.returncode,'terminal_epoch':time.time(),'original_process_EOF':True};write(d/'client-terminal.json',report_client)
     break
    except BaseException as e:
     message='Retirement '+type(e).__name__+': '+str(e)
     if message not in errors and len(errors)<64:errors.append(message)
     write(d/'retirement-unresolved.json',{'passed':False,'ownership_retained':True,'original_name':name,'original_label':owner,'error':message[:4096],'epoch':time.time()});time.sleep(1)
   try:command([str(ROOT/'bin/xpu-health'),'--card','0','--img',HEALTH],d.name+'-post-health')
   except BaseException as e:errors.append('Post-health '+str(e))
   try:
    command(['journalctl','-k','--since','@'+str(int(started)),'--no-pager'],d.name+'-kernel')
    if re.search(r'Fault response|Faulted Address|Engine reset|Timedout job|segfault|GPU HANG|CAT error',(out/(d.name+'-kernel.stdout')).read_text(),re.I):raise ValueError('Fresh kernel fault')
   except BaseException as e:errors.append('Kernel '+str(e))
  if not terminal or terminal['State']['ExitCode']!=0 or terminal['State']['OOMKilled'] or terminal['State']['Error'] or terminal['State']['Paused'] or terminal['State']['Restarting'] or terminal['State']['Dead'] or terminal['State']['Pid']!=0 or not removed:errors.append('Normal owned terminal/absence missing')
  parsed=None
  if not errors:
   if sum((d/n).stat().st_size for n in ('native.stdout','native.stderr'))>4<<20:errors.append('Final output quota exceeded')
   else:
    try:
     parsed=controls((d/'native.stdout').read_text(),54)
     if parsed['summary']['mode']!=('target_off'if mode==0 else'original_mtp_on'):raise ValueError('Actual mode differs')
     if a.phase_timing:parsed['host_phases']=phase_rows((d/'native.stdout').read_text(),parsed)
    except BaseException as e:errors.append('Parse '+str(e))
  result={'mode':mode,'started_epoch':started,'finished_epoch':time.time(),'errors':errors,'passed':not errors,'owned_removed':removed,'actual':parsed,'resolved_residency_lines':[l for l in (d/'native.stderr').read_text(errors='replace').splitlines()if re.search(r'expert cache|slots|resident|mirror|GiB free',l,re.I)],'raw_sha256':{n:sha(d/n)for n in ('native.stdout','native.stderr')}};write(d/'arm-report.json',result);report['arm_results'].append(result)
  if errors:raise ValueError('Arm'+str(mode)+' '+repr(errors))
 try:
  identity('pre');arm(0);arm(1)
  off,on=[v['actual']for v in report['arm_results']]
  if off['ids']!=on['ids']or off['natural_eos']!=on['natural_eos']:raise ValueError('Independent OFF/ON greedy sequence differs')
  report['greedy_IDs_and_EOS_equal']=True;report['full64_equivalence']=len(off['ids'])==64 and len(on['ids'])==64
 except BaseException as e:report['errors'].append(type(e).__name__+': '+str(e))
 finally:
  try:identity('post')
  except BaseException as e:report['errors'].append('Post identity '+str(e))
  for p,h in plan['bindings'].items():
   try:
    if sha(p)!=h:report['errors'].append('Closing source/metadata changed '+p)
   except BaseException as e:report['errors'].append('Closing source/metadata '+str(e))
  try:
   if sha(receipt)!=a.compile_receipt_sha256 or sha(elf)!=a.elf_sha256:report['errors'].append('Closing compile/ELF changed')
  except BaseException as e:report['errors'].append('Closing compile/ELF '+str(e))
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['full64_equivalence']=bool(report['passed'] and report.get('greedy_IDs_and_EOS_equal') and all(len(v['actual']['ids'])==64 for v in report['arm_results']));write(out/'report.json',report);(out/'controller.snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report,ensure_ascii=True))
 return 0 if report['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--compile-root',type=Path,required=True);p.add_argument('--compile-receipt-sha256',required=True);p.add_argument('--elf-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase-timing',action='store_true');raise SystemExit(main(p.parse_args()))
