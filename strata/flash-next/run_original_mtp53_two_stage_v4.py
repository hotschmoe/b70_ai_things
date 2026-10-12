"""Root-owned original MTP paired controls and real P2048/GEN512 diagnostic."""
import argparse,hashlib,json,os,re,shlex,signal,subprocess,time,math
from pathlib import Path
ROOT=Path('/mnt/vm_8tb/github/b70_ai_things');HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-publisher-mtp53-two-stage-runner-proposal-v4.json'
FIXTURE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/natural-p2048-tokenizer-v1/corpus.json')
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
def set_arg(argv,key,value):
 if argv.count(key)!=1:raise ValueError('Unique original option required '+key)
 result=argv.copy();result[result.index(key)+1]=value;return result

def native_options(cfg,mode,ids):
 if mode not in('controls','off','on')or len(ids)!=(1 if mode=='controls'else 2048)or(mode=='controls'and ids!=[248045]):raise ValueError('Exact controls or natural corpus required')
 native=cfg['native_argv'].copy();native[0]='/diagnostic/original-mtp53-two-stage'
 native=set_arg(native,'--prefill','512');native=set_arg(native,'--max-context','4096')
 if any(k in native for k in('--layer-split','--split-device','--tokens','--max-new','--mtp','--batch-mtp')):raise ValueError('Unexpected inherited topology/draft/input')
 native+=['--layer-split','24','--split-device','1','--trim-stage-weights','--tokens',','.join(map(str,ids)),'--max-new','64'if mode=='controls'else'512']
 required={'--spec':'2','--batch':'0','--prompt-cache':'0','--conversation-cache-mib':'0','--conversation-cache-slots':'0','--adapt-every':'0','--adapt-swaps':'0','--pipeline-windows':'0','--short-read':'0','--eos-ids':'248044,248046'}
 if any(native.count(k)!=1 or native[native.index(k)+1]!=v for k,v in required.items())or '--no-prefill-borrow'not in native or '--stop-eos'not in native:raise ValueError('Actual native pair source guard differs')
 return native

def records(text,kind):
 rows=[]
 for line in text.splitlines():
  if line.startswith(kind+' '):
   parts=[x.split('=',1)for x in line.split()[1:]]
   if any(len(x)!=2 for x in parts)or len({x[0]for x in parts})!=len(parts):raise ValueError('Duplicate/malformed native fields')
   rows.append(dict(parts))
 return rows

def residency(text):
 if text.count('strata generate: layer split: CUDA0 runs layers 0-23\n')!=1:raise ValueError('Actual primary24 constructor missing')
 primary=re.findall(r'strata generate: expert cache (\d+) slots,',text)
 secondary=re.findall(r'strata generate: layer split: CUDA1 runs layers 24-47, expert cache (\d+) slots .*slot 0 verified\n',text)
 mirrors=re.findall(r'stage mirror: layers(\d+)-(\d+) missing(\d+) mirrored(\d+) bytes(\d+) ',text)
 if len(primary)!=1 or len(secondary)!=1 or len(mirrors)!=2 or [(int(v[0]),int(v[1]))for v in mirrors]!=[(0,23),(24,47)]:raise ValueError('Actual two cache/map constructors required')
 values=[]
 for row,slots in zip(mirrors,map(int,[primary[0],secondary[0]])):
  lb,le,missing,mirrored,size=map(int,row)
  if not 0<slots<=12288 or missing!=mirrored or missing!=12288-slots or(size==0)!=(mirrored==0):raise ValueError('Actual expert residency dimensions differ')
  values.append({'lb':lb,'le':le,'slots':slots,'mirrored':mirrored,'mirror_bytes':size})
 if 'the prompt path allocates its own buffers'not in text or re.search(r'trying a smaller expert cache|trying a \d+-token chunk| - LOW:',text):raise ValueError('Original noBorrow/no fallback required')
 reserve='MTP53 PAIR reserve_device=1 context4096 original_tensor_bytes=2775258112 both_execution_arms_resident=1'
 if text.count(reserve)!=1:raise ValueError('Same original drafter residency before sizing required')
 return values

def control_rows(text):
 restored=records(text,'MTP53_PAIR_CONTROL_RESTORED');steps=records(text,'MTP53_PAIR_CONTROL_STEP')
 if len(restored)!=6 or len(steps)!=3 or records(text,'MTP53_PAIR_TOKEN')or records(text,'MTP53_PAIR_PASS'):raise ValueError('Exact control-only roster required')
 position=1
 for trial in range(3):
  group=restored[2*trial:2*trial+2]
  for stage,row in enumerate(group):
   expected={'trial':trial,'stage':stage,'device':stage,'lb':stage*24,'le':(stage+1)*24,'base_position':position,'touched_cells':position+2,'state_KV_exact':1}
   if {k:int(v)for k,v in row.items()}!=expected:raise ValueError('Exact actual two-stage restored fields differ')
  row={k:int(v)for k,v in steps[trial].items()};accepted=row.get('accepted');keep=row.get('committed')
  if row.get('trial')!=trial or accepted not in(0,1)or keep not in(1,2)or row.get('target_greedy_equal')!=1 or(trial==0 and accepted!=0)or(trial==1 and accepted!=1):raise ValueError('Forced reject/accept/own controls missing')
  position+=keep
  if row.get('position')!=position:raise ValueError('All-stage committed position differs')
 return {'restored':restored,'steps':steps,'both_stage_state_restored':True,'three_control_steps':True}

def natural_rows(text,mode):
 tokens=records(text,'MTP53_PAIR_TOKEN');windows=records(text,'MTP53_PAIR_WINDOW');summary=records(text,'MTP53_PAIR_PASS');chunks=records(text,'MTP53_PAIR_PREFILL')
 if len(summary)!=1 or not tokens or records(text,'MTP53_PAIR_CONTROL_STEP')or 'MTP53_HOST_PHASE 'in text:raise ValueError('Natural-only phaseOFF roster required')
 s=summary[0];ids=[int(r['id'])for r in tokens];eos=[n for n,v in enumerate(ids)if v in(248044,248046)]
 if s['mode']!=mode or [int(r['ordinal'])for r in tokens]!=list(range(1,len(ids)+1))or any(not 0<=v<248320 for v in ids)or(eos and eos!=[len(ids)-1])or(not eos and len(ids)!=512)or len(ids)>512:raise ValueError('Actual bounded natural output differs')
 if any(int(r['eos'])!=int(ids[n]in(248044,248046))for n,r in enumerate(tokens)):raise ValueError('Native EOS flags differ')
 times=[int(r['host_monotonic_ns'])for r in tokens]
 if any(v<=0 for v in times)or any(b<a for a,b in zip(times,times[1:])):raise ValueError('Native token chronology differs')
 seen=0
 for row in chunks:
  t=int(row['rows'])
  if not 1<=t<=512 or int(row['begin'])!=seen or int(row['end'])!=seen+t or int(row['target_batched'])!=1 or int(row['device'])!=1 or int(row['owned_draft_catchup'])!=int(mode=='on'):raise ValueError('Actual last-stage batched prompt coverage differs')
  seen+=t
 if seen!=2047:raise ValueError('All 2047 actual prompt rows required')
 position=2048;emitted=1;accepted=rejected=0
 for row in windows:
  remain=512-emitted;keep=int(row['committed']);a=int(row['accepted'])
  if a not in(0,1)or keep not in(1,2)or keep>remain or int(row['remaining'])!=remain or int(row['input'])!=ids[emitted-1]or emitted+keep>len(ids):raise ValueError('Actual verify/commit history differs')
  want=2 if mode=='on'and a and ids[emitted]not in(248044,248046)else 1
  if keep!=want or(mode=='off'and(a or int(row['guess'])!=-1))or(remain==1 and a)or(a and int(row['guess'])!=ids[emitted]):raise ValueError('Actual draft acceptance/budget/target pick differs')
  if mode=='on':accepted+=a;rejected+=1-a
  emitted+=keep;position+=keep
  if int(row['position'])!=position:raise ValueError('Actual all-stage commit position differs')
 expected={'prompt_tokens':2048,'generated':len(ids),'useful':len(ids)-len(eos),'natural_eos':int(bool(eos)),'accepted_windows':accepted,'rejected_windows':rejected,'draft_capacity':4096,'draft_cells':position if mode=='on'else 0,'resident_original_tensor_control_both_arms':1,'fullmath_qualified':0,'API_qualified':0,'speed_qualified':0}
 if emitted!=len(ids)or any(int(s[k])!=v for k,v in expected.items()):raise ValueError('Actual final prompt/output/state counters differ')
 for key in('prompt_wall_ms','decode_wall_ms'):
  if not math.isfinite(float(s[key]))or float(s[key])<=0:raise ValueError('Actual host duration invalid')
 return {'ids':ids,'natural_eos':bool(eos),'summary':s,'tokens':tokens,'windows':windows,'prefill':chunks,'token_gap_ns':[b-a for a,b in zip(times,times[1:])],'physical_device_timing_qualified':False}

def selection(case_index,repeat_count):
 if (case_index is not None and(type(case_index)is not int or case_index not in(0,1,2)))or type(repeat_count)is not int or repeat_count not in(1,2):raise ValueError('Explicit selected case and repeats required')
 return list(range(3))if case_index is None else[case_index],repeat_count

def fresh_reset_marker(text):
 expected='MTP53_PAIR_FRESH_RESET stages=2 devices=0,1 ranges=0:24,24:48 native52_public_reset=1 before_target_bind=1'
 rows=[line for line in text.splitlines()if line.startswith('MTP53_PAIR_FRESH_RESET')]
 if rows!=[expected]:raise ValueError('Unique successful actual V5 BOTH-target fresh reset required')
 return True

def main(a):
 for fd,card in((8,0),(9,1)):
  if not os.path.samefile('/proc/self/fd/'+str(fd),'/mnt/vm_8tb/b70/gpu.lock.'+str(card)):raise ValueError('Both original pair lease descriptors required')
 plan=read(PLAN)
 for p,h in plan['bindings'].items():
  if sha(p)!=h:raise ValueError('Runner exact source/metadata changed '+p)
 recipe=read(a.compile_root/'compile-recipe.json');receipt=a.compile_root/'output/receipt.json';elf=a.compile_root/'output/original-mtp53-two-stage'
 if sha(receipt)!=a.compile_receipt_sha256 or sha(elf)!=a.elf_sha256:raise ValueError('Genuine compile receipt/ELF changed')
 r=read(receipt)
 if r.get('passed')is not True or r.get('devices_exposed')is not False or r.get('model_payload_read')is not False or r['recipe_sha256']!=sha(a.compile_root/'compile-recipe.json') or recipe['two_stage_source_plan_sha256']!=sha(HERE/'original-publisher-mtp53-two-stage-proposal-v5.json') or recipe['compiler_image']!='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7':raise ValueError('Actual compile provenance differs')
 proposal=read(HERE/'original-publisher-mtp53-two-stage-proposal-v5.json')
 if sha(Path(proposal['actual52_SDK'])/'receipt.json')!=proposal['actual52_receipt_sha256']:raise ValueError('Actual SDK receipt changed')
 for source,dest in proposal['new_overlay_sources'].items():
  if sha(a.compile_root/'overlay'/dest)!=proposal['own_source_sha256'][source]:raise ValueError('Actual compiled overlay changed '+source)
 if sha(a.compile_root/'overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp')!=recipe['native_MAIN_two_stage_sha256']:raise ValueError('Actual paired MAIN source changed')
 cfg=read(BASE);corpus=read(FIXTURE);cases=corpus['cases'];ids=[248045]
 if len(cases)!=3 or any(len(v['ids'])!=2048 or any(type(x)is not int or not 0<=x<248320 for x in v['ids'])for v in cases):raise ValueError('Authentic three P2048 cases required')
 if a.phase=='natural':
  prior=read(a.controls_report);prior_root=a.controls_report.parent
  if sha(a.controls_report)!=a.controls_report_sha256 or prior.get('passed')is not True or prior.get('phase')!='controls' or prior.get('elf_sha256')!=a.elf_sha256 or prior.get('source_plan_sha256')!=sha(PLAN):raise ValueError('Exact successful same-source paired controls required')
  for name,want in prior['artifacts'].items():
   if sha(prior_root/name)!=want:raise ValueError('Original controls bytes changed '+name)
  if len(prior['arm_results'])!=1 or prior['arm_results'][0]['passed']is not True or prior['arm_results'][0]['owned_removed']is not True or prior.get('errors')!=[] or prior.get('common_full5_scan_count')!=2:raise ValueError('Normal controls retirement/identity missing')
  for phase in('pre','post'):
   rows=read(prior_root/(phase+'-full5.json'))
   if len(rows)!=5 or any(row.get('passed')is not True for row in rows):raise ValueError('Recorded complete controls identity failed')
  parsed=control_rows((prior_root/'controls/native.stdout').read_text());parsed['residency']=residency((prior_root/'controls/native.stderr').read_text());fresh_reset_marker((prior_root/'controls/native.stderr').read_text())
  if parsed!=prior['arm_results'][0]['actual']:raise ValueError('Original controls semantic proof changed')
 case_indices,repeats=selection(a.case_index,a.repeat_count)
 out=a.output
 if not out.is_absolute()or out.resolve()!=out or out.exists():raise ValueError('New canonical output required')
 out.mkdir();write(out/'source-plan.snapshot.json',plan);write(out/'tokenizer-fixture.json',corpus);write(out/'compile-recipe.snapshot.json',recipe)
 report={'selected_case_indices':case_indices,'repeats_per_arm':repeats,'whole_three_role_repeat_matrix_qualified':False,'first_case_screen_only':a.phase=='natural'and case_indices==[0]and repeats==1,'phase':a.phase,'started_epoch':time.time(),'passed':False,'errors':[],'source_plan_sha256':sha(PLAN),'producer_pid':os.getpid(),'compile_receipt_sha256':sha(receipt),'elf_sha256':sha(elf),'arm_results':[],'common_full5_scan_count':0,'API_qualified':False,'fullmath_qualified':False,'instrumented_native_CLI_diagnostic':True,'draft_numerical_qualified':False,'speed_qualified':False,'population_stability_qualified':False,'prefill_speed_qualified':False}
 stop=[]
 def signal_stop(n,f):stop.append(n)
 for n in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(n,signal_stop)
 def command(argv,prefix,timeout=180):
  write(out/(prefix+'.command.json'),argv)
  with(out/(prefix+'.stdout')).open('wb')as so,(out/(prefix+'.stderr')).open('wb')as se:p=subprocess.run(argv,stdout=so,stderr=se,timeout=timeout)
  if p.returncode:raise ValueError(prefix+' rc '+str(p.returncode))
 def health(label):
  command([str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],label+'-strict',210)
  command([str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'],label+'-compiled',240)
 def identity(phase):
  manifest=read(IDENTITY);rows=[]
  if manifest['passed']is not True or len(manifest['rows'])!=4:raise ValueError('Original locked four-shard identity required')
  mtp=Path('/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-flash-next/unsloth-ud-q4-k-xl/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf')
  for old in manifest['rows']+[{'path':str(mtp),'expected_sha256':'8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231'}]:
   p=Path(old['path']);before=stat5(p);h=sha(p);after=stat5(p);rows.append({'path':str(p),'before':before,'after':after,'sha256':h,'expected_sha256':old['expected_sha256'],'passed':before==after and h==old['expected_sha256']})
  write(out/(phase+'-full5.json'),rows);report['common_full5_scan_count']+=1
  if not all(v['passed']for v in rows):raise ValueError('Original full5 identity differs')
 def arm(mode,case_index=0,repeat=0):
  nonlocal ids
  ids=[248045]if mode=='controls'else cases[case_index]['ids']
  d=out/('controls'if mode=='controls'else mode+'-'+str(case_index)+'-'+str(repeat));d.mkdir();started=time.time();name='b70-mtp53-pair-'+str(os.getpid())+'-'+str(mode)+'-'+str(case_index)+'-'+str(repeat);label='b70.mtp53.greedy';owner=str(os.getpid())+'-'+str(mode)+'-'+str(case_index)+'-'+str(repeat);proc=None;errors=[];terminal=None;removed=False
  env=cfg['native_env'].copy();env.update({'STRATA_ONE_TOKEN_COMMIT':'0','STRATA_VERIFY_MIRROR_STAGE_T1':'0','STRATA_ORIGINAL_MTP53':'1','STRATA_ORIGINAL_MTP53_PAIR_MODE':mode,'STRATA_ORIGINAL_MTP53_PHASE_TIMING':'0','ZE_AFFINITY_MASK':'0,1'})
  native=native_options(cfg,mode,ids);mounts=cfg['read_only_mounts'].copy()
  for src,dst in list(mounts.items()):
   if dst=='/diagnostic':del mounts[src]
  mounts[str(a.compile_root/'output')]='/diagnostic'
  argv=['docker','run','--name',name,'--label',label+'='+owner,'--network','none','--cpus','16','--memory','105g','--memory-swap','105g','--user',str(os.getuid())+':'+str(os.getgid()),'--device','/dev/dri','--workdir',cfg['workdir']]
  for gid in sorted({os.stat('/dev/dri/'+n).st_gid for n in('card0','card1','renderD128','renderD129')}):argv+=['--group-add',str(gid)]
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
   if row['Config']['Labels'].get(label)!=owner or row['Config']['Image']!=cfg['image']or row['Config']['Entrypoint']!=['/bin/bash']or row['Config']['Cmd']!=argv[-2:]or type(row['RestartCount'])is not int or row['RestartCount']!=0:raise ValueError('Original container identity/command differs')
   actual_env=row['Config']['Env']
   if any(actual_env.count(k+'='+v)!=1 for k,v in env.items()):raise ValueError('Actual native environment differs')
   expected_binds=[src+':'+dst+':ro'for src,dst in mounts.items()]+[str(d)+':/results:rw']
   if sorted(row['HostConfig']['Binds'])!=sorted(expected_binds):raise ValueError('Actual complete bind roster differs')
   return row
  (d/'native.stdout').touch();(d/'native.stderr').touch()
  try:
   health(d.name+'-pre-health')
   if stop:raise ValueError('Signal latched before launch')
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
     try:write(d/'retirement-unresolved.json',{'passed':False,'ownership_retained':True,'original_name':name,'original_label':owner,'error':message[:4096],'epoch':time.time()})
     except BaseException:pass
     time.sleep(1)
   try:health(d.name+'-post-health')
   except BaseException as e:errors.append('Post-health '+str(e))
   try:
    command(['journalctl','-k','--since','@'+str(int(started)),'--no-pager'],d.name+'-kernel')
    if re.search(r'Fault response|Faulted Address|Engine reset|Timedout job|segfault|GPU HANG|CAT error',(out/(d.name+'-kernel.stdout')).read_text(),re.I):raise ValueError('Fresh kernel fault')
   except BaseException as e:errors.append('Kernel '+str(e))
  if proc is None or proc.returncode!=0:errors.append('Original foreground process normal EOF missing')
  if not terminal or terminal['State']['ExitCode']!=0 or terminal['State']['OOMKilled'] or terminal['State']['Error'] or terminal['State']['Paused'] or terminal['State']['Restarting'] or terminal['State']['Dead'] or terminal['State']['Pid']!=0 or not removed:errors.append('Normal owned terminal/absence missing')
  parsed=None
  if not errors:
   if sum((d/n).stat().st_size for n in ('native.stdout','native.stderr'))>4<<20:errors.append('Final output quota exceeded')
   else:
    try:
     parsed=control_rows((d/'native.stdout').read_text())if mode=='controls'else natural_rows((d/'native.stdout').read_text(),mode)
     fresh_reset_marker((d/'native.stderr').read_text())
     parsed['residency']=residency((d/'native.stderr').read_text())
    except BaseException as e:errors.append('Parse '+str(e))
  result={'mode':mode,'case_index':case_index,'repeat':repeat,'started_epoch':started,'finished_epoch':time.time(),'errors':errors,'passed':not errors,'owned_removed':removed,'actual':parsed,'resolved_residency_lines':[l for l in (d/'native.stderr').read_text(errors='replace').splitlines()if re.search(r'expert cache|slots|resident|mirror|GiB free',l,re.I)],'raw_sha256':{n:sha(d/n)for n in ('native.stdout','native.stderr')}};write(d/'arm-report.json',result);report['arm_results'].append(result)
  if errors:raise ValueError('Arm'+str(mode)+' '+repr(errors))
 try:
  identity('pre')
  if a.phase=='controls':arm('controls')
  else:
   for mode in('off','on'):
    for case_index in case_indices:
     for repeat in range(repeats):arm(mode,case_index,repeat)
   rows=report['arm_results']
   for case_index in case_indices:
    group=[v['actual']for v in rows if v['case_index']==case_index]
    if any((v['ids'],v['natural_eos'])!=(group[0]['ids'],group[0]['natural_eos'])for v in group):raise ValueError('Actual OFF/ON/repeat token history or finish differs')
   if any(v['actual']['residency']!=rows[0]['actual']['residency']for v in rows):raise ValueError('Actual stage residency/map dimensions differ')
   report['greedy_IDs_and_EOS_equal']=True
   report['whole_three_role_repeat_matrix_qualified']=case_indices==[0,1,2]and repeats==2
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
  if stop:report['errors'].append('Signals latched '+repr(stop))
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['whole_three_role_repeat_matrix_qualified']=bool(report['passed']and report['whole_three_role_repeat_matrix_qualified']);report['full512_equivalence']=bool(report['passed'] and report.get('greedy_IDs_and_EOS_equal') and all(len(v['actual']['ids'])==512 for v in report['arm_results']));(out/'controller.snapshot.py').write_bytes(Path(__file__).read_bytes());report['artifacts']={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name not in('report.json',)};write(out/'report.json',report);print(json.dumps(report,ensure_ascii=True))
 return 0 if report['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--compile-root',type=Path,required=True);p.add_argument('--compile-receipt-sha256',required=True);p.add_argument('--elf-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['controls','natural'],required=True);p.add_argument('--case-index',type=int,choices=[0,1,2]);p.add_argument('--repeat-count',type=int,choices=[1,2],default=2);p.add_argument('--controls-report',type=Path);p.add_argument('--controls-report-sha256');raise SystemExit(main(p.parse_args()))
