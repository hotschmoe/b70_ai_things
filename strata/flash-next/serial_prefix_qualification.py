#!/usr/bin/env python3
"""Separate leased serial native-GEN fidelity/cache fixture. Parent owns GPU health.
prepare is CPU-only. run is an explicit GPU action and checks inherited leases.
"""
import argparse,array,hashlib,json,math,os,queue,re,shlex,signal,subprocess,sys,threading,time
from pathlib import Path
import c1_serve_controller as c1
from run_source_upload_oracle_full import verify_model_identity
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return c1.sha(p)
def read(p):return c1.read(p)
def require(ok,msg):c1.require(ok,msg)
def write(p,j):c1.write(p,j)
EXTRAS=['0013-sycl-bounded-full-model-fidelity-observer.patch','0014-sycl-prefix-work-telemetry.patch','0015-sycl-explicit-full-state-prefix-pin.patch','0016-sycl-explicit-all-stage-fresh-request.patch']

def engine_identity(root,baseline):
 r=read(root/'receipt.json');require(r['build_rc']==0 and r['external_source_unchanged'] and r['plan_snapshot_unchanged'],'Combined source build incomplete')
 require(r['image']==c1.BASE_IMAGE,'Runtime image differs')
 b=read(baseline['engine_receipt']);old={Path(p['path']).name:p['sha256'] for p in b['patches']};new={Path(p['path']).name:p['sha256'] for p in r['patches']}
 require(set(new)==set(old)|set(EXTRAS) and all(new[n]==v for n,v in old.items()),'Combined generation has unexpected source patches')
 for n in EXTRAS:require(new[n]==sha(ROOT/'strata/flash-next/patches'/n),'New source patch changed')
 for rel,h in r['patched_source_sha256'].items():require(sha(root/'source'/rel)==h,'Source changed: '+rel)
 require(sha(root/'build/strata')==r['binary_sha256'][str(root/'build/strata')],'Executable changed')
 return r

def prepare(a):
 m=c1.validate_prepared(a.prepared.resolve());model_identity=verify_model_identity(a.model_identity,read(ROOT/'strata/flash-next/model-lock.json'),[Path(x['path']) for x in m['model_shards']]);root=a.engine_root.resolve();engine_identity(root,m)
 a.output.mkdir(parents=True,exist_ok=False)
 args=list(read(a.prepared/'server-config.json')['args'])
 for flag,value in [('--prompt-cache','3'),('--prompt-cache-root','1'),('--prompt-cache-every','1000000000')]:
  if flag in args:args[args.index(flag)+1]=value
  else:args += [flag,value]
 for flag in ['--batch','--suffix-draft','--lookup-chain','--adapt-swaps']:
  require(args[args.index(flag)+1]=='0','Serial fixed math route differs '+flag)
 require('--mtp' not in args and '--pipeline-windows' not in args and '--adapt-every' in args and args[args.index('--adapt-every')+1]=='0' and '--no-prefill-borrow' in args,'Unqualified adaptive/MTP/pipeline/prefill borrowing route')
 code=ROOT/'strata/flash-next/serial_prefix_token_fixture.py'
 command=['docker','run','--rm','--network','none','--user','1000:1000','-v',str(root/'source')+':/src:ro','-v',m['pack']+':/pack:ro','-v',str(code)+':/fixture.py:ro',m['runtime']['image'],'exec /opt/b70-c1-python/bin/python /fixture.py']
 tokens=json.loads(subprocess.check_output(command,text=True,timeout=90));require(tokens['vocab']==248320,'Exact vocabulary differs')
 manifest=read(a.prepared/'artifact-identity.json');require(all(tokens['tokenizer_sha256'][name]==h for name,h in manifest['tokenizer_files'].items()),'Tokenizer export identity differs')
 for flag,value in [('--turn-token',str(tokens['turn_token']))]:
  if flag in args:args[args.index(flag)+1]=value
  else:args += [flag,value]
 # Full transport is native protocol; no API served-ID or concurrency qualification.
 j={'schema':1,'status':'CPU prepared; no GPU launch','prepared':str(a.prepared.resolve()),'prepared_sha256':sha(a.prepared/'prepared.json'),'engine_root':str(root),'engine_receipt_sha256':sha(root/'receipt.json'),'model_identity':model_identity,'identity_parser_sha256':sha(ROOT/'strata/flash-next/run_source_upload_oracle_full.py'),'args':args,'tokens':tokens,'cards':m['cards'],'image':m['runtime']['image'],'baseline_source_upload':m['upload_lifecycle'],'new_engine_source_upload_qualified':False,'identity_chain':'baseline complete390 source gate plus unchanged base patches and separately listed observer/prefix increments; not a new390 upload claim','controller_sha256':sha(Path(__file__)),'token_fixture_sha256':sha(code),'max_new':64,'primary_model_name':'hotschmoe-dd','research_alias':m['alias']+'-prefix-diag0013-16-nativeGEN-v1','transport':'direct native GEN; no /v1/models or API/concurrent claim','correctness':{'full_logits':'bitwise finite equality on matched armed fresh/hit requests','diagnostic_flag0':'exact output IDs + LP20 only; no raw flag0 logits available','activations0_1':'full248320 first logits + exact IDs; residuals only activations1'},'armed_requests_per_process':6,'groups':['basic_off','basic_logits','basic_layers','root','pin','turn','parked','eviction','cancel']}
 write(a.output/'plan.json',j);print(json.dumps({'plan':str(a.output/'plan.json'),'GPU_executed':False}))

class Protocol:
 def __init__(self,command,directory,ready_timeout=900):
  self.directory=directory;self.stdout=[];self.stderr=[];self.q=queue.Queue();self.log=directory/'engine.stderr.log';self.err=self.log.open('w')
  self.p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,start_new_session=True)
  self.threads=[]
  def pump(stream,rows,events):
   for line in stream:
    line=line.rstrip('\n');rows.append(line)
    if events:self.q.put((time.monotonic(),line))
    else:self.err.write(line+'\n');self.err.flush()
   if events:self.q.put((time.monotonic(),None))
  for stream,rows,events in [(self.p.stdout,self.stdout,True),(self.p.stderr,self.stderr,False)]:
   t=threading.Thread(target=pump,args=(stream,rows,events),daemon=True);t.start();self.threads.append(t)
  deadline=time.monotonic()+ready_timeout
  while True:
   _,line=self.q.get(timeout=max(.001,deadline-time.monotonic()))
   require(line is not None,'Engine ended before readiness')
   if line.startswith('ERR'):raise ValueError(line)
   if line.startswith('READY'):self.ready=line;break
 def send(self,line):self.p.stdin.write(line+'\n');self.p.stdin.flush()
 def request(self,label,ids,fresh=0,pin=0,cancel=None,max_new=64):
  start=time.monotonic();err_start=len(self.stderr);rows=[];times=[];stopped=False
  command='GEN %d fresh=%d pin=%d ckpt=1 temperature=0 logprobs=20 '%(max_new,fresh,pin)+' '.join(map(str,ids));self.send(command)
  deadline=start+900
  while True:
   stamp,line=self.q.get(timeout=max(.001,deadline-time.monotonic()));require(line is not None,'Unexpected engine EOF')
   rows.append(line);times.append(stamp-start)
   if line.startswith('ERR'):raise ValueError(line)
   if cancel and not stopped and ((cancel=='prefill' and line.startswith('PP ')) or (cancel=='decode' and line.startswith('T '))):self.send('STOP');stopped=True
   if line.startswith('DONE '):break
  # DONE follows all diagnostic writes; reader thread may lag without GPU effect.
  for _ in range(200 if (self.directory/'ARM').exists() else 0):
   if any('"event":"finish"' in x for x in self.stderr[err_start:]):break
   time.sleep(.005)
  raw={'label':label,'ids':ids,'fresh':fresh,'pin':pin,'cancel_requested':cancel,'stop_sent':stopped,'command':command,'stdout':rows,'stdout_seconds':times,'stderr':self.stderr[err_start:],'output_ids':[int(x.split()[1]) for x in rows if x.startswith('T ')],'LP':[x for x in rows if x.startswith('LP ')],'done':rows[-1],'ready':self.ready}
  write(self.directory/(label+'.json'),raw);return raw
 def close(self):
  if self.p.poll() is None:self.send('QUIT')
  try:rc=self.p.wait(timeout=90)
  except subprocess.TimeoutExpired:self.p.terminate();self.p.wait(timeout=20);rc=self.p.returncode
  for t in self.threads:t.join(timeout=5)
  self.err.close();(self.directory/'engine.stdout.log').write_text('\n'.join(self.stdout)+'\n',encoding='ascii')
  return rc

def extract(request,capture_dir,require_vectors=True):
 led=[json.loads(x[len('PREFIX_DIAG '):]) for x in request['stderr'] if x.startswith('PREFIX_DIAG ')]
 finish=[x for x in led if x.get('event')=='finish'];require(len(finish)==1,'Missing/ambiguous completed work ledger')
 f=finish[0];require(f['prompt_tokens']==len(request['ids']),'Prompt length differs')
 selection=[x for x in led if x.get('event')=='selection'];require(len(selection)==1,'Missing/ambiguous selection')
 spans=[x for x in led if x.get('event')=='evaluated_span'];require(sum(x['prompt_rows'] for x in spans)==f['evaluated_prompt_rows'],'Ledger total differs from spans')
 if not f['cancelled']:
  require(f['actual_reused']+f['evaluated_prompt_rows']==len(request['ids']),'Actual evaluated/reused prompt coverage incomplete')
  covered=[i for x in spans for i in range(max(0,x['lo']),min(len(request['ids']),x['hi']))]
  require(covered==list(range(f['actual_reused'],len(request['ids']))),'Prompt spans overlap or leave gaps')
  require(f['stage_prompt_rows']==f['evaluated_prompt_rows']*selection[0]['stages'],'Stage work multiplier differs')
 if request['fresh']:require(f['actual_reused']==0,'fresh did not reset work')
 vectors=[]
 begins=[x for x in request['stderr'] if x.startswith('SFD request ')]
 if require_vectors:
  require(len(begins)==1,'Missing/ambiguous fidelity request identity')
  fields=dict(re.findall(r'(\w+)=([^ ]+)',begins[0]));require([int(x) for x in fields['ids'].split(',')]==request['ids'],'Captured source input IDs differ')
 for line in request['stderr']:
  if not line.startswith('SFD vector '):continue
  row=dict(re.findall(r'(\w+)=([^ ]+)',line));rel=Path(row['file']).name;p=capture_dir/rel
  require(p.is_file() and not p.is_symlink() and p.stat().st_size==int(row['bytes']),'Raw vector missing/truncated')
  require(not require_vectors or row['request']==fields['request'] and row['pid']==fields['pid'],'Vector request ownership differs')
  floats=array.array('f');floats.frombytes(p.read_bytes())
  if sys.byteorder!='little':floats.byteswap()
  require(all(math.isfinite(v) for v in floats),'Captured nonfinite vector')
  row['path']=str(p);row['sha256']=sha(p);vectors.append(row)
 require(len({v['path'] for v in vectors})==len(vectors),'Raw capture file reused under another descriptor')
 logits=[x for x in vectors if x['phase']=='first_logits_before_sampler'];
 if require_vectors and not f['cancelled']:require(len(logits)==1 and int(logits[0]['bytes'])==248320*4 and int(logits[0]['pos'])==len(request['ids'])-1 and int(logits[0]['token'])==request['ids'][-1],'Full first-token logit coverage/position differs')
 if logits and not f['cancelled']:
  values=array.array('f');values.frombytes(Path(logits[0]['path']).read_bytes())
  if sys.byteorder!='little':values.byteswap()
  require(request['output_ids'] and 0<=request['output_ids'][0]<248320 and values[request['output_ids'][0]]==max(values),'Captured first logits do not predict emitted greedy token')
 residuals=[x for x in vectors if x['phase']=='first_window_residual'];
 if residuals:
  require(sorted(int(x['layer']) for x in residuals)==list(range(48)) and all(int(x['bytes'])==10240*4 for x in residuals),'Complete first-window layer residual coverage differs')
  descriptors={int(x['stage']):(int(x['lb']),int(x['le'])) for x in residuals}
  require(sorted(descriptors)==list(range(selection[0]['stages'])),'Captured stage count differs')
  cursor=0
  for stage,(lo,hi) in sorted(descriptors.items()):require(lo==cursor and lo<hi<=48,'Stage layer bounds overlap or leave gap');cursor=hi
  require(cursor==48 and all(descriptors[int(x['stage'])]==(int(x['lb']),int(x['le'])) and int(x['lb'])<=int(x['layer'])<int(x['le']) and int(x['pos'])==len(request['ids'])-1 and int(x['token'])==request['ids'][-1] for x in residuals),'Layer residual identity/ownership differs')
  if logits:require(int(logits[0]['stage'])==len(descriptors)-1 and int(logits[0]['le'])==48,'Logit head stage differs')
 return {'ledger':f,'selection':selection[0],'spans':spans,'vectors':vectors,'logits':logits,'residuals':residuals}

def compare_vectors(a,b):
 ra=Path(a['path']).read_bytes();rb=Path(b['path']).read_bytes();require(len(ra)==len(rb),'Vector shape differs')
 x=array.array('f');x.frombytes(ra);y=array.array('f');y.frombytes(rb)
 if sys.byteorder!='little':x.byteswap();y.byteswap()
 require(all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y),'Nonfinite logits/residual')
 d=[abs(float(u)-float(v)) for u,v in zip(x,y)]
 return {'bitwise_equal':ra==rb,'floats':len(x),'max_abs':max(d,default=0),'rmse':math.sqrt(sum(t*t for t in d)/len(d)) if d else 0,'left_sha256':a['sha256'],'right_sha256':b['sha256']}

class ComparisonError(ValueError):
 def __init__(self,message,evidence):super().__init__(message);self.evidence=evidence

def compare_requests(left,right,lmeta,rmeta,full=True):
 result={'output_ids_equal':left['output_ids']==right['output_ids'],'LP20_equal':left['LP']==right['LP'],'finish_equal':left['done'].split()[5]==right['done'].split()[5],'natural_completion':left['done'].split()[5]=='stop' and right['done'].split()[5]=='stop'}
 flags=list(result.values())
 if full:
  result['first_logits']=compare_vectors(lmeta['logits'][0],rmeta['logits'][0]);flags.append(result['first_logits']['bitwise_equal'])
  if lmeta['residuals'] and rmeta['residuals']:
   x={int(v['layer']):v for v in lmeta['residuals']};y={int(v['layer']):v for v in rmeta['residuals']};result['first_window_residuals']={str(i):compare_vectors(x[i],y[i]) for i in range(48)}
   differing=[i for i in range(48) if not result['first_window_residuals'][str(i)]['bitwise_equal']]
   result['first_divergent_residual_layer']=differing[0] if differing else None;flags.append(not differing)
 result['passed']=all(flags)
 if not result['passed']:raise ComparisonError('Strict output/logit/residual/natural-completion comparison failed',result)
 return result

def option(args,flag,value):
 if flag in args:args[args.index(flag)+1]=str(value)
 else:args += [flag,str(value)]

def decode_output(pack,ids):
 # Exact exported token strings; GPT-2 byte inverse used by the source tokenizer.
 p=Path(pack)/'tokenizer';vocab=read(p/'vocab.json');reverse={i:t for t,i in vocab.items()};types=read(p/'token_type.json')
 bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];extra=0
 for b in range(256):
  if b not in bs:bs.append(b);cs.append(256+extra);extra+=1
 inverse={chr(c):b for b,c in zip(bs,cs)}
 selected=[reverse[i] for i in ids if types[i] not in (3,4)]
 return bytes(inverse[c] for token in selected for c in token).decode('utf-8',errors='strict')

def run_process(plan,m,output,name,diag,activations,group):
 directory=output/name;directory.mkdir();capture=directory/'captures';capture.mkdir();arm=directory/'ARM'
 args=list(plan['args']);parking=group in {'eviction','parked'}
 option(args,'--conversation-cache-mib',512 if parking else 0);option(args,'--conversation-cache-slots',1 if parking else 0)
 env=dict(read(Path(plan['prepared'])/'server-config.json')['env']);require('STRATA_CKPT_REREAD' not in env and 'STRATA_STATE_HASH' not in env,'Diagnostic reset/hash toggles are forbidden')
 env.update({'STRATA_FIDELITY_DIAG':str(diag),'STRATA_FIDELITY_DIAG_ACTIVATIONS':str(activations),'STRATA_FIDELITY_DIAG_ARM':'/results/ARM','STRATA_FIDELITY_DIAG_DIR':'/results/captures','STRATA_PREFIX_DIAG':'1','STRATA_PREFIX_DIAG_ARM':'/results/ARM','STRATA_ARTIFACT_IDENTITY_SHA256':sha(output/'plan.snapshot.json')})
 engine=Path(plan['engine_root']);container='b70-prefix-'+str(os.getpid())+'-'+name
 command=['docker','run','-i','--name',container,'--label','b70.prefix.plan='+sha(output/'plan.snapshot.json'),'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',m['pack']+':/pack:ro','-v',str(ROOT/read(ROOT/'strata/flash-next/model-lock.json')['destination'])+':/model:ro','-v',str(directory)+':/results']
 for k,v in sorted(env.items()):command += ['-e',k+'='+v]
 command += [plan['image'],'cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+args)]
 write(directory/'command.json',command);p=None;rows=[];error=None
 try:
  p=Protocol(command,directory)
  # Identical useful warmups execute before ARM. They do not consume six-request capture quota.
  for key in ['A','B','C']:p.request('warm-'+key,plan['tokens']['cases'][key],fresh=1,max_new=1)
  arm.write_text('ARM after READY and fixed warmups\n',encoding='ascii')
  cases=plan['tokens']['cases'];requests=[]
  def req(label,key,fresh=0,pin=0,cancel=None):
   r=p.request(label,cases[key],fresh,pin,cancel);r['decoded_output_without_special_tokens']=decode_output(m['pack'],r['output_ids']);write(directory/(label+'.json'),r);meta=extract(r,capture,diag==1);requests.append({'raw':r,'meta':meta});write(directory/'requests.json',requests);return r,meta
  if group=='basic':
   for key in ['A','B','C']:req(key,key,fresh=1)
  elif group in {'root','pin','turn'}:
   ka,kb=('pin_A','pin_B') if group=='pin' else ('turn_A','turn_B') if group=='turn' else ('A','B')
   pin=plan['tokens']['pin'] if group=='pin' else 0
   ref,rm=req('reference',kb,1,pin)
   req('establish',ka,1,pin)
   hit,hm=req('hit',kb,0,pin)
   expected=plan['tokens']['pin' if group=='pin' else 'turn_shared' if group=='turn' else 'root']
   require(hm['ledger']['actual_reused']==expected,'Actual selected prefix differs from targeted '+group+' boundary')
   rows.append(compare_requests(ref,hit,rm,hm))
   # Repeat the matched triple in the same warmed process; no latency comparison.
   ref2,rm2=req('reference2',kb,1,pin);req('establish2',ka,1,pin);hit2,hm2=req('hit2',kb,0,pin)
   require(hm2['ledger']['actual_reused']==expected,'Second matched hit boundary differs')
   rows.append(compare_requests(ref2,hit2,rm2,hm2));rows.append(compare_requests(ref,ref2,rm,rm2))
  elif group=='parked':
   expected=plan['tokens']['turn_boundaries']['B']
   for suffix in ['', '2']:
    ref,rm=req('reference'+suffix,'B',1)
    req('independent-context'+suffix,'A',1)
    hit,hm=req('restored'+suffix,'B')
    require(hm['selection']['route']=='ram_snapshot' and hm['ledger']['actual_reused']==expected,'Complete parked snapshot restore route/boundary was not exercised')
    rows.append(compare_requests(ref,hit,rm,hm))
  elif group=='eviction':
   ref,rm=req('reference','evict_B',1)
   for key in ['evict_A','evict_C','evict_D']:req('other-'+key,key,1)
   after,am=req('after-eviction','evict_B')
   require(am['ledger']['evictions']>rm['ledger']['evictions'] and am['ledger']['actual_reused']==0,'No actual bounded eviction/cold reconstruction evidence')
   final,fm=req('fresh-final','evict_B',1);rows += [compare_requests(ref,after,rm,am),compare_requests(after,final,am,fm)]
  elif group=='cancel':
   ref,rm=req('reference','pin_B',1)
   cancelled,cm=req('cancel-prefill','pin_B',1,cancel='prefill')
   require(cancelled['stop_sent'] and cm['ledger']['cancelled'],'Prefill STOP did not cancel')
   after,am=req('after-prefill','pin_B');fresh,fm=req('fresh-after-prefill','pin_B',1)
   rows += [compare_requests(ref,after,rm,am),compare_requests(after,fresh,am,fm)]
   cancelled,cm=req('cancel-decode','A',1,cancel='decode')
   require(cancelled['stop_sent'] and cm['ledger']['cancelled'],'Decode STOP did not cancel')
   final,fm=req('after-decode-fresh','pin_B',1);rows.append(compare_requests(ref,final,rm,fm))
  require(len(requests)<=6,'Capture quota exceeded')
  write(directory/'requests.json',requests)
 except Exception as e:
  error=str(e)
  if isinstance(e,ComparisonError):rows.append(e.evidence)
 finally:
  rc=None
  try:rc=p.close() if p else None
  except Exception as e:error=(error+'; ' if error else '')+'protocol close: '+str(e)
  try:
   state=c1.inspected(container)['State'];require(not state['Running'],'Engine remains live')
   require(state['ExitCode']==0 and not state.get('OOMKilled'),'Engine did not exit normally')
   subprocess.run(['docker','rm',container],check=True,capture_output=True,text=True)
   removed=not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+container+'$'],text=True).strip()
   require(removed,'Container remains registered')
  except Exception as e:
   # Owned cleanup only; retain failure. No broad process kill or recovery from this controller.
   subprocess.run(['docker','stop','--time','30',container],capture_output=True,text=True,timeout=50)
   subprocess.run(['docker','rm',container],capture_output=True,text=True,timeout=30)
   error=(error+'; ' if error else '')+'teardown: '+str(e);state=None;removed=False
  result={'group':group,'name':name,'diagnostic':diag,'activations':activations,'passed':error is None and rc==0,'error':error,'terminal':state,'removed':removed,'engine_rc':rc,'comparisons':rows,'transport':'native GEN','capture_bound':6,'cancelled_work_counter_scope':'completed spans only'}
  write(directory/'result.json',result)
 return result

def run(a):
 plan=read(a.plan);require(plan['controller_sha256']==sha(Path(__file__)),'Controller changed after preparation')
 require(plan['token_fixture_sha256']==sha(ROOT/'strata/flash-next/serial_prefix_token_fixture.py'),'Token fixture code changed')
 m=c1.validate_prepared(Path(plan['prepared']));require(sha(Path(plan['prepared'])/'prepared.json')==plan['prepared_sha256'],'Original prepared source changed')
 engine_identity(Path(plan['engine_root']),m);require(sha(Path(plan['engine_root'])/'receipt.json')==plan['engine_receipt_sha256'],'Combined source receipt changed')
 require(sha(ROOT/'strata/flash-next/run_source_upload_oracle_full.py')==plan['identity_parser_sha256'],'Full identity verifier changed')
 verify_model_identity(Path(plan['model_identity']['path']),read(ROOT/'strata/flash-next/model-lock.json'),[Path(x['path']) for x in m['model_shards']])
 c1.leased(plan['cards']);h=read(a.pre_health);require(h.get('passed') and set(plan['cards'])<=set(h['cards']) and 0<=time.time()-h['finished_epoch']<=300,'Fresh parent-owned pre-health missing')
 require(h.get('files'),'Parent pre-health has no source command/log evidence')
 for f in h['files']:require(sha(f['path'])==f['sha256'],'Parent health evidence changed')
 if len(plan['cards'])>1:
  require(a.one_card_receipt,'Matched combined one-card diagnostic required before pair')
  r=read(a.one_card_receipt);require(r.get('passed') and r.get('numerical_and_teardown_passed') and r.get('post_health_passed') and r.get('group') in {'basic','all'} and len(r['cards'])==1 and r['engine_receipt_sha256']==plan['engine_receipt_sha256'],'Matched one-card evidence absent')
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.snapshot.json',plan)
 stopping=[False]
 def stop(sig,frame):
  if not stopping[0]:stopping[0]=True;raise InterruptedError('Parent stop signal '+str(sig))
 for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stop)
 groups=['basic','root','pin','turn','parked','eviction','cancel'] if a.group=='all' else [a.group]
 results=[];error=None
 if a.group not in {'all','basic'}:
  require(a.basic_receipt,'Basic flag-off/on equivalence must pass before prefix fixture')
  b=read(a.basic_receipt);require(b.get('passed') and b.get('numerical_and_teardown_passed') and b.get('post_health_passed') and b.get('group') in {'basic','all'} and b['engine_receipt_sha256']==plan['engine_receipt_sha256'] and b['cards']==plan['cards'],'Matched basic receipt missing')
 try:
  for group in groups:
   verify_model_identity(Path(plan['model_identity']['path']),read(ROOT/'strata/flash-next/model-lock.json'),[Path(x['path']) for x in m['model_shards']])
   if group=='basic':
    local=[]
    for name,diag,acts in [('basic_off',0,0),('basic_logits',1,0),('basic_layers',1,1)]:
     r=run_process(plan,m,a.output,name,diag,acts,'basic');results.append(r);require(r['passed'],'Basic process failed: '+str(r['error']));local.append(read(a.output/name/'requests.json'))
    pairs=[]
    for off,logits,layers in zip(*local):
     pairs.append({'label':off['raw']['label'],'flag0_on':compare_requests(off['raw'],logits['raw'],off['meta'],logits['meta'],False),'activations0_1':compare_requests(logits['raw'],layers['raw'],logits['meta'],layers['meta'])})
    write(a.output/'basic-equivalence.json',pairs)
   else:
    r=run_process(plan,m,a.output,group,1,1,group);results.append(r);require(r['passed'],group+' failed: '+str(r['error']))
   verify_model_identity(Path(plan['model_identity']['path']),read(ROOT/'strata/flash-next/model-lock.json'),[Path(x['path']) for x in m['model_shards']])
 except Exception as e:
  error=str(e)
  if isinstance(e,ComparisonError):write(a.output/'failed-comparison.json',e.evidence)
 report={'schema':1,'status':'awaiting parent post-health' if error is None else 'FAILED strict serial gate','passed':False,'numerical_and_teardown_passed':error is None and all(r['passed'] for r in results),'post_health_passed':False,'error':error,'cards':plan['cards'],'group':a.group,'engine_receipt_sha256':plan['engine_receipt_sha256'],'plan_sha256':sha(a.output/'plan.snapshot.json'),'pre_health_sha256':sha(a.pre_health),'finished_epoch':time.time(),'results':results,'transport':'native GEN; no API/concurrency/shelf/clean latency claim','flag0_full_logits_observed':False,'complete_prefix_qualified':False,'new_engine390_source_upload_qualified':False}
 write(a.output/'report.json',report);print(json.dumps({'report':str(a.output/'report.json'),'numerical_and_teardown_passed':report['numerical_and_teardown_passed']}))
 if error:raise SystemExit(1)

def finalize(a):
 p=a.output/'report.json';r=read(p);h=read(a.post_health);require(h.get('passed') and set(r['cards'])<=set(h['cards']) and h.get('finished_epoch',0)>=r['finished_epoch'],'Parent post-health does not cover completed fixture/devices')
 require(h.get('files'),'Parent post-health has no source command/log evidence')
 for row in h['files']:require(sha(row['path'])==row['sha256'],'Post-health artifact changed')
 require(r['numerical_and_teardown_passed'],'Strict numerical/teardown gate failed')
 plan=read(a.output/'plan.snapshot.json');m=read(Path(plan['prepared'])/'prepared.json')
 identity=verify_model_identity(a.model_identity,read(ROOT/'strata/flash-next/model-lock.json'),[Path(x['path']) for x in m['model_shards']])
 require(read(a.model_identity)['started']>=int(r['finished_epoch']),'Full post-run model hash scan must start after fixture')
 r['post_model_identity']=identity
 r.update(passed=True,post_health_passed=True,post_health_sha256=sha(a.post_health),status='PASSED bounded serial native fixture; no concurrency/API qualification');r['bounded_serial_prefix_passed']=r['group']=='all';r['complete_prefix_qualified']=False;write(p,r)

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
 q=sub.add_parser('prepare');q.add_argument('--prepared',type=Path,required=True);q.add_argument('--engine-root',type=Path,required=True);q.add_argument('--model-identity',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('run');q.add_argument('--plan',type=Path,required=True);q.add_argument('--pre-health',type=Path,required=True);q.add_argument('--one-card-receipt',type=Path);q.add_argument('--basic-receipt',type=Path);q.add_argument('--group',choices=['basic','root','pin','turn','parked','eviction','cancel','all'],default='basic');q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('finalize');q.add_argument('--output',type=Path,required=True);q.add_argument('--post-health',type=Path,required=True);q.add_argument('--model-identity',type=Path,required=True)
 a=p.parse_args();{'prepare':prepare,'run':run,'finalize':finalize}[a.mode](a)
if __name__=='__main__':main()
