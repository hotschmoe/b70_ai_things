"""Root-owned native four-slot target/original comparison; HTTP and restart gates stay open."""
import argparse,hashlib,json,os,queue,re,shlex,signal,subprocess,time,threading
from pathlib import Path
from batch_numerical_protocol_v2 import Roster
from native_source41_cli_protocol_v4 import Protocol as ProtectedProtocol
from run_original_mtp53_shifted_v2 import exact_binds,absent,stat5,MTP,MTP_SHA,RUNTIME,HEALTH
import prepare_original_mtp53_serve_runtime_v2 as intake
import source58_model_sdk_binding_v1 as sdk_binding
import source_page_watchdog_v3 as pages
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-mtp53-serve-qualification-source-plan-v2.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for v in iter(lambda:f.read(8<<20),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def need(v,s):
 if not v:raise ValueError(s)
class NativeProtocol(ProtectedProtocol):
 is_protocol=staticmethod(lambda line:line.startswith(('READY','INFO ','ERR','FATAL','RESUME ','REUSED ','PP ','T ','LP ','DONE ','BT ','BADM ','BDONE ','strata batch: stale/unidentified','MTP53_SERVE_REUSED','ORIGINAL_')))
def request_command(rid,slot,ids,fresh):
 return 'BGEN %d 64 fresh=%d pin=%d ckpt=1 temperature=0 rid=%d '%(slot,fresh,len(ids)-1,rid)+' '.join(map(str,ids))
def output_view(row):
 ids=row['generated'];done=row['done'].split();reason=done[3]if done[0]=='BDONE'else('stop'if ids[-1]in(248044,248046)else'length')
 need(ids and len(ids)<=64 and (reason!='length'or len(ids)==64),'Complete actual native output required')
 eos=[i for i,t in enumerate(ids)if t in(248044,248046)];need(not eos or eos==[len(ids)-1],'Natural EOS must terminate')
 return {'input_ids':row['ids'],'output_ids':ids,'finish':reason,'slotgen':row['slotgen'],'admission_done':row['admission_done'],'done':row['done']}
def native_command(base,runtime,sdk,d,mode):
 name='b70-mtp53-serve-'+str(os.getpid())+'-'+str(mode);owner=str(os.getpid())+'-'+str(mode);label='b70.mtp53.serve.qualifier'
 env=dict(runtime['native_env']);env['STRATA_ORIGINAL_MTP53_SERVING']=str(mode);env.update(STRATA_CRITICAL_PATH_TRACE='0',STRATA_BATCH_FIDELITY_DIAG='0',STRATA_FIDELITY_DIAG='0',STRATA_LAYER0_Q8_DIAG='0',STRATA_PREFIX30='0',STRATA_PLE_INPUT33='0',SYCL_UR_TRACE='0',STRATA_SLOT_OWNER_TRACE='1',STRATA_MIRROR_OWNER_TRACE='1')
 argv=list(runtime['native_argv']);mounts=dict(base['read_only_mounts'])
 for src,dest in list(mounts.items()):
  if dest in('/diagnostic','/src','/build','/original-mtp','/original-serve','/original.gguf'):del mounts[src]
 mounts[str(sdk/'source')]='/src';mounts[str(sdk/'build')]='/build';mounts.update(runtime['additional_readonly_mounts'])
 need(base['image']==RUNTIME,'Exact current native runtime image required')
 command=['docker','run','-i','--name',name,'--label',label+'='+owner,'--network','none','--cpus','16','--memory','105g','--memory-swap','105g','--user',str(os.getuid())+':'+str(os.getgid()),'--workdir',base['workdir']]
 devices=['/dev/dri/card0','/dev/dri/renderD128','/dev/dri/card1','/dev/dri/renderD129'];groups=sorted({str(os.stat(p).st_gid)for p in devices})
 for p in devices:command+=['--device',p]
 for g in groups:command+=['--group-add',g]
 binds=[src+':'+dest+':ro'for src,dest in mounts.items()]+[str(d)+':/results:rw']
 for bind in binds:command+=['-v',bind]
 for k,v in env.items():command+=['-e',k+'='+v]
 command+=['--entrypoint','/bin/bash',RUNTIME,'-lc','if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; exec '+shlex.join(argv)]
 return command,argv,env,mounts,binds,devices,groups
def diagnostic_identity(a,model_manifest):
 prior=Path(a.diagnostic_prior_root);need(a.diagnostic_prior_report_sha256 and prior.is_absolute()and prior.resolve()==prior and sha(prior/'report.json')==a.diagnostic_prior_report_sha256,'Authentic completed prior report required')
 report=read(prior/'report.json');need(report['finished_epoch']>=report['started_epoch']and report['actual_compile_receipt_sha256']==a.build_receipt_sha256 and report['actual_ELF_sha256']==a.elf_sha256,'Prior same native source/build experiment required')
 need(report['source_plan_sha256']==sha(HERE/'original-mtp53-serve-qualification-source-plan-v1.json'),'Explicit historical failedV1 source namespace required')
 for name in ('pre-full5.json','post-full5.json'):
  need(report['artifacts'][name]==sha(prior/name),'Authentic prior completed full5 receipt required')
 before=read(prior/'pre-full5.json');after=read(prior/'post-full5.json');want=model_manifest['rows']+[{'path':str(MTP),'expected_sha256':MTP_SHA}]
 need(len(before)==len(after)==len(want)==5,'Complete original publisher roster required')
 for old,new,expected in zip(before,after,want):
  need(old['passed']is True and new['passed']is True and old['path']==new['path']==expected['path']and old['sha256']==old['expected_sha256']==new['sha256']==new['expected_sha256']==expected['expected_sha256']and old['before']==old['after']==new['before']==new['after']==stat5(Path(expected['path'])),'Prior publisher SHA/unchanged current stat5 required')
 return {'prior_root':str(prior),'prior_report_sha256':a.diagnostic_prior_report_sha256,'prior_experiment_passed':report['passed'],'prior_identity_receipts_passed':True,'prior_finished_epoch':report['finished_epoch'],'prior_post_full5_sha256':sha(prior/'post-full5.json'),'rows':after,'current_full_payload_scan_performed':False}
def main(a):
 for fd,card in ([]if a.construction_only else [(8,0),(9,1)]):need(os.path.samefile('/proc/self/fd/'+str(fd),'/mnt/vm_8tb/b70/gpu.lock.'+str(card)),'Both original GPU leases required')
 plan=read(PLAN)
 for p,h in plan['files'].items():need(sha(ROOT/p)==h,'Qualification source changed '+p)
 need(sha(a.base_blueprint)==a.base_blueprint_sha256 and sha(a.fixtures)==a.fixtures_sha256 and sha(a.model_identity)==a.model_identity_sha256,'Pinned root software/input/identity metadata required')
 base=read(a.base_blueprint);fixture=read(a.fixtures)['fixtures'];need(len(fixture)==2 and [len(f['ids'])for f in fixture]==[54,524],'Authentic short/long tokenizer fixtures required')
 need(all(all(type(t)is int and 0<=t<248320 for t in f['ids'])for f in fixture),'Original tokenizer IDs required')
 out=a.output;need(out.is_absolute()and out.resolve()==out and not out.exists(),'New canonical output required');out.mkdir()
 model_manifest=read(a.model_identity);need(model_manifest['passed']is True and len(model_manifest['rows'])==4,'Locked MAIN four-shard manifest required')
 diagnostic=diagnostic_identity(a,model_manifest)if a.diagnostic_prior_root else None
 recipe_path=out/'intake.json';runtime=intake.prepare(a.compile_root,a.build_receipt_sha256,a.elf_sha256,a.base_blueprint,a.base_blueprint_sha256,recipe_path)
 build=read(a.compile_root/'output/receipt.json');sdk=Path(build['actual58']['SDK']);sdk_binding.configure(sdk,build['actual58']['receipt_sha256'])
 if a.construction_only:
  commands=[{'mode':mode,'command':native_command(base,runtime,sdk,out/('target-baseline'if mode==0 else'original-private-MTP'),mode)[0]}for mode in (0,1)];write(out/'construction.json',{'commands':commands,'pure_command_construction_passed':True,'GPU_launched':False,'model_payload_read':False,'actual_source_plan_sha256':sha(PLAN)});print(json.dumps({'pure_command_construction_passed':True,'output':str(out/'construction.json')}));return 0
 need(sdk_binding.binding()==build['actual58'],'Current actual58 source/ABI closure changed')
 report={'schema':1,'source_plan_sha256':sha(PLAN),'producer_pid':os.getpid(),'started_epoch':time.time(),'passed':False,'errors':[],'arms':[],'full5_scan_count':0,'HTTP_qualified':False,'persistent_cache_restart_qualified':False,'cache_eviction_qualified':False,'full_state_readback_qualified':False,'math_qualified':False,'speed_qualified':False,'served_model_names':['hotschmoe-dd','qwen3.8-flash-next-unsloth-ud-q4-k-xl-original-mtp53-research'],'actual_compile_receipt_sha256':a.build_receipt_sha256,'actual_ELF_sha256':a.elf_sha256,'identity_mode':'prior_authenticated_whole_hash_plus_current_stat5_and_known3_pages'if diagnostic else'fresh_pre_post_full5','diagnostic_identity':diagnostic,'native_coherence_promotion_qualified':False,'shelf_qualified':False}
 stopped=[]
 saved={n:signal.signal(n,lambda number,frame:stopped.append(number))for n in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
 def check(argv,d,label,timeout=240):
  write(d/(label+'.command.json'),argv);start=time.time()
  with(d/(label+'.stdout')).open('wb')as so,(d/(label+'.stderr')).open('wb')as se:r=subprocess.run(argv,stdout=so,stderr=se,timeout=timeout)
  write(d/(label+'.result.json'),{'started_epoch':start,'finished_epoch':time.time(),'return_code':r.returncode,'stdout_sha256':sha(d/(label+'.stdout')),'stderr_sha256':sha(d/(label+'.stderr'))});need(r.returncode==0,label+' failed')
 def current_diagnostic(phase):
  start=time.time();rows=[]
  for old in diagnostic['rows']:
   p=Path(old['path']);need(p.resolve()==p and p.is_file()and not p.is_symlink()and stat5(p)==old['after'],'Diagnostic current canonical stat5 changed');rows.append({'path':str(p),'current_stat5':stat5(p)})
  third=Path(diagnostic['rows'][2]['path']);need(third.name.endswith('00003-of-00004.gguf'),'Exact publisher third shard required');value=pages.inspect_pages(third,(*pages.KNOWN_PAGES,(33530449920,'d6143e2592b92f8780244fa33efe380b0df686cdeb8e445f2db9ac149b6db20b')))
  if not value['passed']:
   for row in value['rows']:(out/(phase+'-bad-page-'+str(row['offset'])+'.raw')).write_bytes(row['data'])
  need(value['passed'],'Diagnostic known source pages changed')
  value['rows']=[{k:v for k,v in row.items()if k!='data'}for row in value['rows']]
  return {'phase':phase,'started_epoch':start,'finished_epoch':time.time(),'read_mode':'prior_authenticated_whole_hash_plus_current_stat5','fresh_model_bytes_read':False,'known_page_bytes_read':12288,'rows':rows,'known3_pages':value,'qualification_qualified':False,'prior_report_sha256':diagnostic['prior_report_sha256']}
 def full5(phase):
  if diagnostic:
   write(out/(phase+'-diagnostic-identity.json'),current_diagnostic(phase));return
  rows=[]
  for old in model_manifest['rows']+[{'path':str(MTP),'expected_sha256':MTP_SHA}]:
   p=Path(old['path']);need(p.resolve()==p and p.is_file()and not p.is_symlink(),'Canonical original asset required');before=stat5(p);digest=sha(p);after=stat5(p);rows.append({'path':str(p),'before':before,'after':after,'sha256':digest,'expected_sha256':old['expected_sha256'],'passed':before==after and digest==old['expected_sha256']})
  report['full5_scan_count']+=1;write(out/(phase+'-full5.json'),rows);need(all(r['passed']for r in rows),'Fresh original MAIN/MTP whole identity differs')
 def arm(mode,baseline=None):
  d=out/('target-baseline'if mode==0 else'original-private-MTP');d.mkdir();errors=[];guard_stop=threading.Event();guard_thread=None;active=[];native=None;row_saved=None;removed=False;forced=False;attempted=False
  name='b70-mtp53-serve-'+str(os.getpid())+'-'+str(mode);owner=str(os.getpid())+'-'+str(mode);label='b70.mtp53.serve.qualifier'
  command,argv,env,mounts,binds,devices,groups=native_command(base,runtime,sdk,d,mode)
  write(d/'native.command.json',command);write(d/'recipe.json',{'native_argv':argv,'native_env':env,'readonly_mounts':mounts,'requests':'four authentic54/524 prompts, 64 cap each; fresh and reuse cohorts plus owned cancellation'})
  def inspect():
   r=subprocess.run(['docker','inspect',name],capture_output=True,timeout=20)
   if r.returncode:need(absent(r.stderr,name),'Unknown original container state; lease retained');return None
   row=read_bytes(r.stdout)[0];cfg=row['Config'];h=row['HostConfig'];need(cfg['Labels'].get(label)==owner and cfg['Image']==RUNTIME and cfg['Entrypoint']==['/bin/bash']and cfg['Cmd']==command[-2:]and cfg['OpenStdin']is True and cfg['AttachStdin']is True,'Exact original Docker owner/command required')
   need(exact_binds(h['Binds'],binds)and h['Privileged']is False and h['NetworkMode']=='none'and not h.get('DeviceRequests')and not h.get('DeviceCgroupRules'),'Exact mount/device domain required')
   need(sorted((x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions'])for x in h['Devices'])==sorted((p,p,'rwm')for p in devices)and sorted(h['GroupAdd'])==groups,'Exact paired render ownership required')
   need(all(type(h[k])is int and h[k]==v for k,v in{'NanoCpus':16000000000,'Memory':105<<30,'MemorySwap':105<<30}.items())and type(row['RestartCount'])is int and row['RestartCount']==0,'Typed normal runtime bounds required');return row
  def cleanup():
   nonlocal forced
   while True:
    try:
     row=inspect()
     if row and row['State']['Running']:
      forced=True;subprocess.run(['docker','stop','--time','20',row['Id']],capture_output=True,check=True,timeout=40)
     return
    except BaseException as e:
     if 'Cleanup '+str(e)not in errors:errors.append('Cleanup '+str(e))
     write(d/'cleanup-pending.json',{'error':str(e),'lease_retained':True});time.sleep(1)
  result={'mode':mode,'started_epoch':time.time(),'passed':False,'errors':errors,'cohorts':[]}
  try:
   check([str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],d,'pre-health');check([str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'],d,'pre-collective')
   if diagnostic:
    def watch():
     try:
      with(d/'diagnostic-active-identity.jsonl').open('x',encoding='ascii')as f:
       n=0
       while not guard_stop.is_set():
        f.write(json.dumps(current_diagnostic('active-'+str(mode)+'-'+str(n)),ensure_ascii=True)+'\n');f.flush();n+=1
        if guard_stop.wait(1):break
     except BaseException as e:errors.append('Active diagnostic identity '+str(e));stopped.append('identity_guard')
    guard_thread=threading.Thread(target=watch,daemon=True);guard_thread.start()
   need(not stopped,'Signal before original launch');attempted=True;native=NativeProtocol(command,d,cleanup,ready_timeout=900,active=active);roster=Roster(4)
   for line in native.stdout:roster.consume(line)
   roster.capacity();write(d/'startup-info.json',roster.info)
   for cohort in range(3):
    rids=[cohort*10+i+1 for i in range(4)];fresh=0 if cohort==1 else 1
    for slot,rid in enumerate(rids):
     ids=fixture[slot%2]['ids'];roster.submit(rid,slot,ids,64);native.send(request_command(rid,slot,ids,fresh))
    deadline=time.monotonic()+1800;cancel=False
    while any(roster.requests[r]['done']is None for r in rids):
     need(not stopped and time.monotonic()<deadline,'Signal/request deadline');
     try:stamp,line=native.q.get(timeout=.25)
     except queue.Empty:continue
     need(line is not None,'Native early EOF or pipe quota');roster.consume(line)
     if cohort==2 and not cancel and roster.requests[rids[0]]['admitted']and len(roster.requests[rids[0]]['generated'])>=2 and not roster.requests[rids[0]]['done']:
      old=roster.requests[1];native.send('BSTOP 0 rid=1 slotgen='+str(old['slotgen']));native.send(roster.cancel(rids[0]));cancel=True
    need(cohort!=2 or cancel,'Actual admitted cancellation was not exercised')
    rows=[output_view(roster.requests[r])for r in rids];write(d/('cohort-'+str(cohort)+'.json'),rows);result['cohorts'].append(rows)
    reference=result['cohorts'][0]if baseline is None else baseline['cohorts'][0]
    for i,row in enumerate(rows):
     want=reference[i];need(row['input_ids']==want['input_ids'],'Original input changed')
     if cohort==2 and i==0:need(row['finish']=='cancel'and row['output_ids']==want['output_ids'][:len(row['output_ids'])],'Cancellation must preserve exact target output prefix')
     else:need(row['output_ids']==want['output_ids']and row['finish']==want['finish'],'Complete target/original history mismatch')
   need(native.close()==0 and native.capture['passed']is True and native.capture['forced_cleanup']is False,'Normal original native EOF/pipe completion required')
   write(d/'native.capture.json',native.capture)
   text=(d/'engine.combined.log').read_text('ascii');need('strata batch: stale/unidentified BSTOP0 rejected'in text,'Actual old owner cancellation was not refused')
   reuse=[line for line in text.splitlines()if line.startswith('MTP53_SERVE_REUSED ')];result['observed_shifted_reuse_events']=reuse;result['native_complete_histories_equal']=True;result['cancellation_prefix_equal']=True;result['stale_owner_refused']=True
   # No claim that four terminal replies prove entire internal byte state or persistent restart.
   result['shifted_cache_reuse_observed']=bool(reuse)if mode else False
  except BaseException as e:errors.append(type(e).__name__+': '+str(e))
  finally:
   if native is None and active:native=active[-1]
   if native is not None and not hasattr(native,'capture'):
    try:native.close(failed=True)
    except BaseException as e:errors.append('Protocol retirement '+str(e))
   if attempted:
    while True:
     try:
      row=inspect()
      if row and row['State']['Running']:cleanup();continue
      if native is not None and native.p.poll()is None:time.sleep(.25);continue
      need(native is not None,'Original attempted launch identity unavailable; lease retained')
      if row:row_saved=row;write(d/'terminal.json',row);subprocess.run(['docker','rm',row['Id']],capture_output=True,check=True,timeout=20);continue
      removed=True;break
     except BaseException as e:
      if 'Retirement '+str(e)not in errors:errors.append('Retirement '+str(e))
      write(d/'retirement-pending.json',{'error':str(e),'lease_retained':True});time.sleep(1)
   guard_stop.set()
   if guard_thread is not None:
    guard_thread.join(5)
    if guard_thread.is_alive():errors.append('Active identity observer failed to retire')
   try:
    need(row_saved is not None and removed and not forced and native.p.returncode==0 and native.capture['passed']is True and native.capture['forced_cleanup']is False,'Actual normal owner/EOF/removal required')
    state=row_saved['State'];need(all(state[k]is False for k in ['Running','Paused','Restarting','OOMKilled','Dead'])and type(state['Pid'])is int and state['Pid']==0 and type(state['ExitCode'])is int and state['ExitCode']==0 and state['Error']=='','Typed normal native terminal required');result.update(owned_removed=True,original_return_code=0,original_EOF=True,forced_cleanup=False)
   except BaseException as e:errors.append('Terminal '+str(e))
   for label,argv in [('post-health',[str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH]),('post-collective',[str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180']),('post-kernel',['journalctl','-k','--since','@'+str(int(result['started_epoch'])),'--no-pager'])]:
    try:check(argv,d,label)
    except BaseException as e:errors.append(label+' '+str(e))
   try:need(not re.search(r'Fault response|Faulted Address|Engine reset|Timedout job|segfault|GPU HANG|CAT error',(d/'post-kernel.stdout').read_text('ascii'),re.I),'Fresh kernel fault')
   except BaseException as e:errors.append('Kernel '+str(e))
   result['finished_epoch']=time.time();result['passed']=not errors;write(d/'result.json',result)
  need(result['passed'],'Native arm failed '+str(errors));return result
 try:
  full5('pre');baseline=arm(0);report['arms'].append(baseline);original=arm(1,baseline);report['arms'].append(original);report['four_slot_complete_histories_equal']=True;report['native_private_slot_coherence_qualified']=not bool(diagnostic);report['native_histories_observed_equal']=True;report['native_coherence_promotion_qualified']=not bool(diagnostic);report['owned_cancellation_and_stale_refusal_qualified']=not bool(diagnostic);report['owned_cancellation_and_stale_refusal_observed']=True
 except BaseException as e:report['errors'].append(type(e).__name__+': '+str(e))
 finally:
  if stopped:report['errors'].append('Signals '+str(stopped))
  try:need(sdk_binding.binding()==build['actual58'],'Closing actual58 source/ABI closure changed')
  except BaseException as e:report['errors'].append('SDK '+str(e))
  try:full5('post')
  except BaseException as e:report['errors'].append('Closing identity '+str(e))
  for p,h in plan['files'].items():
   try:need(sha(ROOT/p)==h,'Closing qualification source changed')
   except BaseException as e:report['errors'].append('Source '+str(e))
  for p,h in [(a.base_blueprint,a.base_blueprint_sha256),(a.fixtures,a.fixtures_sha256),(a.model_identity,a.model_identity_sha256),(a.compile_root/'output/receipt.json',a.build_receipt_sha256),(a.compile_root/'output/original-mtp53-serve',a.elf_sha256)]:
   try:need(sha(p)==h,'Closing actual metadata/ELF changed')
   except BaseException as e:report['errors'].append('Input '+str(e))
  for n,h in saved.items():signal.signal(n,h)
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['artifacts']={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='report.json'};write(out/'report.json',report)
 print(json.dumps({'passed':report['passed'],'errors':report['errors'],'report':str(out/'report.json')},ensure_ascii=True));return 0 if report['passed']else 1
def read_bytes(v):return json.loads(v)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--diagnostic-prior-root',type=Path);p.add_argument('--diagnostic-prior-report-sha256');p.add_argument('--construction-only',action='store_true');p.add_argument('--compile-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 for k in ['base-blueprint','fixtures','model-identity']:p.add_argument('--'+k,type=Path,required=True);p.add_argument('--'+k+'-sha256',required=True)
 p.add_argument('--build-receipt-sha256',required=True);p.add_argument('--elf-sha256',required=True);raise SystemExit(main(p.parse_args()))
