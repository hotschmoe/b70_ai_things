#!/usr/bin/env python3
"""Inside-container loaded fixture runner. Root compiles and leases execution."""
import argparse,ast,copy,hashlib,json,os,queue,re,signal,subprocess,threading,time
from pathlib import Path
import pti_owned_temporal_control_v2 as control
import probe_unitrace_indexer40_v5 as clocks
import pti_paused_fixture_coverage_v1 as coverage
import pti_bounded_gen4_contract_v1 as sampled
import collect_source43_sampled_model_pti_v1 as strict_clock
TOOL_SHA='9f481a37a2884f7c7d499f33b5f41dbc00e2c0d089a9e786d059881803417b0c'
LIB_SHA='c72c6b4f42d8fe1a2c352286f5099280038b2136843924489d84a01887f43c4b'
require=control.require

def bound(path,limit):
 path=Path(path);require(path.resolve()==path and path.is_file()and not path.is_symlink(),'Canonical raw file required');before=path.stat();require(before.st_size<=limit,'Bounded fixture output required');raw=path.read_bytes();after=path.stat();require(control.signature(before)==control.signature(after)and len(raw)==after.st_size and path.read_bytes()==raw,'Exact consumed bytes changed');return raw,{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'stat':control.signature(after)}

def libraries(path):
 started=time.time();cmd=['/usr/bin/ldd',str(path)];proc=subprocess.run(cmd,capture_output=True,timeout=20);require(proc.returncode==0 and b'not found'not in proc.stdout+proc.stderr,'All actual hard dependencies must resolve')
 raw=proc.stdout.decode('ascii');files={}
 for name in sorted(set(re.findall(r'(/[A-Za-z0-9_./+:-]+)',raw))):
  original=Path(name);target=original.resolve();data,b=bound(target,64<<20);files[name]={'target':str(target),'binding':b}
 require(files,'Actual resolved dependency roster empty');return {'command':cmd,'return_code':proc.returncode,'stdout':raw,'stderr':proc.stderr.decode('ascii'),'resolved_files':files,'started_epoch':started,'finished_epoch':time.time()}

def library_projection(value):return {**value,'stdout':re.sub(r'\(0x[0-9a-fA-F]+\)','(ADDRESS)',value['stdout']),'started_epoch':None,'finished_epoch':None}

def fixture_clock_port(properties):
 # Original strict grammar and sampled-origin/thread joins, with observed
 # cardinality allowed to be zero for the stock paused negative control.
 base=strict_clock.inherited
 node=copy.deepcopy(next(n for n in ast.parse(Path(base.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='clock_port'))
 changed=0
 for n in ast.walk(node):
  if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='ast'and n.attr=='Gt':n.attr='GtE';changed+=1
 require(changed==1,'Exactly one stock zero-instance cardinality scope change')
 statement=next(n for n in node.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and getattr(n.value.func,'id',None)=='exec');at=node.body.index(statement)
 node.body[at:at]=ast.parse("node=extend_clock(node)\nnamespace['bridge_grammar']=sampled_bridge").body
 namespace={**vars(base),'extend_clock':strict_clock.extend_clock,'sampled_bridge':strict_clock.sampled_bridge}
 exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[strict-zero-or-observed-count]','exec'),namespace)
 return namespace['clock_port'](properties)

def audit_rows(rows,events,retention,cut,properties):
 require(type(cut)is int and cut>=0,'Exact paused endpoint sequence required')
 require(rows and all(type(r['sequence'])is int and r['sequence']==i+1 for i,r in enumerate(rows)),'Original ordered clock sequence required')
 sampled.audit_sampled_rows(rows)
 require(all(r.get('namespace')=='b70-pti-clock-export-v1'and r.get('source_base')=='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'and type(r.get('schema'))is int and r['schema']==1 for r in rows),'Original clock namespace/source changed')
 instances=[r for r in rows if r['kind']=='instance_normalization'];require(not any(r['sequence']<=cut for r in instances),'Warmup normalization must remain excluded')
 devices=[r for r in rows if r['kind']=='device_origin'];require(len(devices)==1,'One actual masked device origin required');device=devices[0]
 require(all(device[k]==properties[k] for k in ('timer_frequency_hz','kernel_valid_bits','global_valid_bits','timer_mask','uuid_word0_native_u64','uuid_word1_native_u64')),'Actual API-property device association changed')
 gpu=[e for e in events if e.get('ph')=='X'and e.get('cat')=='gpu_op'];result=coverage.child_coverage(instances,gpu,retention);result.update(coverage.regular_replay_path(events))
 selected=[]
 for r in instances:
  keys=('timer_mask','timer_frequency_hz','ns_per_cycle_ieee754_u64','raw_kernel_start_ticks','raw_kernel_end_ticks','submit_device_ticks','submit_host_ns')
  require(all(type(r[k])is int and 0<=r[k]<(1<<64)for k in keys),'Exact raw u64 timestamps required')
  require(all(r[k]==device[k]for k in('device_handle','timer_frequency_hz','timer_mask','ns_per_cycle_ieee754_u64')),'Instance device changed')
  require(clocks.normalize_clock({k:r[k]for k in keys})==(r['normalized_start_host_ns'],r['normalized_end_host_ns']),'Original normalization arithmetic changed')
  if r['instance_id']in result['selected_child_instance_ids']:
   duration=(r['raw_kernel_end_ticks']-r['raw_kernel_start_ticks'])&(r['timer_mask']);require(0<duration<(r['timer_mask']+1)//2,'Selected duration empty/wrap-ambiguous');selected.append(r)
 if retention:
  byid={e['args']['id']:e for e in gpu};ordered=sorted(selected,key=lambda r:0 if 'PausedRegularFixtureA'in byid[str(r['instance_id'])]['name']else 1)
  gap=(ordered[1]['raw_kernel_start_ticks']-ordered[0]['raw_kernel_end_ticks'])&device['timer_mask'];require(gap<(device['timer_mask']+1)//2,'In-order A->B raw tick ordering absent')
 hosts=[e for e in events if e.get('ph')=='M'and e.get('name')=='process_name'and e.get('args',{}).get('name','').startswith('HOST<')]
 require(len(hosts)==1 and type(hosts[0]['pid'])is int and hosts[0]['pid']==rows[0]['pid'],'Original clock/Chrome actual writer owner differs')
 result.update(no_warmup_normalizations=True,all_observed_normalizations_recomputed=True,physical_idle_qualified=False);return result

def run(retention,fixture_sha,properties):
 out=Path('/results');tool=Path('/pti/bin/unitrace');lib=Path('/pti/lib/libunitrace_tool.so');leaf=Path('/leaf/fixture')
 for f,want in((tool,TOOL_SHA),(lib,LIB_SHA),(leaf,fixture_sha)):require(hashlib.sha256(f.read_bytes()).hexdigest()==want,'Actual fresh ELF bytes changed')
 library_before={str(p):libraries(p)for p in(tool,lib,leaf)}
 owner=control.process_identity(os.getpid());session=control.session_name(owner['pid'],owner['start_ticks'],fixture_sha);env=dict(os.environ);env.update(UNITRACE_CLOCK_METADATA_PATH='/results/unitrace/clock.{pid}.jsonl',UNITRACE_HOST_CLOCK_BRIDGE='1',UNITRACE_CLOCK_QUERY_SAMPLE_512='1',UNITRACE_RETAIN_PAUSED_REGULAR_LISTS='1'if retention else'0');(out/'unitrace').mkdir()
 cmd=[str(tool),'--start-paused','--session',session,'--chrome-call-logging','--chrome-kernel-logging','--chrome-device-logging','-d','-s','-t','--pid','--tid','--verbose','--output-dir-path','/results/unitrace',str(leaf)]
 proc=None;threads=[];lines=queue.Queue();pump_errors=[];error=None;result={};started=time.time();cut=0
 try:
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=True)
  def pump(stream,name):
   total=0
   try:
    with(out/name).open('wb')as f:
     for line in iter(lambda:stream.readline((1<<20)+1),b''):
      total+=len(line);require(total<=8<<20 and len(line)<=1<<20,'Bounded loaded fixture pipe');f.write(line);f.flush();lines.put(line)
   except BaseException as exc:pump_errors.append(type(exc).__name__+': '+str(exc))
  for stream,name in((proc.stdout,'leaf.stdout'),(proc.stderr,'leaf.stderr')):
   t=threading.Thread(target=pump,args=(stream,name));threads.append(t);t.start()
  deadline=time.monotonic()+180
  while True:
   require(not pump_errors and time.monotonic()<deadline and proc.poll()is None,'Warmup did not complete normally '+str(pump_errors))
   try:line=lines.get(timeout=.1)
   except queue.Empty:continue
   if line==b'PAUSED_LIST_WARM_DONE values=1,2\n':break
  child=control.process_identity(proc.pid);require(child['parent_pid']==os.getpid(),'Original live fixture child ownership required');exe=Path('/proc/'+str(proc.pid)+'/exe');require(str(exe.readlink())==str(leaf)and hashlib.sha256(exe.read_bytes()).hexdigest()==fixture_sha,'Original live child executable changed');result.update(actual_child=child,actual_child_exe=str(exe.readlink()),actual_child_exe_sha256=fixture_sha)
  before=control.snapshot(session);require(before['state']==1,'Actual paused IPC required');paths=list((out/'unitrace').glob('clock.*.jsonl'));require(len(paths)==1 and paths[0].name=='clock.'+str(proc.pid)+'.jsonl','Exact actual child clock writer required');raw,_=bound(paths[0],8<<20);rows=[clocks.read_json_line(x)for x in raw.splitlines()];require(raw.endswith(b'\n')and not any(r['kind']=='instance_normalization'for r in rows),'Warmup emitted normalization');cut=rows[-1]['sequence']
  cli=subprocess.run([str(tool),'--resume',session],env={'PATH':'/usr/bin:/bin','LANG':'C'},capture_output=True,timeout=15);require(cli.returncode==0,'Actual resume CLI failed');after=control.snapshot(session);control.check_transition(before,after,'resume');result.update(controller_owner=owner,session=session,before=before,after=after,resume_argv=[str(tool),'--resume',session],resume_return_code=cli.returncode)
  proc.stdin.write(b'R\n');proc.stdin.flush();proc.stdin.close();require(proc.wait(timeout=180)==0,'Loaded fixture failed')
 finally:
  if proc is not None:
   if proc.poll()is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   cleanup_errors=[]
   for t in threads:
    if t.ident is not None:
     t.join(timeout=10)
     if t.is_alive():cleanup_errors.append('Original fixture pipe did not retire')
   for stream in(proc.stdin,proc.stdout,proc.stderr):
    if stream is not None:stream.close()
   require(not cleanup_errors,'; '.join(cleanup_errors))
 require(not pump_errors,'Actual pipe failure '+str(pump_errors))
 stdout,stdout_b=bound(out/'leaf.stdout',8<<20);require([line for line in stdout.splitlines()if line.startswith(b'PAUSED_LIST_')]==[b'PAUSED_LIST_WARM_DONE values=1,2',b'PAUSED_LIST_SELECTED_DONE values=2,4',b'PAUSED_LIST_RETIRED graph=1 allocation=1'],'Exact actual graph/allocation retirement markers required')
 raw,clock_b=bound(paths[0],8<<20);chrome=list((out/'unitrace').glob('*.json'));require(len(chrome)==1,'Unique original Chrome trace');data,chrome_b=bound(chrome[0],32<<20);rows=[clocks.read_json_line(x)for x in raw.splitlines()];require(all(r['pid']==child['pid']for r in rows),'Actual clock child owner changed');result.update(audit_rows(rows,json.loads(data)['traceEvents'],retention,cut,properties));result['strict_clock']=fixture_clock_port(properties)(out,'on');result.update(passed=True,command=cmd,return_code=proc.returncode,started_epoch=started,finished_epoch=time.time(),paused_last_sequence=cut,raw_bindings=[stdout_b,clock_b,chrome_b],graph_and_allocation_retired=True)
 library_after={str(p):libraries(p)for p in(tool,lib,leaf)}
 require({k:library_projection(v)for k,v in library_before.items()}=={k:library_projection(v)for k,v in library_after.items()},'Actual dependency bytes changed')
 result.update(libraries_before=library_before,libraries_after=library_after,actual_mapped_dependency_roster_reconstructed=False)
 (out/'inside-receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--retention',choices=('off','on'),required=True);p.add_argument('--fixture-sha256',required=True);p.add_argument('--properties',type=Path,required=True);a=p.parse_args()
 try:run(a.retention=='on',a.fixture_sha256,json.loads(a.properties.read_bytes())['clock_properties'])
 except BaseException as exc:
  Path('/results/inside-failure.json').write_text(json.dumps({'passed':False,'error':type(exc).__name__+': '+str(exc),'finished_epoch':time.time()})+'\n');raise
