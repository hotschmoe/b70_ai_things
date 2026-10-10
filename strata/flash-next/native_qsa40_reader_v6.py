"""Raw source40 OFF/ON target admission; native arrays are comparisons only."""
import hashlib,json,re,collections
from pathlib import Path
import numpy as np
import native_qsa40_runtime_v6 as r
import native_layer3_qsa_observer_contract_v1 as qsa
from collect_prefix_residual30_v2 import collect
from collect_ple_input33_v1 import collect as collect_input33
from audit_prefix30_lifecycle_v1 import audit_text
from serial37_canonical_json_v3 import canonical,matches_saved
require=r.require

def consume(path,limit=1<<30):
 path=Path(path);require(path.is_file()and not path.is_symlink()and path.absolute()==path.resolve(),'Regular original capture/log required');s=path.stat();require(s.st_size<=limit,'Bounded original read');raw=path.read_bytes();require(len(raw)==s.st_size and path.stat()==s and path.read_bytes()==raw and path.stat()==s,'Consumed original bytes changed');return raw

def numeric(raw,directory,stages):
 expected=r.request_command(qsa.IDS,1,1,None);require(raw['ids']==list(qsa.IDS)and raw['command']==expected and raw['fresh']==1 and type(raw['fresh'])is int and raw['pin']is None and raw['cancel_requested']is None and raw['stop_sent']is False,'Actual exact fresh4/MAX1/absentPIN request required')
 done=raw['done'].split();require(len(done)>=15 and done[0]=='DONE' and int(done[1])==len(raw['output_ids'])==1 and int(done[2])==4 and done[5]in('length','stop')and int(done[8])==0 and int(done[14])==4,'Actual DONE nread/nout/cancel/evaluated counts differ')
 meta=r.numeric.extract_numeric(raw,directory/'captures',True,True,stages);ledger=meta['ledger'];selection=meta['selection'];require(ledger['actual_reused']==ledger['candidate_resume']==0 and ledger['evaluated_prompt_rows']==4 and ledger['evaluated_decode_rows']==0 and ledger['generated']==1 and ledger['cancelled']is False and ledger['finish']in('length','stop'),'Actual fullprompt fresh numeric counters differ')
 require(all(selection[k]==0 for k in ('actual_reused','candidate_resume','parked_bytes','parked_entries','evictions','checkpoints'))and selection['reread']is False,'Actual cacheOFF source counters differ');events=[json.loads(line[4:])for line in raw['stderr']if line.startswith('PCL ')];begin=[e for e in events if e['event']=='begin'];commit=[e for e in events if e['event']=='committed_live'];require(len(begin)==len(commit)==1 and commit[0]['published']is False and commit[0]['chain_updated']is True and commit[0]['live_reusable']is False,'Actual source40 nonreusable committed tuple differs')
 require(begin[0]['tokens']==4 and begin[0]['input_sha256_le32']==r.numeric.le32_digest(list(qsa.IDS)),'Actual lifecycle input digest/count differ')
 for event in events:require(event['pid']==begin[0]['pid']and event['request']==begin[0]['request']and event['event']not in('skip','cache'),'Actual numerical lifecycle PID/request/cache event differs')
 require(commit[0]['phase']=='complete'and commit[0]['finish']==done[5]and commit[0]['ids_truncated']is False and commit[0]['tokens']==4 and commit[0]['ids']==list(qsa.IDS)and commit[0]['sha256_le32']==r.numeric.le32_digest(list(qsa.IDS)),'Actual complete consumed IDs/digest differ')
 spans=[event for event in events if event['event']=='stage_span'];key=lambda e:(e['phase'],e['device'],e['lb'],e['le'],e['lo'],e['hi']);entered=collections.Counter(key(e)for e in spans if not e['complete']);returned=collections.Counter(key(e)for e in spans if e['complete']);require(entered==returned,'Actual stage entry/return pairing incomplete')
 for event in spans:require((event['lb'],event['le'])in stages and event['device']==stages.index((event['lb'],event['le'])),'Actual numeric stage owner differs')
 for bounds in stages:
  covered=sorted(i for e in spans if e['complete']and(e['lb'],e['le'])==bounds for i in range(max(0,e['lo']),min(4,e['hi'])));require(covered==list(range(4)),'Actual numeric fullprompt stage spans missing/duplicate')
 owner=meta['logits'][0];require(begin[0]['pid']==int(owner['pid'])and begin[0]['request']==int(owner['request']),'PCL/SFD PID/request crossjoin differs')
 return meta

def terminal(result,directory,plan,argv,parent_pid):
 require(type(result['producer_pid'])is int and result['producer_pid']==parent_pid and result['passed']is True and result['errors']==[] and type(result['engine_rc'])is int and result['engine_rc']==0 and result['removed']is True and result['forced_cleanup']is False,'Actual clean owned engine terminal differs');obj=result['inspection'];require(matches_saved(obj,directory/'inspection.json'),'Original terminal inspection differs');r.inspection(obj,argv,plan);s=obj['State'];require(s['Running']is False and s['Paused']is False and s['Restarting']is False and s['Dead']is False and type(s['Pid'])is int and s['Pid']==0 and type(s['ExitCode'])is int and s['ExitCode']==0 and s['OOMKilled']is False and s['Error']=='' and s['Status']=='exited','Actual typed Docker terminal differs')
 cap=result['stdout_capture'];require(cap['passed']is True and cap['eof']is True and cap['reader_retired']is True and cap['error']is None and cap['forced_cleanup']is False and cap['return_code']==0 and cap['path']==str(directory/'engine.combined.log')and cap['sha256']==r.sha(directory/'engine.combined.log') and cap['process_started_epoch']<=cap['started_epoch']<=cap['completed_epoch']<=cap['finished_epoch']<=result['finished_epoch'],'Actual original merged log/EOF retirement differs')
 from native_qsa40_owned_launch_v6 import fence
 require(fence(directory),'Actual original attach/session not retired');launch=r.read(directory/'launch.json');retired=r.read(directory/'launch-retired.json');require(launch['client_identity_observed']is True and launch['actual_client_started']is True and launch['owner']['pid']==parent_pid and launch['command']==argv and retired['command']==argv and type(retired['return_code'])is int and retired['return_code']==0,'Actual original launched/retired command/PID differs')
 return cap

def arm(directory,plan,phase,on,binding):
 directory=Path(directory).resolve();result=r.read(directory/'result.json');require(canonical(r.read(directory/'plan.snapshot.json'))==canonical(plan),'Actual child plan snapshot differs');require(result['on']is on,'Actual arm activation differs');argv,env,_=r.command(plan,directory,phase['child_pid'],on,result['render_gid']);require(r.read(directory/'command.json')==argv and r.read(directory/'environment.json')==env,'Actual full native Docker recipe differs');terminal(result,directory,plan,argv,phase['child_pid']);log=consume(directory/'engine.combined.log').decode('ascii');require(log.endswith('\n')and not re.search(r'(^|\n)(ERR|FATAL|SERR)\b|trying a smaller expert cache|trying a \d+-token chunk',log),'Actual errors/capacity fallback/truncation rejected')
 stages=[tuple(v)for _,v in sorted(plan['recipe']['stages'].items())];requests=r.read(directory/'requests.json');require([row['ordinal']for row in requests]==[1,2,3,4],'Actual request roster differs');rows=[]
 for row in requests:
  raw=row['raw'];require(matches_saved(raw,directory/('request-'+str(row['ordinal'])+'.json')),'Original individual request differs');rows.append({'ordinal':row['ordinal'],'raw':raw,'meta':numeric(raw,directory,stages)})
 expected_capture=set()
 for row in rows:
  for field in row['meta']['logits']+row['meta']['residuals']:
   file=Path(field['path']);payload=consume(file,1<<20);require(len(payload)==int(field['bytes'])and hashlib.sha256(payload).hexdigest()==field['sha256'],'Exact consumed SFD bytes/SHA differ');expected_capture.add(file.name)
 require({p.name for p in(directory/'captures').iterdir()}==expected_capture,'Exact native SFD file roster required')
 actual={int(row['meta']['logits'][0]['request']):row['raw']['ids']for row in rows};require(set(actual)=={1,2,3,4},'No warm/stale source request ordinals allowed');p30=collect(directory/'p30',actual,dict(enumerate(stages)),binding,directory/'engine.combined.log');source33=collect_input33(directory/'ple-input',actual,binding,directory/'engine.combined.log',stages[0]);expected_p30=set()
 for frame in p30['frames']:
  expected_p30.add(Path(frame['metadata']).name);consume(frame['metadata'],1<<20)
  for field in frame['fields']:
   payload=consume(field['path'],81920);require(hashlib.sha256(payload).hexdigest()==field['sha256'],'Exact consumed P30 bytes/SHA differ');expected_p30.add(Path(field['path']).name)
 require({p.name for p in(directory/'p30').iterdir()}==expected_p30,'Exact native P30 file roster required')
 expected_input33=set()
 for frame in source33['frames']:
  expected_input33.add(Path(frame['metadata']).name);consume(frame['metadata'],1<<20)
  for field in frame['fields'].values():
   payload=consume(field['path'],10240);require(hashlib.sha256(payload).hexdigest()==field['sha256'],'Exact consumed source33 bytes/SHA differ');expected_input33.add(Path(field['path']).name)
 require({p.name for p in(directory/'ple-input').iterdir()}==expected_input33,'Exact native source33 file roster required')
 life=audit_text(log,dict(enumerate(stages)),True);require(life['passed'],'Actual P30/SFD lifetime audit failed')
 # Each P30 final FFN row joins its own SFD request, not the first equal prefix.
 last=[]
 for frame in p30['frames']:
  d=frame['binding']
  if d['route']!='verifier':continue
  row=next(x for x in rows if int(x['meta']['logits'][0]['request'])==d['request']);require(int(row['meta']['logits'][0]['pid'])==d['pid'],'Actual P30/SFD process differs');residual={int(x['layer']):x for x in row['meta']['residuals']}
  for field in frame['fields']:
   if field['phase']=='ffn':
    other=residual[field['layer']];comparison=r.numeric.compare_vectors(field,other);require(comparison['bitwise_equal'],'Actual P30/SFD last FFN mismatch');last.append((d['request'],field['layer']))
 require(len(last)==196-4 and len(set(last))==192,'Complete192 lastrow layer joins required')
 targets=[];files=sorted((directory/'qsa').glob('*.json'));markers=[line for line in log.splitlines()if line.startswith('QSA3 ')]
 if not on:require(not list((directory/'qsa').iterdir())and not markers,'OFF native observer produced evidence')
 else:
  frames=[];expected_names=set()
  for path in files:
   frame=r.read(path);qsa.recollect(path,binding);frames.append(frame);expected_names.add(path.name);expected_names.update(item['file']for item in frame['fields'])
  require({p.name for p in(directory/'qsa').iterdir()}==expected_names,'Exact native target file roster required')
  for request in range(1,5):qsa.whole_prefix([f for f in frames if f['request']==request],binding,len(stages)==2)
  require(len(frames)==12*len(stages),'Exact allrequest/stage/window native frame coverage required')
  for stage,bounds in enumerate(stages):
   allocations=[line for line in markers if line.startswith('QSA3 allocation ')and 'stage='+str(stage)+' 'in line];releases=[line for line in markers if line.startswith('QSA3 release_returned ')and 'stage='+str(stage)+' 'in line];require(len(allocations)==len(releases)==1,'Unique actual observer allocation/release perstage required');owner=stage==0;quota=sum(row['bytes']*(2 if row['rows']else 1)for row in qsa.fields().values())+16 if owner else 0;pid=frames[0]['pid'];require(allocations[0]==f'QSA3 allocation pid={pid} stage={stage} lb={bounds[0]} le={bounds[1]} owner={int(owner)} bytes={quota} target_only=1'and releases[0]==f'QSA3 release_returned pid={pid} stage={stage} owner={int(owner)}','Actual target owner byte quota/release differs')
  frame_markers=[line for line in markers if line.startswith('QSA3 frame ')];require(len(frame_markers)==len(frames)and len(markers)==len(frames)+2*len(stages),'Exact unique QSA marker roster required')
  for frame in frames:
   require(frame['pid']==int(rows[frame['request']-1]['meta']['logits'][0]['pid']),'Actual target/SFD PID association differs');name=f"qsa3-{frame['pid']}-r{frame['request']}-s{frame['stage']}-{frame['route']}-p{frame['first_position']}-n{frame['rows']}.json";marker=f"QSA3 frame pid={frame['pid']} request={frame['request']} stage={frame['stage']} owner={int(frame['owner'])} route={frame['route']} p0={frame['first_position']} rows={frame['rows']} fields={29 if frame['owner']else 0} nonce={frame['nonce']} metadata=/results/qsa/{name}";require(frame_markers.count(marker)==1,'Actual source-bound frame marker differs');targets.append({'metadata':str(directory/'qsa'/name),'frame':frame})
 return {'rows':rows,'p30':p30,'source33':source33,'lifecycle':life,'targets':targets,'result_sha256':r.sha(directory/'result.json'),'log_sha256':r.sha(directory/'engine.combined.log')}

def p30_vectors(value):
 out={}
 for frame in value['frames']:
  b=frame['binding']
  for field in frame['fields']:
   key=(b['request'],b['first_position'],b['rows'],field['layer'],field['phase']);require(key not in out,'Duplicate wholeprefix P30 numerical key');out[key]=field
 return out

def comparisons(off,on):
 from layer0_numerical_qualification_v9 import compare_numeric
 numeric=[compare_numeric(a,b)for a,b in zip(off['rows'],on['rows'])];left=p30_vectors(off['p30']);right=p30_vectors(on['p30']);require(set(left)==set(right)and sum(key[2]for key in left)==2304,'Complete exact2304 wholeprefix phase pairs required');phases=[]
 for key in sorted(left):
  pair=r.numeric.compare_vectors(left[key],right[key]);require(pair['bitwise_equal'],'QSA observer changed actual wholeprefix output');phases.append({'key':list(key),'logical_row_pairs':key[2],**pair})
 return {'full49':numeric,'wholeprefix':phases,'full49_pairs':196,'wholeprefix_pairs':2304,'observer_bitwise_equivalent':True,'full_model_math_qualified':False}
