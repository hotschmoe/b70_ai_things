"""NEW source41 target intake; captures are comparisons, never math operands."""
import hashlib,json,re,collections
from pathlib import Path
from types import SimpleNamespace
import serial_prefix_qualification_v6 as numeric_helpers
from serial_prefix_qualification_v8 import request_command
from serial37_canonical_json_v3 import read_unique,canonical,matches_saved
import native_layer3_qsa_observer_contract_v1 as qsa
from collect_prefix_residual30_v2 import collect
from collect_ple_input33_v1 import collect as collect_input33
from audit_prefix30_lifecycle_v1 import audit_text

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_binding():
 plan=Path(__file__).with_name('native-qsa41-fidelity-source-plan-v1.json');record=read_unique(plan);root=plan.parent.parents[1]
 for name,want in record['files'].items():require(sha(root/name)==want,'Current explicit41 fidelity source changed '+name)
 return sha(plan)
def consume(path,limit=1<<30):
 path=Path(path);require(path.is_file()and not path.is_symlink()and path.absolute()==path.resolve(),'Regular exact original capture required')
 def sig():
  s=path.stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 before=sig();require(before[2]<=limit,'Bounded raw capture required');raw=path.read_bytes();require(len(raw)==before[2]and sig()==before and path.read_bytes()==raw and sig()==before,'Consumed raw capture changed');return raw
r=SimpleNamespace(request_command=request_command,numeric=numeric_helpers,read=read_unique,sha=sha)
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

def actor_header(actor_binding):
 require(type(actor_binding)is dict and type(actor_binding['materialized_mode'])is int and actor_binding['materialized_mode']in(0,1)and type(actor_binding['observer_enabled'])is bool and type(actor_binding['engine_pid'])is int and actor_binding['engine_pid']>0,'Typed actual41 configured mode/observer/PID required')
 require(actor_binding['observer_enabled']is True,'Original41 localization requires observer capture, independently of materialized mode')
 root=Path(actor_binding['capture_root']);require(root.resolve()==root and root.is_dir()and not root.is_symlink(),'Exact original capture root required')
 for role in('command','result','phase_receipt'):
  path=Path(actor_binding[role+'_path']);require(path.parent==root and hashlib.sha256(consume(path,16<<20)).hexdigest()==actor_binding[role+'_sha256'],'Original actor artifact path/SHA changed '+role)
 result=read_unique(actor_binding['result_path']);phase=read_unique(actor_binding['phase_receipt_path']);require(result['passed']is True and result['error']is None and result['errors']==[]and type(result['engine_rc'])is int and result['engine_rc']==0 and type(result['return_code'])is int and result['return_code']==0,'Actual captured result failed or untyped');require(phase['passed']is True and phase['error']is None and type(phase['return_code'])is int and phase['return_code']==0,'Actual enclosing capture phase failed or untyped')
 from source41_sdk_binding_v1 import binding
 current=binding();require(canonical(current)==canonical(actor_binding['source41_sdk']),'Fresh actual41 SDK/header/source/8ABI changed')
 require(sha(Path(current['engine_root'])/'source/sycl/include/strata/core/layer3_qsa_target_observer.hpp')==sha(qsa.HEADER),'Actual41 observer schema/header differs from explicit target contract')
 require(current['actual_receipt_source_rows']==67 and current['independently_required_source_rows']==69 and current['old_source40_oracle_or_runtime_transferred']is False,'Source40 proof cannot become actual41')
 command=read_unique(actor_binding['command_path']);flag='STRATA_QSA_INDEXER_MATERIALIZED_HALF='+str(actor_binding['materialized_mode']);values=[command[i+1]for i,word in enumerate(command[:-1])if word=='-e'and command[i+1].startswith('STRATA_QSA_INDEXER_MATERIALIZED_HALF=')];require(values==[flag],'Actual materialized mode must match exact producer command environment')
 require(actor_binding['stages']==[[0,0,48]]and all(type(x)is int for row in actor_binding['stages']for x in row),'Bounded original41 one-card full48 stage required')
 return root

def captured_arm(actor_binding,requests_given):
 source=source_binding();directory=actor_header(actor_binding);observer=actor_binding['observer_enabled'];binding=actor_binding['binding_sha256'];stages=[(0,48)]
 log=consume(directory/'engine.combined.log').decode('ascii');require(log.endswith('\n')and not re.search(r'(^|\n)(ERR|FATAL|SERR)\b|trying a smaller expert cache|trying a \d+-token chunk',log),'Actual errors/capacity fallback/truncation rejected')
 requests=r.read(directory/'requests.json');require(canonical(requests)==canonical(requests_given),'Caller/raw request roster changed');require([row['ordinal']for row in requests]==[1,2,3,4],'Actual request roster differs');rows=[]
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
 if not observer:require(not list((directory/'qsa').iterdir())and not markers,'OFF native observer produced evidence')
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
 require(all(int(row['meta']['logits'][0]['pid'])==actor_binding['engine_pid']for row in rows),'Actual actor/PCL/SFD PID differs')
 require(source_binding()==source and actor_header(actor_binding)==directory,'Current source/actual actor bindings changed after raw recollection')
 return {'actor_binding':actor_binding,'materialized_mode':actor_binding['materialized_mode'],'observer_enabled':observer,'rows':rows,'p30':p30,'source33':source33,'lifecycle':life,'targets':targets,'result_sha256':r.sha(directory/'result.json'),'log_sha256':r.sha(directory/'engine.combined.log')}
