"""Bounded native2 cache observations; no API, clean-speed or full-family claims."""
import array, hashlib, json, math, re
from pathlib import Path

def require(ok, message):
 if not ok: raise ValueError(message)

def fields(line): return dict(re.findall(r'(\w+)=([^ ]+)',line))

def command_for(row):
 prefix='GEN 8' if row['slot']<0 else 'BGEN %d 8'%row['slot']
 return prefix+' fresh=%d pin=1536 ckpt=1 temperature=0 rid=%d '%(row['fresh'],row['rid'])+' '.join(map(str,row['ids']))

def vectors(directory, expected, emitted):
 directory=Path(directory); lines=(directory/'engine.combined.log').read_text(encoding='ascii').splitlines()
 by_rid={row['rid']:{} for row in expected}; files=set(); replays=[]; rows=[]; total=0
 for line in lines:
  if line.startswith('SBF replay '): replays.append(fields(line))
  if line.startswith('SBF row '): rows.append(fields(line))
  if not line.startswith('SBF vector '): continue
  f=fields(line); rid=int(f['rid']); require(rid in by_rid,'Unexpected observed RID')
  stage=int(f['stage']); layer=int(f['layer']); phase=f['phase']; key=(phase,stage,layer)
  require(key not in by_rid[rid] and f['canonical']=='le_f32' and f['observed_only']=='1','Duplicate/noncanonical vector')
  require((stage,int(f['lb']),int(f['le'])) in ((0,0,24),(1,24,48)),'Actual exact paired stage differs')
  require((layer==-1 and stage==1 and int(f['floats'])==248320) or (0<=layer<48 and stage==layer//24 and int(f['floats'])==10240),'Actual layer/head geometry differs')
  target=directory/'captures'/Path(f['file']).name
  require(f['file']=='/results/captures/'+target.name and target.is_file() and not target.is_symlink() and target.name not in files,'Actual raw vector path/identity differs')
  before=target.stat(); raw=target.read_bytes(); after=target.stat()
  require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Raw vector changed during read')
  require(len(raw)==int(f['bytes'])==int(f['floats'])*4,'Actual complete vector length differs')
  values=array.array('f'); values.frombytes(raw)
  require(all(math.isfinite(x) for x in values),'Nonfinite raw vector')
  if layer==-1:
   index=int(phase.startswith('batch_step_')); require(len(emitted[rid])>index and values[emitted[rid][index]]==max(values),'Actual greedy emitted token differs from full captured head')
  joined=[p for p in replays if all(p.get(k)==f[k] for k in ('pid','enginegen','stage','lb','le','epoch','graph','event'))]
  require(len(joined)==1 and all(joined[0].get(k)=='1' for k in ('full_roster','input_output_verified','stage_context_verified')),'Missing actual completed immutable graph frame')
  owner=[p for p in rows if all(p.get(k)==f[k] for k in ('pid','enginegen','stage','epoch','graph','event','rid','requestgen','slot','slotgen','pos','token','row','batchrows'))]
  require(len(owner)==1 and owner[0]['selected']=='1','Raw row lacks exact current owner/frame association')
  files.add(target.name); total+=len(raw)
  by_rid[rid][key]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),pos=int(f['pos']),token=int(f['token']),slot=int(f['slot']),slotgen=int(f['slotgen']),event=int(f['event']),batchrows=int(f['batchrows']))
 require(total<=64<<20 and files=={p.name for p in (directory/'captures').iterdir() if p.is_file()},'Original observer quota or exact file census differs')
 for row in expected:
  phases=['direct_gen'] if row['slot']<0 else ['admission','batch_step']
  want={(phase+'_residual',layer//24,layer) for phase in phases for layer in range(48)}|{(phase+'_logits_before_sampler',1,-1) for phase in phases}
  require(set(by_rid[row['rid']])==want,'Missing all48 residuals/full head for actual request/step')
  for key,value in by_rid[row['rid']].items():
   later=key[0].startswith('batch_step_'); require(value['pos']==len(row['ids'])-1+int(later),'Observed prefix position differs')
   if not later: require(value['token']==row['ids'][-1],'Observed original prompt-tail token differs')
 return by_rid, total

def requests(actual, expected):
 require(len(actual)==len(expected)==6,'Six complete actual bounded requests required'); result=[]
 for item,want in zip(actual,expected):
  raw=item['raw']; require(item['ordinal']==want['ordinal'] and raw['command']==command_for(want),'Actual original command/ordinal differs')
  require(all(raw[k]==want[k] for k in ('ids','fresh','pin','rid','slot','cohort')) and not raw['LP'] and raw['stop_sent'] is False and raw['cancel_requested'] is None,'Actual request ownership/input/control differs')
  own=raw['stdout']; output=[int(line.split()[1 if line.startswith('T ') else 2]) for line in own if line.startswith(('T ','BT '))]
  require(output==raw['output_ids'] and 1<=len(output)<=8 and all(0<=x<248320 for x in output),'Actual generated history differs')
  admission=[line for line in own if line.startswith('DONE ')]; require(len(admission)==1,'Unique actual admission/serial DONE required'); d=admission[0].split()
  require(int(d[2])==len(want['ids']) and int(d[8])==want['expected_reused'] and int(d[14])==len(want['ids'])-want['expected_reused'],'Actual reused/new-prompt counters differ')
  require(all(math.isfinite(float(d[i])) and float(d[i])>=0 for i in (3,4,13)),'Actual native admission timing/bytes invalid')
  require(all(fields(line).get('rid')==str(want['rid']) for line in own),'Foreign request output attribution')
  generations={fields(line)['slotgen'] for line in own}; require(len(generations)==1 and int(next(iter(generations)))>=int(want['slot']>=0),'Native owner generation changed')
  if want['slot']>=0:
   terminals=[line for line in own if line.startswith('BDONE ')]; require(len(terminals)==1 and terminals[0]==raw['done'] and int(terminals[0].split()[1])==want['slot'] and int(terminals[0].split()[2])==len(output),'Actual batch terminal count/slot differs'); finish=terminals[0].split()[3]
   require(int(d[1])==1 and len([line for line in own if line.startswith('BADM ')])==1,'Actual first-token admission missing')
  else: require(admission[0]==raw['done'] and int(d[1])==len(output),'Actual serial terminal count differs'); finish=d[5]
  require(finish in ('length','stop') and (len(output)==8 if finish=='length' else output[-1] in (248044,248046)) and all(t not in (248044,248046) for t in output[:-1]),'Actual greedy cap/EOS history differs')
  times=raw['stdout_seconds']; require(len(times)==len(own) and times==sorted(times) and all(math.isfinite(t) and t>=0 for t in times),'Actual monotonic receive timestamps invalid')
  result.append(dict(role=want['role'],rid=want['rid'],output_ids=output,finish=finish,actual_reused=int(d[8]),new_prompt_rows=int(d[14]),first_token_receive_seconds=next(t for line,t in zip(own,times) if line.startswith(('T ','BT '))),slotgen=int(next(iter(generations)))))
 directory=Path(actual[0]['raw']['phase_directory']); require(all(Path(x['raw']['phase_directory'])==directory for x in actual),'Single owned process required')
 observed,total=vectors(directory,expected,{r['rid']:r['output_ids'] for r in result})
 logs=(directory/'engine.combined.log').read_text(encoding='ascii').splitlines()
 chain=[fields(line) for line in logs if line.startswith('BCHAIN startup ')]
 require(len(chain)==1 and chain[0]['slots']=='2' and chain[0]['cap']=='3' and int(chain[0]['required'])<=int(chain[0]['chain_limit'])==8192<<20 and int(chain[0]['transfer_limit'])==512<<20,'Actual full-state chain capacity admission missing')
 fc=[json.loads(line[5:]) for line in logs if line.startswith('FC38 ')]
 import struct
 for want in expected:
  begin=[x for x in fc if x.get('kind')=='request_begin' and x['rid']==want['rid']];work=[x for x in fc if x.get('kind')=='request_work' and x['rid']==want['rid']]
  require(len(begin)==len(work)==1,'Actual unique owner/cache work records required')
  b,w=begin[0],work[0];digest=hashlib.sha256(struct.pack('<%dI'%len(want['ids']),*want['ids'])).hexdigest()
  require(b['slot']==want['slot'] and b['tokens']==len(want['ids']) and b['input_sha256_le32']==digest and b['fresh']==bool(want['fresh']) and b['pin_present'] is True and b['pin']==1536 and b['raw_requested'] is True,'Actual source cache request/input differs')
  require(w['slotgen']==b['slotgen']==result[want['ordinal']-1]['slotgen'] and w['command']==b['command'] and w['pid']==b['pid'] and w['slot']==b['slot'] and w['reused']==w['read_from']==want['expected_reused'] and w['reread_to']==-1 and w['prompt']==w['prompt_reached']==len(want['ids']) and w['cancelled'] is False,'Actual complete-state source reuse/work differs')
 require(result[4]['slotgen']>result[2]['slotgen'] and result[5]['slotgen']>result[3]['slotgen'],'Reused private slots need fresh native owner generations')
 for cohort in (2,3):
  receipt=json.loads((directory/('cohort-%d.json'%cohort)).read_bytes()); want=[r for r in expected if r['cohort']==cohort]
  require(receipt['cohort']==cohort and receipt['commands']==[command_for(r) for r in want],'Actual two-request cohort command census differs')
  events=[fields(e['line']) for e in receipt['events'] if e['line'].startswith('SBF batch_event ')]
  require(any(e['rows']=='2' and e['active_mask']=='3' and e['completed']=='1' for e in events),'No actual completed two-row decode window')
  for row in want:
   later=observed[row['rid']][('batch_step_logits_before_sampler',1,-1)]
   require(later['batchrows']==2 and any(int(e['event'])==later['event'] and e['rows']=='2' and e['active_mask']=='3' for e in events),'Observed owner head lacks same actual two-row event')
   require(later['token']==result[row['ordinal']-1]['output_ids'][0],'Observed consumed output history differs')
 for role,serial,cold,hit in (('A',1,3,6),('B',2,4,5)):
  reference=result[serial-1]
  require(all(result[i-1]['output_ids']==reference['output_ids'] and result[i-1]['finish']==reference['finish'] for i in (cold,hit)),'Serial/cold/hit owner output history differs')
  for layer in range(-1,48):
   stage=1 if layer==-1 else layer//24; suffix='logits_before_sampler' if layer==-1 else 'residual'
   a=observed[serial][('direct_gen_'+suffix,stage,layer)]; b=observed[cold][('admission_'+suffix,stage,layer)]; h=observed[hit][('admission_'+suffix,stage,layer)]
   require(a['sha256']==b['sha256']==h['sha256'] and a['bytes']==b['bytes']==h['bytes'],'Serial/cold/hit first residual/head differs')
   a=observed[cold][('batch_step_'+suffix,stage,layer)]; b=observed[hit][('batch_step_'+suffix,stage,layer)]
   require(all(a[k]==b[k] for k in ('sha256','bytes','pos','token')),'Cold/hit full second-step state/head differs after owner swap')
 return dict(requests=result,observer_bytes=total,bounded_native2_shared_divergent_cache_observed=True,actual_two_row_decode_observed=True,full_cache_qualified=False,API_qualified=False,clean_latency_qualified=False)

def equivalence(*arms):
 require(len(arms)==1 and arms[0]['bounded_native2_shared_divergent_cache_observed'] is True,'One actual complete native2 probe required')
 return arms[0]
