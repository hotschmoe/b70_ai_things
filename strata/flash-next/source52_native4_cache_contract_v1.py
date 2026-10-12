"""Current52 bounded native4 cache: original all49, owner histories and reuse gates."""
import hashlib,json,math,struct
from pathlib import Path
from source52_native4_coherence_contract_v1 import require,fields,command_for,vectors

def requests(actual, expected):
 require(len(actual)==len(expected) and len(expected) in (4,8),'Exact four cold or eight prime/hit requests required'); result=[]
 for item,want in zip(actual,expected):
  raw=item['raw']; require(item['ordinal']==want['ordinal'] and raw['command']==command_for(want),'Actual original command/ordinal differs')
  require(all(raw[k]==want[k] for k in ('ids','fresh','pin','rid','slot','cohort')) and not raw['LP'] and raw['stop_sent'] is False and raw['cancel_requested'] is None,'Actual request ownership/input/control differs')
  own=raw['stdout']; output=[int(line.split()[1 if line.startswith('T ') else 2]) for line in own if line.startswith(('T ','BT '))]
  require(output==raw['output_ids'] and len(output)==8 and all(0<=x<248320 for x in output),'Actual generated history differs')
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
 directory=Path(actual[0]['raw']['phase_directory']);require(all(Path(x['raw']['phase_directory'])==directory for x in actual),'One actual owned process per phase required')
 selected=[row for row in expected if row['observe']];require(len(selected)==4,'Exactly four selected source-observer owners required')
 observed,total=vectors(directory,selected,{r['rid']:r['output_ids'] for r in result})
 logs=(directory/'engine.combined.log').read_text(encoding='ascii').splitlines();chain=[fields(line) for line in logs if line.startswith('BCHAIN startup ')]
 require(len(chain)==1 and chain[0]['slots']=='4' and chain[0]['cap']=='3' and int(chain[0]['required'])<=int(chain[0]['chain_limit'])==8192<<20 and int(chain[0]['transfer_limit'])==512<<20,'Actual native4 complete-state chain capacity admission missing')
 fc=[json.loads(line[5:]) for line in logs if line.startswith('FC38 ')]
 for want in expected:
  begin=[x for x in fc if x.get('kind')=='request_begin' and x['rid']==want['rid']];work=[x for x in fc if x.get('kind')=='request_work' and x['rid']==want['rid']]
  require(len(begin)==len(work)==1,'Unique actual current owner/work source records required');b,w=begin[0],work[0];out=result[want['ordinal']-1]
  digest=hashlib.sha256(struct.pack('<%dI'%len(want['ids']),*want['ids'])).hexdigest()
  require(b['slot']==want['slot'] and b['tokens']==len(want['ids']) and b['input_sha256_le32']==digest and b['fresh']==bool(want['fresh']) and b['pin_present'] is True and b['pin']==1536 and b['raw_requested'] is want['observe'],'Exact authentic native4 cache request/input/capture identity differs')
  require(w['slotgen']==b['slotgen']==out['slotgen'] and w['command']==b['command'] and w['pid']==b['pid'] and w['slot']==b['slot'] and w['reused']==w['read_from']==want['expected_reused'] and w['reread_to']==-1 and w['prompt']==w['prompt_reached']==len(want['ids']) and w['cancelled'] is False,'Exact complete-state source reused/new prompt work differs')
 for cohort in sorted({row['cohort'] for row in expected}):
  owners=[row for row in expected if row['cohort']==cohort];require(len(owners)==4 and {row['slot'] for row in owners}==set(range(4)),'Actual four unique private owners required')
  receipt=json.loads((directory/('cohort-%d.json'%cohort)).read_bytes());require(receipt['cohort']==cohort and receipt['commands']==[command_for(row) for row in owners],'Exact submitted native4 cohort differs')
  events=[fields(e['line']) for e in receipt['events'] if e['line'].startswith('SBF batch_event ')]
  require(any(e['rows']=='4' and e['active_mask']=='15' and e['completed']=='1' for e in events),'Actual completed native4 cohort rows4/mask15 missing')
  captured=[row for row in owners if row['observe']]
  if not captured:continue
  require(len(captured)==4,'Whole native4 selected cohort required');later_events=set()
  for row in captured:
   later=observed[row['rid']][('batch_step_logits_before_sampler',1,-1)]
   require(later['batchrows']==4 and later['slot']==row['slot'] and later['slotgen']==result[row['ordinal']-1]['slotgen'] and any(int(e['event'])==later['event'] and e['rows']=='4' and e['active_mask']=='15' for e in events),'Actual later head lacks same current four-row owner/window')
   require(later['token']==result[row['ordinal']-1]['output_ids'][0],'Actual consumed first generated token/history differs');later_events.add(later['event'])
  require(len(later_events)==1,'All four later heads must share one actual window')
 if len(expected)==8:
  require(all(result[4+i]['slotgen']>result[i]['slotgen'] for i in range(4)),'Hit owners must replace each prime slot generation')
  prime={row['role']:row for row in result[:4]}
  for row in result[4:]:require(row['output_ids']==prime[row['role']]['output_ids'] and row['finish']==prime[row['role']]['finish'],'Prime/hit full owner histories differ')
 first=[];later=[]
 for row in selected:
  rid=row['rid'];first.append(dict(role=row['role'],vectors=[dict(stage=stage,layer=layer,**observed[rid][('admission'+('_residual' if layer>=0 else '_logits_before_sampler'),stage,layer)]) for layer in range(-1,48) for stage in [layer//24 if layer>=0 else 1]]));later.append(dict(role=row['role'],vectors=[dict(stage=stage,layer=layer,**observed[rid][('batch_step'+('_residual' if layer>=0 else '_logits_before_sampler'),stage,layer)]) for layer in range(-1,48) for stage in [layer//24 if layer>=0 else 1]]))
 return dict(requests=result,captured_requests=[result[row['ordinal']-1] for row in selected],observer_bytes=total,first49=first,later49=later,actual_four_row_decode_observed=True,original_fullmath_qualified=False,full_cache_qualified=False,API_qualified=False,clean_latency_qualified=False)

def equivalence(cold,primed):
 require(cold['actual_four_row_decode_observed'] is True and primed['actual_four_row_decode_observed'] is True and len(cold['requests'])==4 and len(primed['requests'])==8,'Exact fresh cold4 and prime4/hit4 phases required')
 c={row['role']:row for row in cold['requests']};p={row['role']:row for row in primed['requests'][:4]};h={row['role']:row for row in primed['captured_requests']};require(set(c)==set(p)==set(h)==set('ABCD'),'Four authentic independent/shared-divergent roles required')
 for role in 'ABCD':
  require(c[role]['actual_reused']==p[role]['actual_reused']==0 and h[role]['actual_reused']==1536 and h[role]['new_prompt_rows']==c[role]['new_prompt_rows']-1536,'Actual expected cache reuse/new work differs')
  require(all(c[role][key]==p[role][key]==h[role][key] for key in ('output_ids','finish')),'Cold/prime/hit complete8-token histories differ')
 for lane in ('first49','later49'):
  left={row['role']:row['vectors'] for row in cold[lane]};right={row['role']:row['vectors'] for row in primed[lane]};require(set(left)==set(right)==set('ABCD'),'All four full49 raw comparisons required')
  for role in 'ABCD':
   require(len(left[role])==len(right[role])==49,'Each owner needs all48 residuals/full head')
   for a,b in zip(left[role],right[role]):require(all(a[k]==b[k] for k in ('stage','layer','sha256','bytes','pos','token')),'Cold/hit first or later full49 mismatch')
 return dict(bounded_native4_shared_divergent_cache_passed=True,actual_completed_rows=4,all4_reused1536=True,new_prompt_rows={role:h[role]['new_prompt_rows'] for role in 'ABCD'},entire8_output_histories_equal=True,first49_bitwise=True,later49_bitwise=True,original_fullmath_qualified=False,full_cache_qualified=False,API_qualified=False,clean_latency_qualified=False,scope='actual cold4/prime4/hit4; first/later49 bitwise and full8 histories; no cache family/fairness/API qualification')
