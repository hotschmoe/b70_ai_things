"""Same-host native token timing and complete-history coherence; no broad promotion."""
import json,math
from pathlib import Path
from native2_fairness_protocol_v1 import require,fields,command_for

def percentile(values,p):
 if not values:return None
 ordered=sorted(values);where=(len(ordered)-1)*p;lo=int(where);hi=min(lo+1,len(ordered)-1);return ordered[lo]+(ordered[hi]-ordered[lo])*(where-lo)
def gap_summary(times):
 gaps=[1000*(b-a) for a,b in zip(times,times[1:])]
 return dict(gaps=len(gaps),P50_ms=percentile(gaps,.5),P95_ms=percentile(gaps,.95),maximum_ms=max(gaps) if gaps else None,percentile_definition='linear interpolation at (n-1)*p',token_receive_monotonic_seconds=times)

def requests(rows,expected):
 require(len(rows)==len(expected)==2,'Exact long512 and incoming64 owners required');views=[]
 for row,want in zip(rows,expected):
  raw=row['raw'];require(row['ordinal']==want['ordinal'] and raw['command']==command_for(want) and all(raw[k]==want[k] for k in ('ids','rid','slot','cohort')) and raw['fresh']==1 and raw['pin']==1536 and not raw['LP'] and raw['stop_sent'] is False and raw['cancel_requested'] is None,'Exact source/input/sampling/current owner differs')
  lines=raw['stdout'];times=raw['stdout_monotonic_seconds'];relative=raw['stdout_seconds'];send=raw['send_monotonic_ns']/1e9
  require(len(lines)==len(times)==len(relative) and times==sorted(times) and all(math.isfinite(x) and x>=send for x in times) and relative==[x-send for x in times],'Actual same-host receive/send clock association differs')
  require(all(fields(line).get('rid')==str(want['rid']) for line in lines),'Foreign output owner attribution');generations={fields(line)['slotgen'] for line in lines};require(len(generations)==1 and int(next(iter(generations)))>=int(want['slot']>=0),'Stale native owner generation')
  token_rows=[(int(line.split()[1 if line.startswith('T ') else 2]),stamp) for line,stamp in zip(lines,times) if line.startswith(('T ','BT '))];tokens=[t for t,_ in token_rows];require(tokens==raw['output_ids'] and 0<len(tokens)<=want['max_new'] and all(0<=t<248320 for t in tokens),'Actual complete greedy history differs')
  done=[line for line in lines if line.startswith('DONE ')];require(len(done)==1,'Unique actual admission/serial DONE required');d=done[0].split();require(int(d[2])==2048 and int(d[8])==0 and int(d[14])==2048 and all(math.isfinite(float(d[i])) and float(d[i])>=0 for i in (3,4,13)),'Actual complete fresh prompt/reuse counters invalid')
  if want['slot']>=0:
   terminals=[line for line in lines if line.startswith('BDONE ')];require(len(terminals)==1 and terminals[0]==raw['done'] and int(terminals[0].split()[1])==want['slot'] and int(terminals[0].split()[2])==len(tokens) and int(d[1])==1 and len([line for line in lines if line.startswith('BADM ')])==1,'Exact native slot admission/terminal differs');finish=terminals[0].split()[3]
  else:require(done[0]==raw['done'] and int(d[1])==len(tokens),'Actual serial completion count differs');finish=d[5]
  require(finish in ('length','stop') and (len(tokens)==want['max_new'] if finish=='length' else tokens[-1] in (248044,248046)) and all(t not in (248044,248046) for t in tokens[:-1]),'Natural EOS/cap differs')
  if want['role']=='long':require(len(tokens)>=256,'Actual long output must reach256 tokens')
  token_times=[stamp for _,stamp in token_rows];views.append(dict(role=want['role'],rid=want['rid'],slot=want['slot'],slotgen=int(next(iter(generations))),ids=raw['ids'],output_ids=tokens,finish=finish,send_monotonic_ns=raw['send_monotonic_ns'],first_token_receive_monotonic_seconds=token_times[0],completion_receive_monotonic_seconds=times[-1],TTFT_ms=1000*(token_times[0]-send),completion_ms=1000*(times[-1]-send),native_admission_prompt_ms=float(d[3]),timing=gap_summary(token_times)))
 directory=Path(rows[0]['raw']['phase_directory']);require(all(Path(row['raw']['phase_directory'])==directory for row in rows),'One loaded owner per phase required')
 logs=(directory/'engine.combined.log').read_text(encoding='ascii').splitlines();require(not (directory/'ARM').exists() and not (directory/'NO_CAPTURE_ARM').exists() and not list((directory/'captures').iterdir()),'Raw capture must remain unarmed and empty')
 require(not any(line.startswith(('SBF allocation ','SBF vector ','SBF replay ','SBF row ','SBF span ','FC38 ','SFD ','PCL ','ERR','FATAL')) for line in logs),'Disabled heavy observer/error emitted evidence')
 chain=[fields(line) for line in logs if line.startswith('BCHAIN startup ')];require(len(chain)==1 and chain[0]['slots']=='2' and chain[0]['cap']=='3' and int(chain[0]['required'])<=int(chain[0]['chain_limit'])==8192<<20 and int(chain[0]['transfer_limit'])==512<<20,'Actual bounded state/tier admission required')
 native=expected[0]['slot']>=0;arrival=None;paired=[];window_gaps=None
 if native:
  cohort=json.loads((directory/'cohort-0.json').read_bytes());arrival=cohort['arrival'];require(arrival is not None and len(arrival['long_emitted_ids'])==256 and arrival['long_emitted_ids']==views[0]['output_ids'][:256] and arrival['long_slotgen']==views[0]['slotgen'],'Incoming arrival must bind actual ongoing256-token history')
  trigger=views[0]['timing']['token_receive_monotonic_seconds'][255];incoming=views[1]['send_monotonic_ns']/1e9;require(arrival['trigger_receive_monotonic_seconds']==trigger and trigger<=incoming<views[0]['completion_receive_monotonic_seconds'] and arrival['send_monotonic_ns']==views[1]['send_monotonic_ns'],'Actual incoming send must follow256th receive while first owner is live')
  active=None;active_stamp=None;owner_rows=[]
  def close_window():
   if active and active['rows']=='2':
    require(incoming<=active_stamp<=min(x['completion_receive_monotonic_seconds'] for x in views),'Actual two-owner window must complete while both owners are live')
    require(active['active_mask']=='3' and active['completed']=='1' and len(owner_rows)==2 and {int(line.split()[1]) for line,_ in owner_rows}=={0,1},'Actual two-row event/private token census differs')
    for line,stamp in owner_rows:
     f=fields(line);slot=int(line.split()[1]);require(int(f['rid'])==views[slot]['rid'] and int(f['slotgen'])==views[slot]['slotgen'] and incoming<=stamp<=views[slot]['completion_receive_monotonic_seconds'],'Actual same-window owner identity/overlap differs')
    paired.append(int(active['event']))
  for event in cohort['events']:
   line=event['line'];stamp=event['host_monotonic_seconds']
   if line.startswith('SBF batch_event '):close_window();active=fields(line);active_stamp=stamp;owner_rows=[]
   elif line.startswith('BT '):owner_rows.append((line,stamp))
  close_window()
  a_times=views[0]['timing']['token_receive_monotonic_seconds'];b_end=views[1]['completion_receive_monotonic_seconds'];cross=[1000*(b-a) for a,b in zip(a_times,a_times[1:]) if a<=incoming<=b];during=[1000*(b-a) for a,b in zip(a_times,a_times[1:]) if incoming<=b<=b_end]
  require(cross,'Arrival-straddling long-stream token gap required');window_gaps=dict(arrival_straddling_long_gap_ms=cross[0],long_gaps_while_second_request_active=dict(count=len(during),P50_ms=percentile(during,.5),P95_ms=percentile(during,.95),maximum_ms=max(during) if during else None))
 census=[fields(line) for line in logs if line.startswith('SBF batch_event ')]
 row_counts={str(n):sum(int(x['rows'])==n for x in census) for n in (1,2)}
 require(all(x['completed']=='1' and x['rows'] in ('1','2') and int(x['active_mask']) in (1,2,3) for x in census),'Completed bounded native event census required')
 require(sum(int(x['rows']) for x in census)==sum(line.startswith('BT ') for row in rows for line in row['raw']['stdout']),'Actual completed row/token census differs')
 return dict(observed_completed_window_rows=row_counts,observed_two_row_execution=bool(paired),native2_simultaneous_coherence_qualified=False,streams=views,arrival=arrival,actual_two_row_events=paired,arrival_window_gaps=window_gaps,same_host_monotonic_domain=True,default_scheduler_observed=native,raw_capture_armed=False,full_cache_qualified=False,API_qualified=False,stability_qualified=False)
