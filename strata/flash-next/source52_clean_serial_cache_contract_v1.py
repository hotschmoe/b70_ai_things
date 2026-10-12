"""Actual native DONE/RESUME/REUSED protocol only; no SFD/PCL authority."""
import math,re
from serial37_canonical_json_v3 import canonical
def require(ok,msg):
 if not ok:raise ValueError(msg)
def requests(rows,expected):
 require(type(rows)is list and len(rows)==len(expected)==6,'Six original requests required');views=[]
 for row,want in zip(rows,expected):
  raw=row['raw'];require(type(row['ordinal'])is int and row['ordinal']==want['ordinal']and raw['ids']==want['ids']and type(raw['fresh'])is int and raw['fresh']==want['fresh']and type(raw['pin'])is int and raw['pin']==1536 and raw['stop_sent']is False and raw['cancel_requested']is None,'Exact clean native request differs');require(raw['command']=='GEN 64 fresh=%d pin=1536 ckpt=1 temperature=0 '%want['fresh']+' '.join(map(str,want['ids'])),'Exact LP/seed-omitted GEN64 command differs');lines=raw['stdout'];require(type(lines)is list and all(type(line)is str for line in lines)and not any(line.startswith(('ERR','FATAL','LP '))for line in lines),'Native error/logprobs emitted');done=lines[-1].split();require(done==raw['done'].split()and len(done)==16 and done[0]=='DONE'and all(re.fullmatch('[0-9]+',done[i])for i in(1,2,6,7,8,9,10,11,12,14,15)),'Exact actual52 native DONE16 schema required');ids=[]
  for line in lines:
   if line.startswith('T '):
    fields=line.split();require(len(fields)==2 and fields[1].isdigit()and 0<=int(fields[1])<248320,'Actual native T shape/vocabulary differs');ids.append(int(fields[1]))
  reused=want['expected_reused'];work=len(want['ids'])-reused;require(int(done[1])==len(ids)==len(raw['output_ids'])and ids==raw['output_ids']and 0<len(ids)<=64 and int(done[2])==len(want['ids'])and int(done[8])==reused and int(done[14])==work,'Native output/input/reused/read_n counters differ');require(done[5]in('length','stop')and(done[5]!='length'or len(ids)==64)and(done[5]!='stop'or ids[-1]in(248044,248046))and all(i not in(248044,248046)for i in ids[:-1]),'Natural terminal/cap64/EOS differs');require(not raw['LP']and [line for line in lines if line.startswith('REUSED ')]==['REUSED '+str(reused)]and [line for line in lines if line.startswith('RESUME ')]==['RESUME '+str(reused)],'Native reuse markers disagree with DONE');require(all(math.isfinite(float(done[i]))and float(done[i])>=0 for i in(3,4,13)),'Native host timing/bytes nonfinite');require(all(int(done[i])==0 for i in(6,7,9,10,15)),'No-MTP/no-lookup/offload counters differ');times=raw['stdout_seconds'];require(len(times)==len(lines)and all(type(t)in(int,float)and math.isfinite(t)and t>=0 for t in times)and times==sorted(times),'Original monotonic receive timestamps differ');require(not any(line.startswith(('SFD ','PCL ','PDIAG '))for line in raw['stderr']),'Disabled diagnostic emitted native records')
  token_times=[t for line,t in zip(lines,times)if line.startswith('T ')];views.append({'ids':raw['ids'],'output_ids':ids,'finish':done[5],'actual_reused':reused,'new_prompt_rows':work,'first_token_receive_seconds':token_times[0],'completion_receive_seconds':times[-1],'token_gaps_seconds':[b-a for a,b in zip(token_times,token_times[1:])],'native_prompt_ms':float(done[3]),'native_decode_ms':float(done[4]),'native_read_n':int(done[14]),'native_resume_count':int(done[8])})
 for indices in((0,1,4),(2,3,5)):
  first=views[indices[0]]
  for i in indices[1:]:require(views[i]['output_ids']==first['output_ids']and views[i]['finish']==first['finish'],'Clean within-role repeats differ')
 return views

def equivalence(left,right):
 require(len(left)==len(right)==6,'Both complete clean arms required')
 for a,b in zip(left,right):require(a['ids']==b['ids']and a['output_ids']==b['output_ids']and a['finish']==b['finish'],'Clean cold/hit output/finish differs')
 return {'bounded_basic_cache_passed':True,'full_cache_qualified':False,'capture_free_native_protocol':True,'rows':[{'cold_ttft_seconds':a['first_token_receive_seconds'],'cached_ttft_seconds':b['first_token_receive_seconds'],'actual_reused':b['actual_reused'],'new_prompt_rows':b['new_prompt_rows']}for a,b in zip(left,right)]}
