#!/usr/bin/env python3
"""Source39 complete actual native submissions including direct GEN and partial cancellation."""
import re
from batch_numerical_prefixes_v2 import token_sha

def command(line):
 words=line.split()
 if not words or words[0] not in ('GEN','BGEN'):return None
 mode=words[0];slot=int(words[1]) if mode=='BGEN' else None;bound=int(words[2 if mode=='BGEN' else 1]);f=dict(re.findall(r'(\w+)=([^ ]+)',line));rid=int(f.get('rid','0'));ids=[int(x) for x in words[-1].split(',')]
 if rid<=0 or not 1<=bound<=64 or not 1<=len(ids)<=2048 or any(not 0<=t<248320 for t in ids):raise ValueError('Actual native submitted shape/identity invalid')
 if mode=='BGEN' and not 0<=slot<6:raise ValueError('Native slot bound invalid')
 return {'mode':mode,'slot':slot,'max_new':bound,'rid':rid,'ids':ids,'generated':[],'slotgen':None,'terminal':False}

def prefix_jobs(events):
 active={};history={};begins={};jobs={};vectors={};stops=[];native_multrow=[]
 for row in events:
  kind=row['kind']
  if kind=='engine_begin':
   call=row['call'];assert call not in begins and row['rendered_matches_submitted'] is True;begins[call]=row
  if kind=='native_send':
   line=row['line'];c=command(line)
   if c:
    key=(row['engine_pid'],c['rid']);past=history.setdefault(key,[])
    if past:
     if not past[-1]['terminal']:raise ValueError('Native continuation before prior segment terminal')
     expected=past[0]['ids']+[t for seg in past for t in seg['generated']]
     if c['ids']!=expected:raise ValueError('Native continuation differs from exact own consumed/generated history')
    if row.get('call') not in begins:raise ValueError('Submitted segment lacks request-local actual API begin')
    source=begins[row['call']]
    if not past and c['ids']!=source['submitted_ids']:raise ValueError('Initial native/API input differs')
    c.update(call=row['call'],engine_pid=row['engine_pid'],send_sequence=row['sequence']);past.append(c);active[key]=c
   if line.startswith(('BSTOP ','STOP','BYIELD ')):stops.append(row)
  if kind!='native_receive':continue
  line=row['line'];f=dict(re.findall(r'(\w+)=([^ ]+)',line));words=line.split()
  if line.startswith('SBF batch_event '):
   if int(f['rows'])>=2 and f['completed']=='1':native_multrow.append(row)
  if line.startswith(('T ','BT ','DONE ','BADM ','BDONE ')):
   rid=int(f.get('rid','0'));generation=int(f.get('slotgen','-1'));seg=active[row['engine_pid'],rid]
   if seg['terminal']:raise ValueError('Late native output on completed segment')
   if seg['slotgen'] is None:seg['slotgen']=generation
   if generation<0 or generation!=seg['slotgen']:raise ValueError('Native generation changed within segment')
   if words[0] in ('T','BT'):
    if words[0]=='BT' and (seg['mode']!='BGEN' or int(words[1])!=seg['slot']):raise ValueError('Foreign native slot token')
    token=int(words[1 if words[0]=='T' else 2]);assert 0<=token<248320;seg['generated'].append(token)
   if words[0]=='DONE':
    if len(words)<16 or int(words[1])!=len(seg['generated']) or int(words[2])!=len(seg['ids']) or not 0<=int(words[8])<=int(words[8])+int(words[14])<=len(seg['ids']) or (words[5]!='cancel' and int(words[8])+int(words[14])!=len(seg['ids'])):raise ValueError('Actual native segment DONE generated/consumed/reused counts differ')
    if seg['mode']=='GEN':seg['terminal']=True
   if words[0]=='BADM':
    if seg['mode']!='BGEN' or int(words[1])!=seg['slot']:raise ValueError('Foreign admission')
    if words[2]=='0':seg['terminal']=True
   if words[0]=='BDONE':
    if seg['mode']!='BGEN' or int(words[1])!=seg['slot'] or int(words[2])!=len(seg['generated']):raise ValueError('Foreign slot terminal/count')
    seg['terminal']=True
  if not line.startswith('SBF vector '):continue
  rid=int(f['rid']);key=(row['engine_pid'],rid);seg=active[key];position=int(f['pos']);token=int(f['token']);take=position+1-len(seg['ids']);ids=seg['ids']+seg['generated'][:max(0,take)]
  if take<0 or take>len(seg['generated']) or len(ids)!=position+1 or ids[-1]!=token:raise ValueError('Raw frame prefix differs from actual native input/history')
  role='direct_gen' if f['phase'].startswith('direct_gen_') else 'solo_migration' if f['phase'].startswith('solo_migration_') else 'admission' if f['phase'].startswith('admission_') else 'later'
  if (role in ('solo_migration','direct_gen'))!=(seg['mode']=='GEN'):raise ValueError('Main frame bound to wrong native segment')
  if role=='direct_gen' and len(history[key])!=1:raise ValueError('Direct first-window field requires first actual GEN segment')
  if role=='solo_migration' and not any(p['mode']=='BGEN' and p['terminal'] for p in history[key][:-1]):raise ValueError('Actual solo migration requires prior native BGEN terminal')
  jobkey=(seg['call'],rid,role);job={'call':seg['call'],'rid':rid,'role':role,'ids':ids,'ids_sha256_le32':token_sha(ids),'position':position,'token':token,'max_new':1,'native_segment_send_sequence':seg['send_sequence'],'original_own_state_math_reference':False}
  if jobkey in jobs and jobs[jobkey]!=job:raise ValueError('Cross-stage/cross-layer API frame prefix differs')
  jobs[jobkey]=job;layer=int(f['layer'])
  if (jobkey,layer) in vectors:raise ValueError('Duplicate API raw layer/head')
  vectors[jobkey,layer]=f
 for key in jobs:
  if set(l for k,l in vectors if k==key)!=set(range(-1,48)):raise ValueError('API selected prefix requires all48/fullhead')
 return {'jobs':list(jobs.values()),'native_segments':{str(k):v for k,v in history.items()},'actual_completed_multirow_events':len(native_multrow),'stop_commands':stops,'all_segments_terminal':all(seg['terminal'] for past in history.values() for seg in past),'full_model_math_qualified':False,'migration_raw_collector_and_serial_comparison_required':True}
