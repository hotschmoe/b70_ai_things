#!/usr/bin/env python3
"""Audit frozen Pi session records, never execute their content.

Counts assistant/user/tool-result bang spans separately. Other repetition flags
are deliberately narrow and miss short gibberish; they are not semantic quality
certification. Stored timestamps and context are not exact wire requests or
abort arrival times. Historical worlds remain separate; no image attribution.
"""
from pathlib import Path
import json,re,hashlib,collections
from datetime import datetime,timezone
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
BUNDLE=args.bundle.resolve()
OUTPUT=args.output.resolve()
OUTPUT.mkdir(parents=True,exist_ok=True)
def walk(x,path='content'):
 if isinstance(x,str):yield path,x
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from walk(v,path+'.'+str(i))
 elif isinstance(x,dict):
  for k,v in x.items():yield from walk(v,path+'.'+k)
def utc(ms):
 if isinstance(ms,(int,float)):return datetime.fromtimestamp(ms/1000,timezone.utc).isoformat()
 return ms
worlds={};sessions=[];events=[]
for world in sorted(p for p in BUNDLE.iterdir() if p.is_dir()):
 counts=collections.Counter();stamps=[]
 for p in sorted(world.glob('*/sessions/*.jsonl')):
  stats=collections.Counter();seen=[];previous_issues=[];rows=[]
  for lineno,line in enumerate(p.read_text().splitlines(),1):
   record=json.loads(line);m=record.get('message',{});role=m.get('role')
   if not role:continue
   stamp=m.get('timestamp',record.get('timestamp'));stamps.append(utc(stamp));stats[role]+=1
   fields=list(walk(m.get('content',[])));bangs=[];repetitions=[]
   for field,s in fields:
    decoded=re.sub(r'\\u0021','!',s,flags=re.I)
    match=re.search(r'!{32,}',decoded)
    if match:bangs.append(dict(field=field,offset=match.start(),run=len(match[0]),field_chars=len(decoded),prefix=decoded[max(0,match.start()-50):match.start()]))
    if role=='assistant':
     extreme=re.search(r'([^\s])\1{1023,}',s)
     if extreme and extreme[1]!='!':repetitions.append(dict(field=field,kind='nonwhitespace_char1024',unit=extreme[1],count=len(extreme[0])))
     # Bounded scalar unit pattern, do not print the generated text.
     lines=s.splitlines();best=0;run=0;last=None
     for item in lines:
      item=item.strip();run=run+1 if item and item==last else (1 if item else 0);last=item;best=max(best,run)
     if best>=32:repetitions.append(dict(field=field,kind='repeated_line32',count=best))
   error=m.get('errorMessage') or '';is429=role=='assistant' and '429' in error
   if role=='assistant':
    stats['stop_'+str(m.get('stopReason'))]+=1
    stats['local429']+=is429
    if error and not is429:stats['other_assistant_errors']+=1
   if bangs:stats[role+'_bang_messages']+=1
   if repetitions:stats['other_repetition_messages']+=1
   if bangs or repetitions or (role=='assistant' and error):
    event=dict(world=world.name,file=str(p.relative_to(BUNDLE)),line=lineno,id=record.get('id'),parentId=record.get('parentId'),role=role,timestamp=stamp,utc=utc(stamp),stopReason=m.get('stopReason'),rawStopReason=m.get('rawStopReason'),responseId=m.get('responseId'),usage=m.get('usage'),bang_fields=bangs,repetition_fields=repetitions,error=error[:240],is429=is429,preceding_assistant_issues=len(previous_issues),first_prior_assistant_issue=previous_issues[0] if previous_issues else None)
    events.append(event);rows.append(event)
    if role=='assistant' and (bangs or repetitions):previous_issues.append(dict(line=lineno,id=record.get('id'),utc=utc(stamp),bang=bool(bangs)))
  counts.update(stats)
  sessions.append(dict(world=world.name,file=str(p.relative_to(BUNDLE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),counts=dict(stats),first_assistant_issue=next((e for e in rows if e['role']=='assistant' and (e['bang_fields'] or e['repetition_fields'])),None)))
 worlds[world.name]=dict(counts=dict(counts),min_timestamp=min(stamps) if stamps else None,max_timestamp=max(stamps) if stamps else None,sessions=sum(s['world']==world.name for s in sessions))
result=dict(worlds=worlds,sessions=sessions,events=sorted(events,key=lambda e:str(e['utc'])),limitations=['Timestamps are stored message/request starts, not exact guard abort arrival.','Strict repetition flags are not exhaustive semantic review.','Session history is not wire payload; context extensions can filter it.','File snapshots are not simultaneous.','Assistant bang fields require manual intent review; toolResult bangs are counted separately.'])
(OUTPUT/'audit.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n')
print(json.dumps(worlds,indent=2,ensure_ascii=True))
print('FIRST CURRENT ASSISTANT ISSUES')
for e in [e for e in result['events'] if e['world']=='current-world-5' and e['role']=='assistant' and (e['bang_fields'] or e['repetition_fields'])][:10]:print(json.dumps(e,ensure_ascii=True))
