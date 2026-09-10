#!/usr/bin/env python3
"""Freeze explicit diagnostic requests, never infer or execute generated tools."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from replay_pi_history import TOOLS,text_blocks

def history(records,cutoff):
    messages=[dict(role='system',content='You are a coding assistant. Continue the user task using the available tools.')];calls=set()
    for record in records[:cutoff-1]:
        m=record.get('message',{});role=m.get('role')
        if role=='assistant' and m.get('stopReason') in ('stop','toolUse'):
            row=dict(role='assistant',content=text_blocks(m.get('content',[])) or None)
            tools=[dict(id=b['id'],type='function',function=dict(name=b['name'],arguments=json.dumps(b['arguments'],ensure_ascii=True))) for b in m.get('content',[]) if b['type']=='toolCall']
            if tools:row['tool_calls']=tools;calls.update(t['id'] for t in tools)
            messages.append(row)
        elif role=='toolResult':
            assert m['toolCallId'] in calls,'refuse orphan introduced by reconstruction'
            messages.append(dict(role='tool',tool_call_id=m['toolCallId'],content=text_blocks(m.get('content',[]))))
        elif role=='user':messages.append(dict(role='user',content=text_blocks(m.get('content',[]))))
    return messages

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    case=next(c for c in json.loads(a.cases.read_text())['cases'] if c['case_id']=='agent1_earliest_garble');source=Path(case['source_path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==case['source_sha256'];records=[json.loads(l) for l in source.read_text().splitlines()]
    rows=[]
    def save(name,kind,payload,group,hit=False,expected=None):
        path=a.out/(name+'.json');raw=json.dumps(payload,ensure_ascii=True,indent=2)+'\n';path.write_text(raw)
        rows.append(dict(id=name,kind=kind,payload=str(path.resolve()),sha256=hashlib.sha256(raw.encode()).hexdigest(),cache_group=group,require_hit=hit,expected=expected))
    for name,cutoff,hit in [('pi-agent1-warm',20,False),('pi-agent1-target0',22,False),('pi-agent1-target1',22,True)]:
        save(name,'reconstruction',dict(messages=history(records,cutoff),tools=TOOLS,temperature=0,seed=42,max_tokens=2048,chat_template_kwargs=dict(enable_thinking=True,reasoning_effort='medium')),'pi-agent1',hit)
    for limit in [64,256,1024,2048]:
        n=limit//4
        prompt=('Background archive: records describe routine warehouse inventory checks. These are context, not requested output.\n'+('The warehouse stores parts and maintains an inventory ledger.\n'*360)+f'\nTASK: Return only one JSON array containing every integer from 1 through {n} inclusive, in ascending order. Do not skip, duplicate or summarize numbers. No markdown or explanation. Finish after the closing bracket.')
        payload=dict(messages=[dict(role='user',content=prompt)],temperature=0,seed=42,max_tokens=limit,chat_template_kwargs=dict(enable_thinking=False))
        for repeat in [0,1]:save(f'array-cap{limit}-repeat{repeat}','exact_array',payload,f'array-{limit}',bool(repeat),list(range(1,n+1)))
    result=dict(cases=rows,original_case=case,limitations=['Controlled reconstruction, NOT exact Pi wire payload: original system/tools/body/sampling/cache_salt unknown.','Failed record22 and later excluded; earlier incomplete-stream record7 skipped; no orphan results introduced.','Reconstruction thinking=on/medium follows stored session level but forwarding unknown; t0/seed42/max2048 are explicit diagnostic controls.','Clean length tasks explicitly thinkingOFF/t0/seed42. Caps64/256/1024/2048 do not guarantee those actual output lengths; actual usage retained.','Reconstructed output structural checks require manual semantic review; clean arrays checked exactly.','No generated tool or code is executed. All cases are collected after content failures; final quality return remains nonzero. Cache reuse gate applies to repeated requests.'])
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(a.out/'manifest.json')
if __name__=='__main__':main()
