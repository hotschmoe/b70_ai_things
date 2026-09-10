#!/usr/bin/env python3
"""CPU parser controls distinguish EOS metadata, string stops, and truncation."""
import json
from probe_stop_tokens import diagnostic_request, parse_sse


def wire(events, done=True):
    return (''.join('data: '+json.dumps(e)+'\n\n' for e in events)+('data: [DONE]\n\n' if done else '')).encode()


request=dict(stream=True,model='hotschmoe-dd',temperature=0,seed=42,max_tokens=4096,messages=[{'role':'user','content':'test'}])
candidate=diagnostic_request(request)
assert set(candidate)-set(request)=={'return_token_ids'} and all(candidate[k]==v for k,v in request.items())
base={'id':'test','choices':[{'index':0,'delta':{'content':'432'},'token_ids':[123],'finish_reason':None}]}
for reason,ids in [(None,[248046]),(248044,[248044]),('STOP',[42])]:
    terminal={'id':'test','choices':[{'index':0,'delta':{},'token_ids':ids,'finish_reason':'stop','stop_reason':reason}]}
    result=parse_sse(wire([base,terminal]))
    assert result['observation_complete'] and result['content']=='432'
    assert result['token_ids']==[123]+ids and result['finishes'][0]['stop_reason']==reason
missing=parse_sse(wire([{'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]}]))
assert missing['token_metadata_chunks']==0 and missing['last_token_ids']==[]
assert not parse_sse(wire([base],False))['observation_complete']
print('PASS request-only metadata delta; EOS/additional stop/string stop retention; omitted token metadata and truncated stream remain distinguishable')
