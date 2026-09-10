import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import probe_concurrent_generation as p

class Concurrent(unittest.TestCase):
 def test_four_dispatches_content_failure_collects_and_salts_reuse(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);phases=[];seen=[];lock=threading.Lock();active=0;peak=0
   for repeat in [0,1]:
    cases=[]
    for i in range(4):
     path=out/f'p{i}.json';path.write_text(json.dumps({'messages':[],'max_tokens':2048}))
     cases.append(dict(id=f'r{repeat}s{i}',payload=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind='exact_array',expected=[1,2],require_hit=bool(repeat),cache_group=f's{i}'))
    phases.append(dict(id=str(repeat),cases=cases))
   def fake(base,payload,timeout,trace,bang_limit):
    nonlocal active,peak
    with lock:active+=1;peak=max(peak,active);seen.append(payload['cache_salt'])
    time.sleep(.04)
    with lock:active-=1
    if trace.name.startswith('r0s0'):raise ValueError('content failure')
    return {'choices':[{'message':{'content':'[1,2]'},'finish_reason':'stop'}],'usage':{'completion_tokens':5,'prompt_tokens':4000,'prompt_tokens_details':{'cached_tokens':1600}}}
   with patch.object(p,'stream_request',fake):rows=p.execute('unused','hotschmoe-dd',{'phases':phases},out)
   self.assertEqual(len(rows),8);self.assertEqual(sum(r['passed'] for r in rows),7);self.assertEqual(peak,4);self.assertEqual(len(set(seen)),4);self.assertTrue(all(seen.count(s)==2 for s in set(seen)))
if __name__=='__main__':unittest.main()
