import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import probe_cancellation as probe
CORPUS=Path('/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/corpus-v2-length/manifest.json')
class Cancellation(unittest.TestCase):
 def test_actual_parser_cancel_point_and_missing_usage(self):
  frames=[{'choices':[{'index':0,'delta':{'content':str(i)},'finish_reason':None}]} for i in range(6)]
  data=b''.join(b'data: '+json.dumps(f).encode()+b'\n\n' for f in frames)
  with tempfile.TemporaryDirectory() as tmp,patch.object(probe.urllib.request,'urlopen',return_value=io.BytesIO(data)):
   path=Path(tmp)/'sse.jsonl';row=probe.stream('http://unused',{},path,cancel_chunks=3)
   self.assertTrue(row['client_cancelled']);self.assertIsNone(row['error']);self.assertEqual(row['response']['choices'][0]['message']['content'],'012')
   self.assertEqual(row['cancellation_point']['content_chunks'],3);self.assertIsNone(row['cancellation_point']['token_count']);self.assertFalse(row['cancellation_point']['usage_available']);self.assertEqual(len(path.read_text().splitlines()),3)
 def test_idle_requires_both_gauges_and_zero(self):
  self.assertFalse(probe.idle_gauges('vllm:num_requests_running 0\n')[0])
  self.assertFalse(probe.idle_gauges('vllm:num_requests_running 0\nvllm:num_requests_waiting 1\n')[0])
  self.assertTrue(probe.idle_gauges('vllm:num_requests_running{engine="0"} 0.0\nvllm:num_requests_waiting{engine="0"} 0\n')[0])
 def test_failed_drain_prevents_recovery_and_pass_keeps_exact_branches(self):
  for drained in [False,True]:
   calls=[]
   def fake_stream(base,payload,path,cancel_chunks=None,timeout=180):
    calls.append(payload)
    if cancel_chunks:return dict(error=None,client_cancelled=True,response={'choices':[{'message':{'content':'[1,2'},'finish_reason':None}]})
    expected=list(range(513,769)) if '513 through 768' in payload['messages'][0]['content'] else list(range(1,513))
    return dict(error=None,client_cancelled=False,response={'choices':[{'message':{'content':json.dumps(expected)},'finish_reason':'stop'}],'usage':{'prompt_tokens':4400,'completion_tokens':500,'prompt_tokens_details':{'cached_tokens':1600}}})
   with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)/'job';argv=['probe','--model','hotschmoe-dd','--corpus',str(CORPUS),'--out',str(out)]
    with patch.object(sys,'argv',argv),patch.object(probe,'request',return_value=json.dumps({'data':[{'id':'hotschmoe-dd'}]})),patch.object(probe,'stream',fake_stream),patch.object(probe,'drain',return_value=drained):self.assertEqual(probe.main(),0 if drained else 1)
    self.assertEqual(len(calls),5 if drained else 3)
    self.assertEqual([p['max_tokens'] for p in calls],[4096,2048,4096,4096,2048] if drained else [4096,2048,4096])
    self.assertEqual(calls[0]['cache_salt'],calls[1]['cache_salt']);self.assertNotEqual(calls[0]['cache_salt'],calls[2]['cache_salt'])
    if drained:self.assertEqual(calls[2],calls[3]);self.assertEqual(calls[3]['cache_salt'],calls[4]['cache_salt'])
if __name__=='__main__':unittest.main()
