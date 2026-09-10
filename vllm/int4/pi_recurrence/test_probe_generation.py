import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('recurrence_generation',HERE/'probe_generation.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
CORPUS=Path('/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/corpus/manifest.json')

class Generation(unittest.TestCase):
 def test_exact_semantics_and_cache_gate(self):
  item={'kind':'exact_array','expected':[1,2,3]}
  response={'choices':[{'message':{'content':'[1,2,3]'},'finish_reason':'stop'}],'usage':{'prompt_tokens':4500,'completion_tokens':7,'prompt_tokens_details':{'cached_tokens':1600}}}
  probe.check(response,item,True)
  for text in ['[1,2,4]','[true,2,3]','[1,2,3,3]']:
   wrong=copy.deepcopy(response);wrong['choices'][0]['message']['content']=text
   with self.assertRaises(AssertionError):probe.check(wrong,item,True)
  wrong=copy.deepcopy(response);wrong['usage']['prompt_tokens_details']['cached_tokens']=0
  with self.assertRaises(AssertionError):probe.check(wrong,item,True)
  wrong=copy.deepcopy(response);wrong['choices'][0]['finish_reason']='length'
  with self.assertRaises(AssertionError):probe.check(wrong,item)
 def test_collects_all_cases_after_initial_failure(self):
  cases=json.loads(CORPUS.read_text())['cases'];calls=[]
  def stream(base,payload,timeout,trace,bang_limit):
   item=cases[len(calls)];calls.append(copy.deepcopy(payload))
   if len(calls)==1:raise RuntimeError('synthetic initial content failure')
   text=json.dumps(item['expected']) if item['kind']=='exact_array' else 'I will continue the requested task.'
   return {'choices':[{'message':{'content':text},'finish_reason':'stop'}],'usage':{'prompt_tokens':4500,'completion_tokens':len(text),'prompt_tokens_details':{'cached_tokens':1600}}}
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'generation'
   argv=['probe','--model','hotschmoe-dd','--corpus',str(CORPUS),'--out',str(out)]
   with patch.object(sys,'argv',argv),patch.object(probe,'request',return_value=json.dumps({'data':[{'id':'hotschmoe-dd'}]})),patch.object(probe,'stream_request',stream):self.assertEqual(probe.main(),1)
   self.assertEqual(len(calls),11)
   summary=json.loads((out/'summary.json').read_text());self.assertFalse(summary['passed']);self.assertEqual(summary['requests'],11)
   self.assertEqual([c['max_tokens'] for c in calls[3:]],[64,64,256,256,1024,1024,2048,2048])
   self.assertTrue(all(c['temperature']==0 and c['seed']==42 for c in calls))
   self.assertTrue(all(c['chat_template_kwargs']['enable_thinking'] is False for c in calls[3:]))
   self.assertEqual(calls[3]['cache_salt'],calls[4]['cache_salt'])

if __name__=='__main__':unittest.main()
