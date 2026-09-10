import copy
import importlib.util
from pathlib import Path
import tempfile
import json
import unittest
HERE=Path(__file__).resolve().parent

def load(name):
 s=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
probe=load('probe');launch=load('launch')

class LongGate(unittest.TestCase):
 def fixture(self):
  item=dict(kind='retrieval',prompt_tokens=185011,max_tokens=128,expected=['A','B','C'])
  row=dict(error=None,text='["A","B","C"]',reasoning='',client_cancelled=False,finish_reason='stop',usage=dict(prompt_tokens=185011,completion_tokens=10,prompt_tokens_details=dict(cached_tokens=184000)))
  return item,row
 def test_exact_secret_tokenizer_and_cache(self):
  item,row=self.fixture();probe.validate(row,item,reuse=True)
  for field,value in [('text','["A","B","D"]'),('finish_reason','length')]:
   changed=copy.deepcopy(row);changed[field]=value
   with self.assertRaises(AssertionError):probe.validate(changed,item,reuse=True)
  for field,value in [('prompt_tokens',185010),('prompt_tokens_details',{'cached_tokens':0})]:
   changed=copy.deepcopy(row);changed['usage'][field]=value
   with self.assertRaises(AssertionError):probe.validate(changed,item,reuse=True)
 def test_cancel_and_repetition(self):
  item,row=self.fixture();row.update(client_cancelled=True,finish_reason=None,usage={})
  probe.validate(row,item,cancel=True)
  row['client_cancelled']=False
  with self.assertRaises(AssertionError):probe.validate(row,item,cancel=True)
  row['client_cancelled']=True;row['text']='!'*64
  with self.assertRaises(AssertionError):probe.validate(row,item,cancel=True)
 def test_prerequisite_refuses_unqualified(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'previous').mkdir();(root/'prerequisite.json').write_text(json.dumps(dict(run=str(root/'previous'),lifecycle=str(root/'rc'))))
   with self.assertRaises(AssertionError):launch.validate(root)

if __name__=='__main__':unittest.main()
