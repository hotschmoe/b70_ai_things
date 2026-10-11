"""Actual tracing functions with a controlled serialized native stand-in."""
import ast,json,struct,hashlib,threading,time,types,unittest
from pathlib import Path
from c1_trace_contract import completion_facts,trace_end_accepted
class Controls(unittest.TestCase):
 def test_two_wrappers_overlap_but_done_owned_perthread(self):
  path=Path(__file__).with_name('c1_api_trace43_concurrent_v1.py');nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('token_sha','parse_done','generate')];records=[];native_lock=threading.Lock();arrivals=threading.Barrier(2)
  engine=types.SimpleNamespace(last={},proc=types.SimpleNamespace(pid=1))
  def parse(engine,line):engine.last={'generated':1,'prompt_read':len(line),'reused':0,'prompt_tokens':len(line),'finish':'length'}
  def native(self,ids,max_new,sampling,cancel,embeddings):
   arrivals.wait(timeout=3)
   with native_lock:
    ns['parse_done'](self,ids);yield ids[0]
   # Allow the other actual native stand-in to replace shared self.last before
   # wrapper finally executes. Thread-owned parse snapshot must remain correct.
   time.sleep(.03)
  ns={'hashlib':hashlib,'struct':struct,'threading':threading,'time':time,'LOCK':threading.Lock(),'LOCAL':threading.local(),'COUNTER':0,'parse_done_original':parse,'original':native,'EOS_IDS':[99],'completion_facts':completion_facts,'emit':lambda row:records.append(row)}
  exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),'[actual-tracer-threadlocal-body]','exec'),ns)
  results=[]
  def caller(ids):ns['LOCAL'].prompt={'ids':ids};results.append(list(ns['generate'](engine,ids,64,{},threading.Event())))
  threads=[threading.Thread(target=caller,args=(ids,))for ids in([7],[8,9])]
  for t in threads:t.start()
  for t in threads:t.join(timeout=5);self.assertFalse(t.is_alive())
  ends=[r for r in records if r['kind']=='engine_end'];self.assertEqual(len(ends),2);self.assertTrue(all(trace_end_accepted(r)for r in ends));self.assertEqual({r['engine_done']['prompt_tokens']for r in ends},{1,2});self.assertEqual({r['call']for r in ends},{1,2});self.assertEqual(sorted(results),[[7],[8]])
 def test_native_delegate_and_token_yield_body_remain_unchanged(self):
  h=Path(__file__).parent;one=ast.parse((h/'c1_api_trace.py').read_text());two=ast.parse((h/'c1_api_trace43_concurrent_v1.py').read_text())
  for name in('encode','close','emit','token_sha'):
   a=next(n for n in one.body if isinstance(n,ast.FunctionDef)and n.name==name);b=next(n for n in two.body if isinstance(n,ast.FunctionDef)and n.name==name);self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
if __name__=='__main__':unittest.main()
