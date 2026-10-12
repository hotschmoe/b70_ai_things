import io,queue,threading,unittest
from types import SimpleNamespace
import original_mtp53_http_compound_v1 as c
class Gates(unittest.TestCase):
 def engine(self):return SimpleNamespace(spawn=(None,None,None,None,{'STRATA_ORIGINAL_MTP53_SERVING':'1'}),ctl=threading.Lock(),slot_cv=threading.Condition(),slot_busy=[False]*4)
 def test_idle_exclusive_and_concurrent_streams(self):
  e=self.engine();cancel=threading.Event()
  with c.stream(e,cancel):
   with c.stream(e,cancel):self.assertEqual(c.gate(e)['active'],2)
   with self.assertRaises(c.CompoundBusy):
    with c.exclusive(e):pass
  with c.exclusive(e):self.assertTrue(c.gate(e)['exclusive'])
  self.assertFalse(c.gate(e)['exclusive']);self.assertEqual(c.gate(e)['active'],0)
 def test_failed_stream_retires(self):
  e=self.engine()
  with self.assertRaises(RuntimeError):
   with c.stream(e,threading.Event()):raise RuntimeError('client disconnect')
  with c.exclusive(e):pass
 def test_legacy_engine_and_independent_pin(self):
  self.assertFalse(c.mode(SimpleNamespace()))
  with self.assertRaises(ValueError):c.session_file(self.engine(),'restore','/tmp/new-compound',None,RuntimeError)
