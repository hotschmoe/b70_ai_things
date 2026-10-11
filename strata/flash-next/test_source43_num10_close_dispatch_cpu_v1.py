"""Execute actual inherited public closing dispatch on tiny controlled metadata."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import validate_source43_num10_zero_v3 as r
class Controls(unittest.TestCase):
 def exercise(self,failed):
  namespace=r.derived();events=[]
  def entry(root):
   events.append('entry')
   if failed:raise RuntimeError('actual entry failure')
   return {'passed':True}
  namespace['_finalized_binding']=entry
  def scan(lock_path,lock,shards,path,after):
   events.append('close');self.assertIs(type(after),float);self.assertEqual(len(shards),4);Path(path).write_text('{}');return {'passed':True}
  with tempfile.TemporaryDirectory()as t,patch.object(r.c,'read',return_value={'destination':'CPU_SYNTHETIC_NONMODEL','files':[{'path':'UD-Q4_K_XL/shard'+str(i)}for i in range(4)]}),patch.object(r.a.life.q.upload_v2,'full_buffered_identity',side_effect=scan),patch.object(r.a.intake,'guard',side_effect=lambda *a:events.append('pages')):
   if failed:
    with self.assertRaisesRegex(RuntimeError,'actual entry failure'):namespace['finalized_binding'](Path(t))
   else:self.assertTrue(namespace['finalized_binding'](Path(t))['current_public_full4_rehashed'])
  self.assertEqual(events,['entry','close','pages'])
 def test_real_public_success_closes(self):self.exercise(False)
 def test_real_public_failed_entry_still_closes(self):self.exercise(True)
 def test_all_inherited_actor_module_attributes_exist(self):
  import ast
  source=Path(r.original.__file__).read_text();attributes={n.attr for n in ast.walk(ast.parse(source))if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='a'}
  self.assertTrue(all(hasattr(r.a,name)for name in attributes),sorted(name for name in attributes if not hasattr(r.a,name)))
if __name__=='__main__':unittest.main()
