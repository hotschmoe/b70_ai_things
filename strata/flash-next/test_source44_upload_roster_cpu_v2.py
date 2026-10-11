import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import source44_upload_read_roster_v2 as q

class Controls(unittest.TestCase):
 def tiny(self,root):
  files=[root/'prerequisite.json',root/'external.json',root/'hidden.bin']
  files[0].write_text('{}');files[1].write_text('{}');files[2].write_bytes(b'original hidden raw')
  return sorted(map(str,files))
 def test_hidden_referenced_file_in_explicit_oracle_schema(self):
  import source44_sdk_binding_v1 as sdk
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);engine=root/'sdk';engine.mkdir();(engine/'build').mkdir();(engine/'source').mkdir();out=root/'upload';out.mkdir();oracle=root/'oracle';oracle.mkdir();external=root/'external';external.mkdir()
   def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return str(p)
   pre=write(root/'pre.json',{'upload_receipt':str(out/'report.json'),'oracle_receipt':str(oracle/'receipt.json'),'pack_receipt':str(root/'pack.json')})
   write(root/'pack.json',{'files':{}});write(engine/'receipt.json',{'binary_sha256':{}});write(root/'sdk-plan.json',{'expected_patched_source_sha256':{}});write(engine/'plan.snapshot.json',{});write(engine/'build/CMakeCache.txt',{})
   write(oracle/'receipt.json',{'library_sha256':{}});write(oracle/'plan.snapshot.json',{'source_roster':{'path':str(external/'raw.bin'),'receipt':str(external/'child.json')},'inventory':str(external/'inventory.json')});write(external/'child.json',{});write(external/'inventory.json',{});(external/'raw.bin').write_bytes(b'hidden outside root')
   write(oracle/'oracle.cpp',{});write(oracle/'source-upload-oracle',{});write(out/'report.json',{'cases':[{'case':'case'}]})
   for n in('controller.py','source-page-watchdog.py','post-model-identity.json','case.log','case-logical-free.json'):write(out/n,{})
   real_add=q.path
   # Keep real tiny external JSON discovery; repo source paths are declared
   # dependencies rather than fixture copies. No real SDK/pack/model read occurs.
   def paths(p):
    p=Path(p)
    if p.is_relative_to(root):return real_add(p)
    return real_add(root/'pack.json')
   with patch.object(sdk,'SDK',engine),patch.object(sdk,'PLAN',root/'sdk-plan.json'),patch.object(q,'path',side_effect=paths),patch.object(q,'json_read',side_effect=lambda p:json.loads(Path(p).read_bytes())):
    result=q.roster(pre)
   self.assertIn(str(external/'child.json'),result);self.assertIn(str(external/'inventory.json'),result);self.assertIn(str(external/'raw.bin'),result)
 def run_bracket(self,kind):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);files=self.tiny(root);calls=[]
   def call():
    calls.append(1)
    if kind=='mutation':(root/'hidden.bin').write_bytes(b'changed hidden raw')
    if kind=='failure':raise ValueError('original failed')
    return {'actual_original':True}
   with patch.object(q,'roster',side_effect=[files,files[:-1]]if kind=='omission'else lambda p:files):
    if kind=='foreign':
     with patch.object(q.os,'getpid',side_effect=[os.getpid(),os.getpid(),os.getpid()+1]):
      with self.assertRaises(ValueError):q.bracket(root/'prerequisite.json',call)
    elif kind=='valid':
     result,proof=q.bracket(root/'prerequisite.json',call);self.assertTrue(result['actual_original']);self.assertEqual(proof['actual_successful_original_upload_calls'],1);self.assertEqual(proof['before']['rows'],proof['after']['rows'])
    else:
     with self.assertRaises(ValueError):q.bracket(root/'prerequisite.json',call)
   self.assertEqual(calls,[1])
 def test_one_original_call(self):self.run_bracket('valid')
 def test_hidden_raw_mutation(self):self.run_bracket('mutation')
 def test_omitted_external_reference_on_exit(self):self.run_bracket('omission')
 def test_foreign_pid(self):self.run_bracket('foreign')
 def test_original_failure_retained(self):self.run_bracket('failure')
 def test_symlink_before_read(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);(root/'data').write_text('{}');(root/'alias').symlink_to(root/'data')
   with patch.object(q,'read_unique',side_effect=AssertionError('bytes read')):
    with self.assertRaises(ValueError):q.json_read(root/'alias')
if __name__=='__main__':unittest.main()

# The original six behavioral predicate controls exercise the actual V2
# per-call interceptor with only its external byte-seal boundary mocked.
import test_masked44_upload_view_cpu_v1 as inherited
import qualify_indexer_materialized44_masked_upload_view_v2 as view
class ViewControls(inherited.Controls):
 def run_fixture(self,kind):
  def bracket(prerequisite,call):return call(),{'actual_successful_original_upload_calls':1,'raw_read_roster_bracketed':True}
  with patch.object(inherited,'q',view),patch.object(view,'bracket',side_effect=bracket):super().run_fixture(kind)
