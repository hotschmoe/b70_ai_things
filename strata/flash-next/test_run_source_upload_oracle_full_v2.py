#!/usr/bin/env python3
"""CPU tiny-file controls only; no actual model or GPU execution."""
import ast,copy,hashlib,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import run_source_upload_oracle_full_v2 as r

class RunnerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.shards=[];self.files=[]
  for i in range(4):
   path=self.root/('fixture-%d.gguf'%i);data=bytes([i+1])*8192;path.write_bytes(data);self.shards.append(path);self.files.append({'path':'UD-Q4_K_XL/'+path.name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  self.lock={'files':self.files,'revision':'CPU_mock_only'};self.lockpath=self.root/'mock-lock.json';self.lockpath.write_text(json.dumps(self.lock))
 def scan(self):return r.full_buffered_identity(self.lockpath,self.lock,self.shards,self.root/'identity.json',time.time()-1)
 def test_complete4_positive(self):
  result=self.scan();self.assertTrue(result['passed']);self.assertEqual(len(result['rows']),4);self.assertTrue(all(row['stat_before']==row['stat_after'] for row in result['rows']))
 def test_badbytes_still_complete4_failure(self):
  self.shards[2].write_bytes(b'X'*8192);result=self.scan();self.assertFalse(result['passed']);self.assertEqual(len(result['rows']),4);self.assertFalse(result['rows'][2]['passed'])
 def test_unchanged_stat_not_pass_for_wrong_expected(self):
  self.lock['files'][2]['sha256']='0'*64;self.assertFalse(self.scan()['passed'])
 def test_before_boundary_rejected(self):
  with self.assertRaises(AssertionError):r.full_buffered_identity(self.lockpath,self.lock,self.shards,self.root/'identity.json',time.time()+60)
 def test_both_exact_failure_views_preserved(self):
  a=b'3'*4096;b=b'4'*4096;path=self.shards[2];path.write_bytes(a+b)
  pages=((0,hashlib.sha256(a).hexdigest()),(4096,'0'*64))
  with patch.object(r.watchdog,'KNOWN_PAGES',pages):
   with self.assertRaises(ValueError):r.guarded_pages(self.shards,self.root,'CPU_mock_poll')
  rows=json.loads(next(self.root.glob('*source-pages.json')).read_text());self.assertFalse(rows['passed']);self.assertEqual(len(rows['rows']),2)
  self.assertEqual([Path(row['preserved_path']).read_bytes() for row in rows['rows']],[a,b]);self.assertEqual(rows['stat_before'],rows['stat_after'])
 def test_good_pages_no_failure_files(self):
  data=self.shards[2].read_bytes();pages=((0,hashlib.sha256(data[:4096]).hexdigest()),(4096,hashlib.sha256(data[4096:]).hexdigest()))
  with patch.object(r.watchdog,'KNOWN_PAGES',pages):self.assertTrue(r.guarded_pages(self.shards,self.root,'mock')['passed'])
  self.assertEqual(list(self.root.glob('*source-pages.json')),[])
 def test_five_cases_config_and_commands_unchanged(self):
  old=ast.parse(Path(r.__file__).with_name('run_source_upload_oracle_full.py').read_text());new=ast.parse(Path(r.__file__).read_text())
  funcs=lambda tree:{node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
  self.assertEqual(ast.dump(funcs(old)['full_cases']),ast.dump(funcs(new)['full_cases']))
  assignments=lambda tree:{name:ast.dump(node.value) for node in ast.walk(funcs(tree)['main']) if isinstance(node,ast.Assign) for target in node.targets if isinstance(target,ast.Name) for name in [target.id] if name in ('args','cmd')}
  self.assertEqual(assignments(old),assignments(new))
 def test_final_requires_posthash_health_terminal_all5(self):
  base={'expected_cases':list(range(5)),'cases':list(range(5)),'post_health_passed':True,'owned_terminal':True,'post_model_identity':{'passed':True},'post_identity_complete4':True}
  self.assertTrue(r.final_pass(base))
  for key in ('post_health_passed','owned_terminal','post_model_identity','post_identity_complete4'):
   v=copy.deepcopy(base);v.pop(key);self.assertFalse(r.final_pass(v),key)
  v=copy.deepcopy(base);v['cases'].pop();self.assertFalse(r.final_pass(v))
  for key in ('error','cleanup_error','post_health_error','post_identity_error'):
   v=copy.deepcopy(base);v[key]='fault';self.assertFalse(r.final_pass(v),key)
 def test_poll_has_two_page_guard_and_bounded_scope(self):
  source=Path(r.__file__).read_text();self.assertIn("guarded_pages(shards, out, label + '-poll-'",source);self.assertIn('source_guard_between_polls_unobserved=True',source)
 def test_watchdog_pin(self):self.assertEqual(r.sha(Path(r.watchdog.__file__)),r.WATCHDOG_SHA)

if __name__=='__main__':unittest.main()
