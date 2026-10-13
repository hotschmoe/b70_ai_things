"""Tiny real files: authenticated reuse, stat-triggered scan, byte drift rejection."""
import json,hashlib,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
import diagnostic_model_identity_lane_v1 as lane
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
write=lambda p,x:Path(p).write_text(json.dumps(x)+'\n')
c=SimpleNamespace(sha=sha,read=read,write=write)
class Controls(unittest.TestCase):
 def fixture(self,root):
  shards=[root/('tiny%d'%i)for i in range(4)]
  for i,p in enumerate(shards):p.write_bytes(('CPU tiny original%d'%i).encode())
  lock={'revision':'CPU','files':[{'path':'UD-Q4_K_XL/tiny%d'%i,'size':p.stat().st_size,'sha256':sha(p)}for i,p in enumerate(shards)]};lp=root/'lock.json';write(lp,lock);now=time.time()-10
  rows=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),expected_sha256=sha(p),stat_before=lane.stat_fd(p),stat_after=lane.stat_fd(p),passed=True)for p in shards]
  scan=dict(passed=True,lock_sha256=sha(lp),model_revision='CPU',rows=rows,read_mode='ordinary complete buffered read; no direct IO/cache mutation',started=now+2,finished=now+3);write(root/'post-full4.json',scan)
  report=dict(passed=True,post_full4=scan,artifacts={'post-full4.json':sha(root/'post-full4.json')},actual_model_terminal_epoch=now,post_health={'passed':True,'finished_epoch':now+.5},post_before_pages={'epoch':now+1},post_pages={'epoch':now+4},finished_epoch=now+5);write(root/'report.json',report);write(root/'public.json',dict(passed=True,producer_report_sha256=sha(root/'report.json'),root=str(root)))
  prior=dict(root=str(root),public=str(root/'public.json'),report_sha256=sha(root/'report.json'),public_sha256=sha(root/'public.json'),post_full4_sha256=sha(root/'post-full4.json'));return lp,lock,shards,prior
 def test_unchanged_and_replacement_fallback(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);lp,lock,paths,prior=self.fixture(root);calls=[]
   def scanner(lp,lock,shards,out,after):
    calls.append(1);rows=[]
    for p,want in zip(shards,lock['files']):
     before=lane.stat_fd(p);digest=sha(p);rows.append(dict(path=str(p),sha256=digest,expected_sha256=want['sha256'],bytes=p.stat().st_size,stat_before=before,stat_after=lane.stat_fd(p),passed=digest==want['sha256']))
    result=dict(lock_sha256=sha(lp),model_revision=lock['revision'],read_mode='ordinary complete buffered read; no direct IO/cache mutation',passed=all(x['passed']for x in rows),rows=rows);write(out,result);return result
   value=lane.identity(c,lp,lock,paths,root/'first.json',0,scanner,prior);self.assertFalse(value['fresh_model_bytes_read']);self.assertEqual(calls,[])
   temp=root/'replacement';temp.write_bytes(paths[0].read_bytes());temp.replace(paths[0]);value=lane.identity(c,lp,lock,paths,root/'second.json',0,scanner,prior);self.assertTrue(value['fresh_model_bytes_read']);self.assertEqual(len(calls),1)
   paths[0].write_bytes(b'changed bad original')
   with self.assertRaisesRegex(ValueError,'fallback failed'):lane.identity(c,lp,lock,paths,root/'third.json',0,scanner,prior)
   self.assertFalse((root/'third.json').exists())
 def test_tampered_receipt_never_authorizes_scan_shortcut(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);lp,lock,paths,prior=self.fixture(root);(root/'post-full4.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'receipt changed'):lane.identity(c,lp,lock,paths,root/'out.json',0,lambda *x:self.fail('Untrusted prior must refuse'),prior)
 def test_exact_real_owner_and_reader_source_seams(self):
  import run_source62_cached_window_diagnostic_v1 as actor
  from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
  lane.configure('diagnostic',{'CPU':'source construction only'});ns=actor.NS;ports=lane.ports(ns,actor.parent.base.boundary_ports);active=SimpleNamespace(**{**vars(actor),**ns});reader=lane.reader_ports(active,boundary_reader_ports);self.assertEqual(reader['full4'].__name__,'recollect');self.assertIn('diagnostic_identity',ports['main'].__code__.co_names)
if __name__=='__main__':unittest.main()
