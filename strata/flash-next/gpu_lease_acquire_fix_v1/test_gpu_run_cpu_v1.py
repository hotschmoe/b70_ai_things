"""Real flock/process/signal controls on temporary CPU-only lease files."""
import fcntl,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
CANDIDATE=HERE/'gpu-run-candidate.sh'
ORIGINAL=HERE/'gpu-run-original.snapshot.sh'

class Test(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.base=self.root/'cpu-only-lock';self.children=[];self.holders=[]
 def tearDown(self):
  for p in self.children:
   if p.poll() is None:p.kill()
   try:p.communicate(timeout=3)
   except subprocess.TimeoutExpired:pass
  for f in self.holders:f.close()
  self.tmp.cleanup()
 def hold(self,card):
  f=open(str(self.base)+'.'+str(card),'w');fcntl.flock(f,fcntl.LOCK_EX);self.holders.append(f);owner=Path(str(self.base)+'.'+str(card)+'.owner');owner.write_text('UNRELATED_CPU_HOLDER\n');return owner
 def launch(self,wrapper=CANDIDATE,card=None,timeout='0',command=None):
  env={**os.environ,'B70_GPU_LOCK':str(self.base),'B70_GPU_LOCK_TIMEOUT':timeout,'B70_AGENT':'CPU_TEST'}
  args=['bash',str(wrapper)]+([] if card is None else ['--card',str(card)])
  marker=self.root/'EXECUTED';args+=command or [sys.executable,'-c','from pathlib import Path;import sys;Path(sys.argv[1]).write_text("executed")',str(marker)]
  p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);self.children.append(p);return p,marker
 def waiter(self,p):
  until=time.monotonic()+3
  while time.monotonic()<until:
   if p.poll() is not None:self.fail('Wrapper exited before blocked waiter: '+str(p.returncode))
   children=Path('/proc/%d/task/%d/children'%(p.pid,p.pid)).read_text().split()
   for pid in children:
    try:
     argv=Path('/proc/'+pid+'/cmdline').read_bytes().split(b'\0')
     if Path('/proc/'+pid+'/comm').read_text().strip()=='flock' and len(argv)>=3 and argv[-2] in (b'8',b'9') and b'-n' not in argv:return int(pid)
    except (FileNotFoundError,ProcessLookupError):pass
   time.sleep(.01)
  self.fail('No actual queued flock child')
 def close(self,p):
  out,err=p.communicate(timeout=3);return p.returncode,out,err
 def test_original_unbounded_wait_failure_reproduces_unleased_dispatch(self):
  owner=self.hold(0);p,marker=self.launch(ORIGINAL,0);os.kill(self.waiter(p),signal.SIGTERM);rc,_,_=self.close(p)
  self.assertEqual(rc,0);self.assertTrue(marker.exists());self.assertNotEqual(owner.read_text(),'UNRELATED_CPU_HOLDER\n')
 def test_unbounded_waiter_failure_never_publishes_or_dispatches(self):
  owner=self.hold(0);p,marker=self.launch(card=0);os.kill(self.waiter(p),signal.SIGTERM);rc,_,err=self.close(p)
  self.assertNotEqual(rc,0);self.assertFalse(marker.exists());self.assertEqual(owner.read_text(),'UNRELATED_CPU_HOLDER\n');self.assertIn('command will not run',err)
 def test_queued_wrapper_TERM_and_INT_terminate_actual_waiter(self):
  for sig,status in [(signal.SIGTERM,143),(signal.SIGINT,130)]:
   with self.subTest(signal=sig):
    owner=self.hold(0);p,marker=self.launch(card=0);waiter=self.waiter(p);p.send_signal(sig);rc,_,_=self.close(p)
    self.assertEqual(rc,status);self.assertFalse(marker.exists());self.assertEqual(owner.read_text(),'UNRELATED_CPU_HOLDER\n');self.assertFalse(Path('/proc/'+str(waiter)).exists());self.holders.pop().close()
 def test_second_card_failure_releases_first_and_preserves_foreign_owner(self):
  owner=self.hold(1);p,marker=self.launch();os.kill(self.waiter(p),signal.SIGTERM);rc,_,_=self.close(p)
  self.assertNotEqual(rc,0);self.assertFalse(marker.exists());self.assertEqual(owner.read_text(),'UNRELATED_CPU_HOLDER\n');self.assertEqual(Path(str(self.base)+'.0.owner').read_text(),'')
  with open(str(self.base)+'.0','w') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def test_timeout_never_dispatches(self):
  owner=self.hold(0);p,marker=self.launch(card=0,timeout='.05');rc,_,_=self.close(p)
  self.assertEqual(rc,124);self.assertFalse(marker.exists());self.assertEqual(owner.read_text(),'UNRELATED_CPU_HOLDER\n')
 def test_active_TERM_latches_until_foreground_child_finishes(self):
  started=self.root/'STARTED';done=self.root/'DONE'
  code='from pathlib import Path;import sys,time;Path(sys.argv[1]).write_text("started");time.sleep(.35);Path(sys.argv[2]).write_text("done")'
  p,_=self.launch(card=0,command=[sys.executable,'-c',code,str(started),str(done)])
  until=time.monotonic()+3
  while not started.exists() and time.monotonic()<until:time.sleep(.01)
  self.assertTrue(started.exists());p.send_signal(signal.SIGTERM);time.sleep(.05);self.assertIsNone(p.poll());self.assertIn('pid='+str(p.pid),Path(str(self.base)+'.0.owner').read_text())
  with open(str(self.base)+'.0','w') as f:
   with self.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  rc,_,_=self.close(p);self.assertEqual(rc,143);self.assertTrue(done.exists());self.assertEqual(Path(str(self.base)+'.0.owner').read_text(),'')
 def test_success_and_command_failure_close_normally(self):
  p,marker=self.launch(card=1);rc,_,_=self.close(p);self.assertEqual(rc,0);self.assertTrue(marker.exists());self.assertEqual(Path(str(self.base)+'.1.owner').read_text(),'')
  p,_=self.launch(card=1,command=[sys.executable,'-c','raise SystemExit(17)']);rc,_,_=self.close(p);self.assertEqual(rc,17);self.assertEqual(Path(str(self.base)+'.1.owner').read_text(),'')

if __name__=='__main__':unittest.main()
