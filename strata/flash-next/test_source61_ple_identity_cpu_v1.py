"""Execute real generated owner flow against tiny synthetic files only."""
import types,copy,hashlib,json,signal,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_source61_ple_io_v1 as actor
import source61_ple_identity_ports_v1 as identity

class Test(unittest.TestCase):
 def test_diagnostic_mutation_and_missing_final_close(self):
  for mode in ('valid','mutation','fault','child'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);shards=[]
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def read(p):return json.loads(Path(p).read_text())
    def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v))
    for i in range(4):
     p=root/'UD-Q4_K_XL'/str(i);p.parent.mkdir(exist_ok=True);p.write_bytes(b'tiny CPU fixture'+bytes([i]));shards.append(p)
    lock={'destination':'','revision':'synthetic','files':[dict(path=str(p.relative_to(root)),sha256=sha(p),size=p.stat().st_size) for p in shards]};write(root/'model-lock.json',lock)
    rows=[dict(path=str(p),sha256=sha(p),expected_sha256=sha(p),bytes=p.stat().st_size,passed=True,stat_before=actor.c.stat_signature(p),stat_after=actor.c.stat_signature(p)) for p in shards]
    prior=dict(passed=True,lock_sha256=sha(root/'model-lock.json'),model_revision='synthetic',started=time.time()-2,finished=time.time()-1,read_mode='ordinary complete buffered read; no direct IO/cache mutation',rows=rows);write(root/'prior.json',prior)
    if mode=='child':identity.configure('qualifying',gpu_child=True)
    else:identity.configure('diagnostic',str(root/'prior.json'),sha(root/'prior.json'))
    inputs=root/'inputs.json';write(inputs,[]);plan=root/'plan.json';write(plan,{'inputs_path':str(inputs),'template_path':str(root)});source=root/'source.json';write(source,{});out=root/'out';events=[]
    def health(out,label):
     events.append(label);value=dict(passed=True,started_epoch=time.time(),finished_epoch=time.time());write(out/(label+'-health.json'),value);(out/(label+'-journal.log')).write_text('');return value
    def phase(out,label,*args):
     write(out/label/'requests.json',[]);write(out/label/'recipe.json',{'requests':[]});write(out/label/'command.json',['--name',label]);write(out/label/'result.json',{'removal':{'removed':True},'terminal':{'State':{'Running':False}}})
     if mode=='mutation' and label=='CPU-OFF':shards[0].write_bytes(b'mutated')
     if mode=='fault' and label=='CPU-ON':raise RuntimeError('synthetic retired failure')
     return dict(passed=True)
    ns=dict(actor.NS);ns.update(ROOT=root,HERE=root,PLAN=source,PHASES=('CPU-OFF','CPU-ON'),closure=lambda:'CPU',set_sdk=lambda*x:None,set_api=lambda*x:None,set_case=lambda*x:None,set_runtime=lambda*x:None,preflight=lambda*x:{},prerequisite=lambda*x:{},admission=lambda*x:({'source':str(root)},{}),fresh=SimpleNamespace(binding=lambda:{},configure=lambda*x:None,SDK=root,RECEIPT_SHA='CPU'),intake=SimpleNamespace(SDK=root,guard=lambda*x:dict(passed=True,epoch=time.time())),run_phase=phase,stat_snapshot=lambda paths:{str(p):actor.c.stat_signature(p) for p in paths},contract=SimpleNamespace(requests=lambda*x:[],equivalence=lambda*x:{}))
    for name,value in list(ns.items()):
     if isinstance(value,types.FunctionType) and value.__globals__ is actor.NS:ns[name]=types.FunctionType(value.__code__,ns,value.__name__,value.__defaults__,value.__closure__)
    ns['c']=SimpleNamespace(read=read,write=write,sha=sha,stat_signature=actor.c.stat_signature,leased=lambda*x:None,absent=lambda*x:True)
    def forbidden(*args):
     if mode!='child':raise AssertionError('Diagnostic must never reread complete model files')
     events.append('CPU-synthetic-scan');value=copy.deepcopy(prior);value.update(started=time.time(),finished=time.time());write(args[3],value);return value
    ns['life']=SimpleNamespace(tokenizer=lambda*x:[],match_speed_inputs=lambda*x:{},health=health,journal=lambda*x:None,q=SimpleNamespace(upload_v2=SimpleNamespace(full_buffered_identity=forbidden)))
    actual=identity.ports(ns,actor.ORIGINAL_PORTS)
    argv=['CPU','--plan',str(plan),'--output',str(out),'--phase','host-latency','--off-receipt',str(root),'--leased','--speed-trace','CPU','--pti-intake','CPU','--pti-intake-sha256','CPU','--prompt-role','prefill_about512','--api-prepared','CPU','--api-prepared-sha256','CPU','--sdk',str(root),'--sdk-receipt-sha256','CPU']
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGUSR1)}
    try:
     with patch.object(sys,'argv',argv):
      if mode in ('valid','child'):actual['main']()
      else:
       with self.assertRaises(ValueError):actual['main']()
    finally:
     for sig,value in handlers.items():signal.signal(sig,value)
    report=read(out/'report.json');self.assertEqual(report['passed'],mode in ('valid','child'))
    if mode=='child':
     self.assertTrue(report['qualification_pending_CPU_identity']);self.assertEqual(events.count('CPU-synthetic-scan'),1);self.assertFalse((out/'post-full4.json').exists())
     owner=SimpleNamespace(c=ns['c'],HERE=root,ROOT=root,NS={**ns,**actual},closure=lambda:'CPU')
     identity.close_after_child(owner,out,0);closed=read(out/'report.json');self.assertTrue(closed['passed']);self.assertFalse(closed['qualification_pending_CPU_identity']);self.assertEqual(events.count('CPU-synthetic-scan'),2);continue
    self.assertFalse(report['fresh_model_bytes_read']);self.assertFalse((out/'pre-full4.json').exists());self.assertFalse((out/'post-full4.json').exists())
    if mode=='valid':
     broken=copy.deepcopy(report);broken.pop('post_diagnostic_identity')
     with self.assertRaises((ValueError,KeyError)):actual['campaign_stats'](broken,{str(p):actor.c.stat_signature(p) for p in shards})
    self.assertIn('CPU-OFF-post',events)

if __name__=='__main__':unittest.main()
