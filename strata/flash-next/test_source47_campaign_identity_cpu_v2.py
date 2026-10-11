"""One CPU control executes future parent flow: mutation/fault must still close."""
import copy,hashlib,json,os,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_source47_grouped_f16_model_v2 as actor

class Test(unittest.TestCase):
 def test_mutation_fault_and_missing_final_identity_fail_closed(self):
  for mode in ['valid','mutation','native_fault']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);shards=[]
    for i in range(4):
     p=root/'UD-Q4_K_XL'/('tiny%d'%i);p.parent.mkdir(exist_ok=True);p.write_bytes(b'CPU-only original'+bytes([i]));shards.append(p)
    lock={'destination':'','files':[{'path':str(p.relative_to(root))}for p in shards]};(root/'model-lock.json').write_text(json.dumps(lock));inputs=root/'inputs.json';inputs.write_text('[]');plan=root/'plan.json';plan.write_text(json.dumps({'inputs_path':str(inputs),'template_path':str(root)}));source=root/'source.json';source.write_text('{}');out=root/'result'
    events=[];original_sha={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in shards}
    def read(p):return json.loads(Path(p).read_text())
    def write(p,value):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(value))
    def scan(lockpath,locked,paths,output,after):
     events.append(('scan',Path(output).name));rows=[]
     for p in paths:
      signature=actor.c.stat_signature(p);digest=hashlib.sha256(p.read_bytes()).hexdigest();rows.append({'path':str(p),'stat_before':signature,'stat_after':signature,'sha256':digest})
     value={'passed':all(row['sha256']==original_sha[row['path']]for row in rows),'started':time.time(),'finished':time.time(),'rows':rows};write(output,value);return value
    def health(out,label):events.append(('health',label));return {'passed':True,'started_epoch':time.time(),'finished_epoch':time.time()}
    def phase(out,label,*args):
     events.append(('native',label));write(out/label/'requests.json',[]);write(out/label/'recipe.json',{'requests':[]})
     if mode=='mutation'and label=='CPU-OFF':shards[0].write_bytes(b'changed CPU fixture')
     if mode=='native_fault'and label=='CPU-ON':raise RuntimeError('synthetic native failure after owned retirement')
     return {'passed':True}
    ns=dict(actor.NS['_PORTS']);ns.update(ROOT=root,HERE=root,PLAN=source,PHASES=('CPU-OFF','CPU-ON'),closure=lambda:'CPU-source',set_api=lambda *x:None,set_case=lambda*x:None,set_runtime=lambda*x:None,prerequisite=lambda*x:{},preflight=lambda*x:{'CPU-only':True},admission=lambda*x:({'source':str(root)},{}),fresh=SimpleNamespace(binding=lambda:{'CPU-source':True}),intake=SimpleNamespace(SDK=root,guard=lambda*x:{'passed':True,'epoch':time.time()}),run_phase=phase,stat_snapshot=lambda paths:{str(p):actor.c.stat_signature(p)for p in paths},contract=SimpleNamespace(requests=lambda*x:[],equivalence=lambda*x:{'CPU-only':True}))
    ns['c']=SimpleNamespace(read=read,write=write,leased=lambda*x:None,sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
    ns['life']=SimpleNamespace(tokenizer=lambda*x:[],match_speed_inputs=lambda*x:{},health=health,journal=lambda*out:events.append(('journal',out[1])),q=SimpleNamespace(upload_v2=SimpleNamespace(full_buffered_identity=scan)))
    # Copy the actual new compiled parent function into isolated CPU globals.
    main=type(actor.NS['_PORTS']['main'])(actor.NS['_PORTS']['main'].__code__,ns)
    argv=['CPU-parent','--plan',str(plan),'--output',str(out),'--phase','host-latency','--off-receipt',str(root),'--leased','--speed-trace',str(root/'speed'),'--pti-intake',str(root/'pti'),'--pti-intake-sha256','CPU','--prompt-role','prefill_about512','--api-prepared',str(root/'api'),'--api-prepared-sha256','CPU']
    # Preserve caller signal dispositions: this test performs no GPU/container work.
    import signal
    handlers={s:signal.getsignal(s)for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGUSR1]}
    try:
     with patch.object(sys,'argv',argv):
      if mode=='valid':main()
      else:
       with self.assertRaises(ValueError):main()
    finally:
     for s,handler in handlers.items():signal.signal(s,handler)
    report=read(out/'report.json');self.assertEqual([event for event in events if event[0]=='scan'],[('scan','pre-full4.json'),('scan','post-full4.json')]);self.assertIn('post_full4',report);self.assertEqual(report['passed'],mode=='valid')
    if mode=='valid':
     current={str(p):actor.c.stat_signature(p)for p in shards};self.assertTrue(actor.campaign_stats(report,current));broken=copy.deepcopy(report);broken.pop('post_full4')
     with self.assertRaises(ValueError):actor.campaign_stats(broken,current)
     broken=copy.deepcopy(report);broken['per_arm']['CPU-ON']['pre_stat'][str(shards[0])][1]+=1
     with self.assertRaises(ValueError):actor.campaign_stats(broken,current)
if __name__=='__main__':unittest.main()
