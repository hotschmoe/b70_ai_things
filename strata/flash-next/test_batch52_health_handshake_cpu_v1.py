"""Tiny owned protocol, mutation and byte-boundary controls; no device/payload."""
import ast,copy,hashlib,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import batch52_health_handshake_v1 as h
import batch50_byte_epochs_v1 as w
import batch52_witness_binding_v1 as saved
HERE=Path(__file__).resolve().parent
class Controls(unittest.TestCase):
 def setup(self,base):
  source=base/'parent';source.write_text('parent source');controller=base/'controller';controller.write_text('controller source');plan=base/'plan';plan.write_text('immutable admitted plan');digest=h.sha(plan)
  admitted=SimpleNamespace(verify=lambda *args:digest)
  identities=lambda pid:{'pid':pid,'start_ticks':pid+1000};leases=[{'card':c,'device':1,'inode':100+c}for c in (0,1)]
  return source,controller,admitted,digest,identities,leases
 def protocol(self,base):
  source,controller,admitted,digest,identities,leases=self.setup(base);root=base/'handshake'
  with patch.object(h,'pid_identity',side_effect=identities),patch.object(h,'leases',return_value=leases),patch.object(h.os,'getpid',return_value=501):offer=h.offer(root,digest,source,controller,100)
  def boundary(label):now=time.time();return {'label':label,'started_epoch':now,'finished_epoch':now,'rows':[]}
  pack=SimpleNamespace(phase='admission',ready_seal=lambda:boundary('ready_complete_byte_recheck'));sdk=SimpleNamespace(phase='admission',ready_seal=lambda:boundary('ready'))
  with patch.object(h,'pid_identity',side_effect=identities),patch.object(h,'leases',return_value=leases),patch.object(h.os,'getpid',return_value=502),patch.object(h.os,'getppid',return_value=501):ready=h.ready(root,admitted,pack,sdk,source,controller)
  return root,source,controller,admitted,digest,identities,leases,offer,ready
 def test_actual_offer_ready_identity_source_and_plan_join(self):
  with tempfile.TemporaryDirectory()as d:
   root,source,controller,admitted,digest,ids,leases,offer,ready=self.protocol(Path(d))
   with patch.object(h,'pid_identity',side_effect=ids),patch.object(h,'leases',return_value=leases),patch.object(h.os,'getpid',return_value=501):self.assertEqual(h.validate_ready(root,offer,502,digest),ready)
 def test_foreign_PID_start_plan_lease_and_source_fail(self):
  for key in ('child','parent','plan_sha256','lease_identity','parent_source_sha256','offer_sha256'):
   with tempfile.TemporaryDirectory()as d:
    root,source,controller,admitted,digest,ids,leases,offer,ready=self.protocol(Path(d));bad=copy.deepcopy(ready)
    if key in ('child','parent'):bad[key]['start_ticks']+=1
    elif key=='lease_identity':bad[key][0]['inode']+=1
    else:bad[key]='foreign'
    (root/'ready.json').write_text(json.dumps(bad))
    with patch.object(h,'pid_identity',side_effect=ids),patch.object(h,'leases',return_value=leases),patch.object(h.os,'getpid',return_value=501):self.assertRaises(ValueError,h.validate_ready,root,offer,502,digest)
 def test_duplicate_ACK_and_symlink_records_refused(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'record';h.write_new(p,{'x':1});self.assertRaises(FileExistsError,h.write_new,p,{'x':2});link=Path(d)/'link';link.symlink_to(p);self.assertRaises(ValueError,h.read,link)
 def test_no_ACK_timeout_and_parent_loss(self):
  with tempfile.TemporaryDirectory()as d:
   root,source,controller,admitted,digest,ids,leases,offer,ready=self.protocol(Path(d))
   with patch.object(h,'pid_identity',side_effect=ids),patch.object(h,'leases',return_value=leases),patch.object(h.os,'getppid',return_value=501),patch.object(h.time,'monotonic',side_effect=[0,101]):self.assertRaisesRegex(ValueError,'timeout',h.wait_ack,root,ready,admitted,Path(d)/'health',source,controller)
   with patch.object(h,'pid_identity',return_value={'pid':501,'start_ticks':0}),patch.object(h.time,'sleep',side_effect=AssertionError('must not sleep after ownership loss')):self.assertRaisesRegex(ValueError,'Parent exited',h.wait_ack,root,ready,admitted,Path(d)/'health',source,controller)
 def test_unchanged_freshness_rejects_original467_second_sequence(self):
  health={'passed':True,'cards':[0,1],'finished_epoch':100};plan={'cards':[0,1]}
  h.freshness(health,plan,400);self.assertRaises(ValueError,h.freshness,health,plan,400.001);self.assertRaises(ValueError,h.freshness,health,plan,567);self.assertRaises(ValueError,h.freshness,health,plan,99)
 def test_reordered_real_health_after_READY_is_required(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'health';h.write_new(p,{'schema':1,'passed':True,'cards':[0,1],'health_image':h.HEALTH,'started_epoch':1,'finished_epoch':2})
   self.assertRaisesRegex(ValueError,'follow child READY',h.health_binding,p,3)
 def test_parent_health_loop_follows_READY_and_child_final_seals_after_ACK(self):
  parent=(HERE/'qualify_batch_numerical_v52.py').read_text();native=(HERE/'run_batch_serial_controls_v52.py').read_text();ctrl=(HERE/'batch_numerical_execution_v52.py').read_text()
  self.assertLess(parent.index('child=subprocess.Popen'),parent.index("health('pre')"));self.assertLess(parent.index('handshake.validate_ready'),parent.index("health('pre')"));self.assertLess(parent.index("faults('pre-kernel-journal')"),parent.index('handshake.acknowledge'))
  self.assertLess(native.index('engine_binding('),native.index('ready_row=ready'));self.assertLess(native.index('ack_row=wait_ack'),native.index('cheap(plan,current)'));self.assertLess(native.index('ack_row=wait_ack'),native.index('pack_epoch.seal_predevice()'));self.assertLess(native.index('sdk_epoch.seal_predevice()'),native.index('protocol = CacheOffProtocol'));self.assertIn('freshness(health,plan,before_seal_epoch)',native);self.assertIn('<=300',native);self.assertIn('freshness(health,plan,time.time())',native)
  self.assertIn("require(plan['kind']=='serial'",ctrl);self.assertIn("'--handshake'",(HERE/'batch52_runtime_evidence_v1.py').read_text());self.assertNotIn("row['finished_epoch']<=parent['child_started_epoch']",(HERE/'batch52_runtime_evidence_v1.py').read_text())
 def test_original_serial_request_and_raw_extractor_loop_unchanged(self):
  old=ast.parse((HERE/'run_batch_serial_controls_v49.py').read_text());new=ast.parse((HERE/'run_batch_serial_controls_v52.py').read_text())
  def loop(tree):return next(n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='job')
  self.assertEqual(ast.dump(loop(old),include_attributes=False),ast.dump(loop(new),include_attributes=False))
 def test_real_tiny_pack_entry_READY_seal_post_and_mutation(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'tiny';p.write_bytes(b'known tiny bytes');e=w.ReadyPackEpoch([(p,h.sha(p))],60);ready=e.ready_seal();self.assertEqual(e.phase,'admission');e.seal_predevice();e.digest_for(p,h.sha(p));proof=e.finalize();self.assertEqual(len(proof['boundaries']),4)
   fresh=w.ReadyPackEpoch([(p,h.sha(p))],60);saved.binding(proof,fresh,proof['owner_pid'],time.time(),proof['boundaries'][3]['started_epoch'],'pack',ready,ready['finished_epoch']);p.write_bytes(b'changed bytes');self.assertRaises(ValueError,fresh.ready_seal)
 def test_duplicate_READY_and_post_without_seal_refused(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'tiny';p.write_bytes(b'tiny');e=w.ReadyPackEpoch([(p,h.sha(p))],60);e.ready_seal();self.assertRaises(ValueError,e.ready_seal);self.assertRaises(ValueError,e.finalize)
 def test_wrong_four_boundary_or_ACK_order_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'tiny';p.write_bytes(b'tiny');e=w.ReadyPackEpoch([(p,h.sha(p))],60);r=e.ready_seal();e.seal_predevice();e.digest_for(p,h.sha(p));proof=e.finalize();fresh=w.ReadyPackEpoch([(p,h.sha(p))],60)
   for mode in ('ACK','label','missing'):
    bad=copy.deepcopy(proof);ack=r['finished_epoch']
    if mode=='ACK':ack=proof['boundaries'][2]['finished_epoch']+1
    elif mode=='label':bad['boundaries'][1]['label']='entry'
    else:bad['boundaries'].pop(1)
    self.assertRaises(ValueError,saved.binding,bad,fresh,proof['owner_pid'],time.time(),proof['boundaries'][-1]['started_epoch'],'pack',r,ack)
 def test_boolean_card_schema_and_identity_do_not_equal_integers(self):
  for field,value in [('cards',[False,True]),('schema',True),('schema',1.0)]:
   with tempfile.TemporaryDirectory()as d:
    row={'schema':1,'passed':True,'cards':[0,1],'health_image':h.HEALTH,'started_epoch':1,'finished_epoch':2};row[field]=value;p=Path(d)/'health';h.write_new(p,row);self.assertRaisesRegex(ValueError,'typed health',h.health_binding,p,1)
  with tempfile.TemporaryDirectory()as d:
   root,source,controller,admitted,digest,ids,leases,offer,ready=self.protocol(Path(d))
   for field,value in [('schema',True),('generation',52.0),('parent',{'pid':True,'start_ticks':1501}),('lease_identity',[{'card':False,'device':1,'inode':100},{'card':1,'device':1,'inode':101}])]:
    bad=copy.deepcopy(ready);bad[field]=value;self.assertRaises(ValueError,h.record_types,bad)
 def health_fixture(self,base,ready_epoch):
  sources=[base/'vllm/int4/diagnostics/xpu_health_strict.sh',base/'bin/xpu-collective-health',base/'bin/xpu-collective-health.py']
  for p in sources:p.parent.mkdir(parents=True,exist_ok=True);p.write_text('known health source')
  paths=[['--img',h.HEALTH],['--img',h.HEALTH,'--p2p','0','--timeout','180']];rows=[]
  for p,argv,label in zip(sources[:2],paths,('strict','compiled-pair')):
   cmd=[str(p),*argv];command=base/('pre-'+label+'.command.json');log=base/('pre-'+label+'.log');h.write_new(command,cmd);log.write_text('complete original health output');rows.append({'command':cmd,'command_file_sha256':h.sha(command),'path':str(log),'return_code':0,'error':None,'sha256':h.sha(log),'stdout_sha256':h.sha(log),'started_epoch':ready_epoch+.01,'finished_epoch':ready_epoch+.02})
  health={'schema':1,'passed':True,'cards':[0,1],'health_image':h.HEALTH,'files':rows,'checks':copy.deepcopy(rows),'source_sha256':{str(p):h.sha(p)for p in sources},'started_epoch':ready_epoch+.005,'finished_epoch':ready_epoch+.03};path=base/'pre-health.json';h.write_new(path,health);return path,health
 def test_complete_ACK_rejoins_actual_health_bytes_and_source(self):
  with tempfile.TemporaryDirectory()as d:
   base=Path(d);root,source,controller,admitted,digest,ids,leases,offer,ready=self.protocol(base);health_path,health=self.health_fixture(base,ready['ready_epoch']);jlog=base/'pre-kernel-journal.log';jlog.write_text('no faults');jcmd=base/'pre-kernel-journal.command.json';argv=['journalctl','-k','--since','@1','--no-pager'];h.write_new(jcmd,argv);jpath=base/'pre-kernel-receipt.json';h.write_new(jpath,{'return_code':0,'error':None,'sha256':h.sha(jlog),'stdout_sha256':h.sha(jlog),'path':str(jlog),'command':argv,'command_file_sha256':h.sha(jcmd)})
   with patch.object(h,'ROOT',base),patch.object(h,'pid_identity',side_effect=ids),patch.object(h,'leases',return_value=leases),patch.object(h.os,'getpid',return_value=501),patch.object(h.time,'time',return_value=health['finished_epoch']+.1):ack=h.acknowledge(root,ready,health_path,jpath)
   with patch.object(h,'ROOT',base),patch.object(h,'pid_identity',side_effect=ids),patch.object(h,'leases',return_value=leases),patch.object(h.os,'getpid',return_value=502),patch.object(h.os,'getppid',return_value=501),patch.object(h.time,'time',return_value=health['finished_epoch']+.2):self.assertEqual(h.wait_ack(root,ready,admitted,health_path,source,controller),ack)
   source.write_text('changed after READY')
   with patch.object(h,'ROOT',base),patch.object(h,'pid_identity',side_effect=ids),patch.object(h,'leases',return_value=leases),patch.object(h.os,'getpid',return_value=502),patch.object(h.os,'getppid',return_value=501):self.assertRaises(ValueError,h.wait_ack,root,ready,admitted,health_path,source,controller)
 def test_saved_health_rc_error_log_epoch_mutations_fail(self):
  for mutation in ('return_code','error','epoch','command','log'):
   with tempfile.TemporaryDirectory()as d:
    base=Path(d);path,health=self.health_fixture(base,100);bad=copy.deepcopy(health)
    if mutation=='return_code':bad['files'][0]['return_code']=True
    elif mutation=='error':bad['files'][0]['error']='timeout'
    elif mutation=='epoch':bad['files'][0]['started_epoch']=True
    elif mutation=='command':bad['files'][0]['command'][0]='foreign'
    else:Path(bad['files'][0]['path']).write_text('mutated')
    bad['checks']=copy.deepcopy(bad['files']);path.write_text(json.dumps(bad))
    with patch.object(h,'ROOT',base):self.assertRaises(ValueError,h.health_binding,path,100)
 def test_READY_data_cannot_be_mere_matching_record_without_current_bytes(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'tiny';p.write_bytes(b'tiny');e=w.ReadyPackEpoch([(p,h.sha(p))],60);r=e.ready_seal();e.seal_predevice();e.digest_for(p,h.sha(p));proof=e.finalize();fresh=w.ReadyPackEpoch([(p,h.sha(p))],60);bad=copy.deepcopy(proof);bad['boundaries'][1]['rows'][0]['sha256']='0'*64
   self.assertRaises(ValueError,saved.binding,bad,fresh,proof['owner_pid'],time.time(),proof['boundaries'][-1]['started_epoch'],'pack',bad['boundaries'][1],r['finished_epoch'])
if __name__=='__main__':unittest.main()
