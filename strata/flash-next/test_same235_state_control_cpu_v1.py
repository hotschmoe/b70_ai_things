"""Source/input/callable negatives; synthetic API boundaries only."""
import inspect,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import same235_state_control_v1 as c
class Tests(unittest.TestCase):
 def test_exact_235(self):
  jobs=c.jobs();self.assertEqual(len(jobs),4);self.assertEqual([r['request_policy']['strata_fresh'] for r in jobs],[False,False,True,True]);self.assertTrue(all(len(r['ids'])==235 and r['max_new']==64 and not r['pin_present'] and not r['forced_continuation'] for r in jobs));self.assertEqual(jobs[0]['ids'],jobs[2]['ids']);self.assertEqual(jobs[1]['messages'],jobs[3]['messages'])
 def test_unchanged_warm47(self):
  jobs=c.jobs();f,_=c.contract.fixture_binding();self.assertEqual([len(r['ids']) for r in f['fixtures']['warm']],[47,47]);self.assertTrue(all(r['warm_request_policy']=={'strata_fresh':True} and r['warm_max_new']==32 for r in jobs))
 def test_source_only_qualification_scope(self):
  plan=c.source.read(c.PLAN);self.assertTrue(plan['genuine_runtime_preparable']);self.assertFalse(plan['EOS_cause_established']);c.source_binding()
 def test_fake_boundary_serial_request_shape(self):
  calls=[]
  def response(url,model,messages,path,**kwargs):calls.append((messages,kwargs));return {'client_transport_completed':True,'real_client_cancel_requested':False,'rows':[]}
  with tempfile.TemporaryDirectory() as t,patch.object(c.source,'manifest_binding'),patch.object(c.source.c1,'leased'),patch.object(c.client,'cohort',side_effect=response),patch('same235_state_control_protocol_v1.await_warm',return_value=[]):
   r=c.collect_controls('http://127.0.0.1:18339','hotschmoe-dd',{'CPU':'SYNTHETIC'},Path(t)/'out');self.assertEqual(len(r['rows']),4);self.assertEqual([len(a) for a,b in calls],[2,1]*4);self.assertFalse(r['actual_cached_state_handoff_qualified']);self.assertTrue(r['matched_API_concurrent_fresh_control_still_required'])
 def test_wrong_primary(self):
  with tempfile.TemporaryDirectory() as t,patch.object(c.source,'manifest_binding'),patch.object(c.source.c1,'leased'),patch.object(c.client,'cohort') as api:
   with self.assertRaises(ValueError):c.collect_controls('http://127.0.0.1:18339','foreign',{},Path(t)/'out')
   api.assert_not_called()
 def test_actual_transport_error(self):
  with tempfile.TemporaryDirectory() as t,patch.object(c.source,'manifest_binding'),patch.object(c.source.c1,'leased'),patch.object(c.client,'cohort',return_value={'client_transport_completed':False}):
   with self.assertRaises(ValueError):c.collect_controls('http://127.0.0.1:18339','hotschmoe-dd',{},Path(t)/'out')
 def synthetic_protocol(self):
  import same235_state_control_protocol_v1 as protocol
  jobs=c.jobs();events=[];controls=[];term={};fixture,_=c.contract.fixture_binding();call=0
  def body(messages,n,fresh,rid):return dict(model='hotschmoe-dd',messages=messages,stream=True,temperature=0,max_tokens=n,frequency_penalty=0,presence_penalty=0,user=rid,chat_template_kwargs={'enable_thinking':False},reasoning_budget_tokens=0,strata_fresh=fresh)
  def warm_pair(start,job):
   nonlocal call
   client=[];pred=[]
   for i in (0,1):
    call+=1;seq=start+2*i;ids=fixture['fixtures']['warm'][i]['ids'];events.extend([dict(kind='engine_begin',call=call,sequence=seq,epoch=seq+1,engine_pid=22,engine_generation=1,submitted_ids=ids,rendered_prompt={'messages':job['warm_messages'][i]},max_new=32),dict(kind='engine_end',call=call,sequence=seq+30,engine_pid=22,engine_generation=1,cancelled=False)]);term[call]={'actual_native_terminal':{'sequence':seq+29},'actual_client_cancelled':False};rid='warm-'+str(call);client.append(dict(client_index=i,request_id=rid,request=body(job['warm_messages'][i],32,True,rid),error=None,done_received=True,cancel_requested=False,sent_epoch=start-1,finished_epoch=start+35));pred.append({'logical_index':i,'call':call,'native_terminal_sequence':seq+29,'engine_end_sequence':seq+30})
   return {'rows':client},pred
  warm_pair(10,jobs[0]);events.append(dict(kind='native_receive',engine_pid=22,sequence=35,line='SBF batch_event pid=22 enginegen=12345 event=1 rows=2 active_mask=3 completed=1'))
  for index,j in enumerate(jobs):
   warm,pred=warm_pair(50+100*index,j);call+=1;start=100+index*100;rid=call;fresh=j['request_policy']['strata_fresh'];events.extend([dict(kind='engine_begin',call=call,engine_pid=22,engine_generation=1,sequence=start,submitted_ids=j['ids'],rendered_prompt={'messages':j['messages']},epoch=start+1,max_new=64),dict(kind='native_send',call=call,engine_pid=22,sequence=start+2,line='GEN 64 seed=1 fresh='+str(int(fresh))+' rid='+str(rid)+' '+','.join(map(str,j['ids']))),dict(kind='native_receive',engine_pid=22,sequence=start+3,line='BCPUBLIC reset rid='+str(rid)+' slotgen=0 stages=1 fresh='+str(int(fresh))+' pin_present=0 pin=0 lookup_ceiling=0 read_from=0'),dict(kind='engine_end',call=call,engine_pid=22,engine_generation=1,rid=rid,cancelled=False,generated_ids=[248046],sequence=start+5,engine_last={'prompt_tokens':235,'prompt_read':235,'reused':0,'finish':'stop'})]);term[call]={'actual_native_terminal':{'sequence':start+4},'actual_client_cancelled':False};clientrid='target-'+str(call);controls.append({'declared_job':j,'actual_warm_client':warm,'actual_warm_native_predecessor':pred,'actual_target_client':{'rows':[{'request_id':clientrid,'request':body(j['messages'],64,fresh,clientrid),'error':None,'done_received':True,'cancel_requested':False,'sent_epoch':start-1,'finished_epoch':start+6}]}})
  return protocol,events,controls,term
 def test_source_shaped_counter_recollection(self):
  p,events,controls,term=self.synthetic_protocol()
  with patch.object(p,'associate_events',return_value=term),patch.object(p,'actual_phase_legs'):self.assertEqual(len(p.recollect(events,controls,1)),4)
 def test_foreign_reset_reused_or_keys_refused(self):
  for kind in ('pid','reuse','pin','fresh'):
   p,events,controls,term=self.synthetic_protocol()
   if kind=='pid':next(r for r in events if r['kind']=='native_receive')['engine_pid']=99
   elif kind=='reuse':next(r for r in events if r['kind']=='engine_end' and r['call']==5)['engine_last']['reused']=1
   else:
    row=next(r for r in events if r['kind']=='native_send');row['line']=row['line'].replace('fresh=0','fresh=1') if kind=='fresh' else row['line'].replace('rid=5','rid=5 pin=0')
   with patch.object(p,'associate_events',return_value=term),patch.object(p,'actual_phase_legs'),self.assertRaises(ValueError):p.recollect(events,controls,1)
 def test_wrong_or_overlapping_warm_predecessor(self):
  for kind in ('swapped','wholeprior','overlap','body','epoch','priorTargetOverlap'):
   p,events,controls,term=self.synthetic_protocol()
   if kind=='swapped':controls[1]['actual_warm_client']=controls[0]['actual_warm_client']
   elif kind=='wholeprior':
    controls[1]['actual_warm_client']=controls[0]['actual_warm_client'];controls[1]['actual_warm_native_predecessor']=controls[0]['actual_warm_native_predecessor']
   elif kind=='priorTargetOverlap':next(r for r in events if r['kind']=='engine_end' and r['call']==5)['sequence']=160
   elif kind=='overlap':next(r for r in events if r['kind']=='engine_end' and r['call']==3)['sequence']=110
   elif kind=='body':controls[0]['actual_warm_client']['rows'][0]['request']['strata_fresh']=False
   else:controls[0]['actual_warm_client']['rows'][0]['sent_epoch']+=10000
   with patch.object(p,'associate_events',return_value=term),patch.object(p,'actual_phase_legs'),self.assertRaises(ValueError):p.recollect(events,controls,1)
 def test_foreign_engine_incarnation_or_warm_geometry(self):
  for kind in ('callPID','incarnation','eventPID','mask'):
   p,events,controls,term=self.synthetic_protocol()
   if kind=='callPID':next(r for r in events if r['kind']=='engine_begin')['engine_pid']=99
   elif kind=='incarnation':next(r for r in events if r['kind']=='engine_end')['engine_generation']=2
   else:
    e=next(r for r in events if r['kind']=='native_receive' and r['line'].startswith('SBF batch_event '));e['line']=e['line'].replace('pid=22','pid=99') if kind=='eventPID' else e['line'].replace('active_mask=3','active_mask=1')
   with patch.object(p,'associate_events',return_value=term),patch.object(p,'actual_phase_legs'),self.assertRaises(ValueError):p.recollect(events,controls,1)
 def test_native_error_refusal_retained_and_rejected(self):
  p,events,controls,term=self.synthetic_protocol();events.append({'kind':'native_receive','line':'SERR source refusal'})
  with self.assertRaises(ValueError):p.recollect(events,controls,1)
  with tempfile.TemporaryDirectory() as t:
   f=Path(t)/'raw.jsonl';f.write_text(__import__('json').dumps(events[-1])+'\n');self.assertEqual(p.events(f)[0]['line'],'SERR source refusal')
 def test_raw_size_before_decode_and_partial_tail_explicit(self):
  import same235_state_control_protocol_v1 as p
  with tempfile.TemporaryDirectory() as t:
   f=Path(t)/'raw.jsonl';f.write_bytes(b'X'*1025)
   with patch.object(p,'MAX_RAW_LINE',1024),patch.object(p.json,'loads') as decode:
    with self.assertRaises(ValueError):p.events(f)
    decode.assert_not_called()
   f.write_bytes(b'{"kind": "engine_end"')
   self.assertEqual(p.events(f,allow_incomplete_tail=True),[])
   with self.assertRaises(ValueError):p.events(f)
   f.write_bytes(b'{"kind": "native_receive", "line": "ERR real"}')
   with self.assertRaises(ValueError):p.events(f)
 def test_new_parent_controller_exact_source(self):
  import ast
  import qualify_batch_api_same235_state_control_v1 as p
  text=p.adapted_source();ast.parse(text);self.assertIn('import batch_api_same235_state_control_v1 as ctrl',text);self.assertIn("parent['child_launch_started_epoch']=time.time()",text);self.assertIn("'started':identity['started']",text)
 def test_bad_origin_before_lease(self):
  import batch_api_same235_state_control_v1 as p
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as t,patch.object(p.c1,'leased') as lease:
   f=Path(t)/'wrong.json';f.write_text('{}')
   with self.assertRaises(ValueError):p.prepare(SimpleNamespace(origin_plan=f,output=Path(t)/'out'))
   lease.assert_not_called();self.assertFalse((Path(t)/'out').exists())
 def test_new_producer_metadata_sealed(self):
  import run_batch_api_same235_state_control_v1 as p
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);plan={'CPU':'synthetic'};p.original.write(root/'plan.snapshot.json',plan);result={'collection_and_teardown_passed':False};p.seal_report(plan,root,result);actual=p.original.read(root/'report.json');self.assertEqual(actual['same235_state_control_generation'],1);self.assertEqual(actual['buffered_wrapper_generation'],7);self.assertEqual(actual['plan_sha256'],p.original.sha(root/'plan.snapshot.json'));self.assertFalse(actual['underlying_EOS_cause_established']);self.assertFalse(actual['actual_cached_state_handoff_qualified'])
 def test_no_new_sampler(self):
  s=Path(c.client.__file__).read_text();self.assertIn("'temperature':0",s);self.assertIn("'max_tokens':bounds[index]",s);self.assertNotIn('ignore_eos',s);self.assertNotIn('strata_shared_prefix',s)
if __name__=='__main__':unittest.main()
