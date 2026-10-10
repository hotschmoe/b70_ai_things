import ast,copy,json,queue,threading,time,types,unittest
from pathlib import Path
import eos_waiter_observer40_v1 as o
import eos_waiter_terminal_adjudication40_v1 as a
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T193019Z-1kl6uepo/source/serve/server.py')

class Controls(unittest.TestCase):
 def engine(self):
  tree=ast.parse(SDK.read_bytes());method=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='StrataEngine' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_release_slot_when_done')
  def tags(line,rid,generation):
   tags=a.fields(line)
   if tags.get('rid')!=str(rid) or tags.get('slotgen')!=str(generation):raise ValueError('stale')
  ns={'queue':queue,'threading':threading,'time':time,'EngineDied':RuntimeError,'batch_identity_tags':tags};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='__future__']+[method],type_ignores=[]),str(SDK),'exec'),ns)
  class Engine:pass
  Engine._release_slot_when_done=types.FunctionType(o.find_code(compile(SDK.read_bytes(),str(SDK),'exec'),'_release_slot_when_done'),ns,'_release_slot_when_done');engine=Engine();engine.proc=types.SimpleNamespace(pid=200);engine.gen=1;engine.strict_batch_identity=True;engine.slot_q=[queue.Queue()];engine.slot_busy=[True];engine.slot_held=[[]];engine.slot_identity=[(1,9,4)];engine.slot_cv=threading.Condition();engine.stops=[];engine._stop_slot_identity=lambda slot:engine.stops.append(slot)
  server=types.SimpleNamespace(__file__=str(SDK),StrataEngine=Engine);events=[];lock=threading.Lock()
  def emit(row):
   with lock:events.append({**row,'sequence':len(events)+1,'epoch':time.monotonic()})
  handle=o.install(server,emit,lambda engine:{'call':3});return engine,handle,events
 def test_actual_source_waiter_same_returns_and_state(self):
  engine,h,events=self.engine();engine.slot_q[0].put('BDONE 0 2 cancel 1.0 rid=9 slotgen=4');engine._release_slot_when_done(0,[7,8,9]);h.threads[0][1].join(2)
  self.assertEqual(engine.stops,[0]);self.assertEqual(engine.slot_held,[[7,8]]);self.assertEqual(engine.slot_busy,[False]);self.assertTrue(h.proof()['all_waiter_threads_retired']);self.assertTrue(h.proof()['observer_passed']);self.assertEqual(events[-1]['kind'],'eos_waiter_exit')
 def test_stale_packet_is_not_accepted_as_own_terminal(self):
  engine,h,events=self.engine();engine.slot_q[0].put('BDONE 0 2 cancel 1.0 rid=999 slotgen=4');engine.slot_q[0].put('BDONE 0 2 cancel 1.0 rid=9 slotgen=4');engine._release_slot_when_done(0,[7,8,9]);h.threads[0][1].join(2);self.assertEqual([x['line'] for x in events if x['kind']=='eos_waiter_queue_return'],['BDONE 0 2 cancel 1.0 rid=999 slotgen=4','BDONE 0 2 cancel 1.0 rid=9 slotgen=4']);self.assertFalse(h.proof()['threads'][0]['alive'])
 def test_sink_failure_latches_without_changing_waiter(self):
  engine,h,_=self.engine();h.emit=lambda row:(_ for _ in ()).throw(ValueError('sink'));engine.slot_q[0].put('BDONE 0 2 cancel 1.0 rid=9 slotgen=4');engine._release_slot_when_done(0,[7,8,9]);h.threads[0][1].join(2);self.assertFalse(h.proof()['observer_passed']);self.assertEqual(engine.slot_busy,[False])
 def test_missing_call_context_fails_observation_preserves_original(self):
  engine,h,_=self.engine();h.context=lambda engine:{'call':None};engine.slot_q[0].put(None);engine._release_slot_when_done(0,[7,8,9]);time.sleep(.02);self.assertFalse(h.proof()['observer_passed']);self.assertEqual(engine.slot_busy,[False])
 def test_code_mutation_refused(self):
  engine,h,_=self.engine();server=types.SimpleNamespace(__file__=str(SDK),StrataEngine=type('Wrong',(),{'_release_slot_when_done':lambda *args:None}));self.assertRaises(ValueError,o.install,server,lambda x:None,lambda e:{'call':1})
 def test_phase_proof_exact_call_roster(self):
  engine,h,_=self.engine();engine.slot_q[0].put('BDONE 0 2 cancel 1.0 rid=9 slotgen=4');engine._release_slot_when_done(0,[7,8,9]);h.threads[0][1].join(2);self.assertEqual(h.proof([3])['waiters'],1);self.assertEqual(h.proof([99])['waiters'],0);self.assertRaises(ValueError,h.proof,[True]);self.assertRaises(ValueError,h.proof,[3,3])
 def test_unstarted_thread_cannot_prove_retirement(self):
  h=o.Observer(lambda x:None,lambda e:{'call':1});h.waiters=1;h.waiter_calls={1:1};h.threads=[(1,threading.Thread(target=lambda:None,daemon=True))];self.assertFalse(h.proof()['all_waiter_threads_retired'])
 def binding_fixture(self):
  engine,h,events=self.engine();line='BDONE 0 2 cancel 1.0 rid=9 slotgen=4';engine.slot_q[0].put(line);engine._release_slot_when_done(0,[7,8,9]);h.threads[0][1].join(2)
  # Source-shaped records exercise exact waiter gate, independently of old full native grammar.
  for e in events:e['sequence']+=10
  native={'kind':'native_receive','engine_pid':200,'line':line,'sequence':9};events.insert(0,native)
  end={'engine_pid':200,'engine_generation':1,'rid':9,'consumer_closed':True,'cancelled':False,'generated_ids':[8,9],'pinned_eos_ids':[9],'error':copy.deepcopy(a.LEGACY),'engine_last':{'generated':1,'finish':'length'},'sequence':8}
  terminal={'sequence':9,'line':line};tracker=a.Terminals(events,h.proof());return tracker,terminal,end
 def full_fixture(self):
  from test_api_owned_terminal_cpu_v2 import Controls as Original
  t,terminal,end=self.binding_fixture();rows=Original().rows()
  for row in rows:
   row['engine_pid']=200
   if 'call' in row:row['call']=3
   if 'line' in row:row['line']=row['line'].replace('rid=1','rid=9').replace('slotgen=1','slotgen=4')
  rows[0]['submitted_ids']=[7];rows[1]['line']='BGEN 0 32 seed=1 fresh=1 rid=9 7';rows[2]['line']='T 8 rid=9 slotgen=4';rows[3]['line']='DONE 1 1 0 0 length 0 0 0 0 0 0 0 0 1 0 rid=9 slotgen=4';rows[5]['line']='BT 0 9 rid=9 slotgen=4'
  native=rows[6];native['line']=terminal['line'];stop=rows[7];stop['sequence']=7;stop['line']='BSTOP 0 rid=9 slotgen=4';end['kind']='engine_end';end['call']=3;end['sequence']=8
  observer=[x for x in t.events if x['kind'].startswith('eos_waiter_')]
  observer[0]['sequence']=9;observer[1]['sequence']=10;native['sequence']=11;observer[2]['sequence']=12;observer[3]['sequence']=13
  allrows=rows[:6]+[stop,end,observer[0],observer[1],native,observer[2],observer[3]]
  return allrows,t.proof
 def test_full_original_grammar_late_waiter_adjudication_preserves_raw(self):
  rows,proof=self.full_fixture();before=copy.deepcopy(rows);result=a.adjudicate(rows,proof);self.assertTrue(result['named_source40_waiter_adjudication_passed']);self.assertEqual(rows,before);self.assertEqual(result['actual_API_terminal_associations'][0]['legacy_observer_error_preserved'],a.LEGACY)
 def test_full_missing_native_or_wrong_generated_count_refused(self):
  for mode in ('missing','count'):
   rows,proof=self.full_fixture()
   if mode=='missing':rows=[x for x in rows if not(x['kind']=='native_receive' and x['line'].startswith('BDONE '))]
   else:next(x for x in rows if x['kind']=='engine_end')['generated_ids']=[8]
   with self.assertRaises(ValueError):a.adjudicate(rows,proof)
 def test_new_owner_handoff_needs_actual_distinct_admission(self):
  t,terminal,end=self.binding_fixture();exit=next(x for x in t.events if x['kind']=='eos_waiter_exit');exit.update(busy=True,current_slot_identity=[1,10,5]);exit['sequence']=20
  t.events.insert(-1,{'kind':'native_receive','engine_pid':200,'line':'BADM 0 1 rid=10 slotgen=5','sequence':19})
  t.waiter_terminal_binding(3,terminal,end);self.assertTrue(t.waiter_bindings[0]['busy_snapshot'])
 def test_behavioral_named_waiter_gate(self):
  t,terminal,end=self.binding_fixture();t.waiter_terminal_binding(3,terminal,end);self.assertEqual(t.waiter_bindings[0]['legacy_observer_error_preserved'],a.LEGACY)
 def test_client_cancel_retirement_no_EOS_error_waiver(self):
  t,terminal,end=self.binding_fixture();end.update(cancelled=True,error=None,pinned_eos_ids=[]);t.waiter_terminal_binding(3,terminal,end);self.assertTrue(t.waiter_bindings[0]['actual_client_cancelled']);self.assertFalse(t.waiter_bindings[0]['named_EOS_error_adjudicated'])
  t,terminal,end=self.binding_fixture();end['cancelled']=True;self.assertRaises(ValueError,t.waiter_terminal_binding,3,terminal,end)
 def test_foreign_owner_refused(self):
  t,terminal,end=self.binding_fixture();end['rid']=99;self.assertRaises(ValueError,t.waiter_terminal_binding,3,terminal,end)
 def test_thread_still_alive_refused(self):
  t,terminal,end=self.binding_fixture();t.proof['threads'][0]['alive']=True;self.assertRaises(ValueError,t.waiter_terminal_binding,3,terminal,end)
 def test_foreign_busy_array_refused(self):
  t,terminal,end=self.binding_fixture();next(e for e in t.events if e['kind']=='eos_waiter_exit')['same_original_busy_array']=False;self.assertRaises(ValueError,t.waiter_terminal_binding,3,terminal,end)
 def test_new_owner_busy_snapshot_not_old_false(self):
  t,terminal,end=self.binding_fixture();next(e for e in t.events if e['kind']=='eos_waiter_exit')['busy']=True;self.assertRaises(ValueError,t.waiter_terminal_binding,3,terminal,end)
 def test_unknown_error_not_eos_adjudicated(self):
  t,terminal,end=self.binding_fixture();end['error']={'type':'EngineDied','message':'x'};self.assertRaises(ValueError,t.waiter_terminal_binding,3,terminal,end)
 def test_wrong_queue_BDONE_refused(self):
  t,terminal,end=self.binding_fixture();next(e for e in t.events if e['kind']=='eos_waiter_queue_return')['line']='BDONE 0 2 cancel 1.0 rid=99 slotgen=4';self.assertRaises(ValueError,t.waiter_terminal_binding,3,terminal,end)

if __name__=='__main__':unittest.main()
