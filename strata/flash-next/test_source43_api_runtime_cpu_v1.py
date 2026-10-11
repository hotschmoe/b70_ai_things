"""Tiny original-shaped native/API joins and explicit historical byte failure."""
import copy,json,tempfile,unittest,ast,inspect,subprocess,sys,types
import run_source43_api_baseline_v1 as actor
from pathlib import Path
from unittest.mock import patch
import source43_api_evidence_v1 as e
import source43_api_baseline_client_v1 as client
import source43_api_prerequisite_bytes_v1 as history
import c1_source43_api_controller_v1 as ctrl
from serial37_canonical_json_v3 import canonical
class Controls(unittest.TestCase):
 def fixture(self,root):
    out=root/'api-client';out.mkdir();rows=[];trace=[]
    for i in range(8):
        case=client.CASES[i]if i<6 else client.CASES[(i-6)*2];packet=client.body(i,case[1]);index=0 if i in(0,4,6)else 2 if i in(2,5,7)else i;sent=[100+index];generated=[200+index,248046];done={'prompt_read':1,'reused':0,'prompt_tokens':1,'generated':2,'finish':'stop'};facts=e.completion_facts(done,generated,1,[248046],True,False);response={'model':packet['model'],'choices':[{'index':0,'finish_reason':'stop','message':{'content':case[2]}}],'usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3}};row={'case':case[0],'request':packet,'response':response,'started_monotonic_ns':1000 if i>=6 else 1,'finished_monotonic_ns':2000 if i>=6 else 2,'coherent':True};rows.append(row)
        for name,v in [('request',packet),('response',response)]: (out/(name+'-'+str(i)+'.json')).write_text(json.dumps(v))
        trace+=[{'kind':'engine_begin','call':i+1,'epoch':10+i,'engine_pid':123,'submitted_ids':sent,'submitted_ids_sha256':e.digest(sent),'max_new':64,'rendered_prompt':{'ids':sent,'sha256':e.digest(sent),'messages':packet['messages'],'template_kwargs':{'enable_thinking':False}},'rendered_matches_submitted':True,'embeddings':None,'sampling':{'temperature':0},'pinned_eos_ids':[248046]},{'kind':'engine_end','call':i+1,'epoch':10.5+i,'engine_pid':123,'generated_ids':generated,'generated_ids_sha256':e.digest(generated),'engine_done':done,**facts,'done_observation':{'call':i+1,'done':done},'consumer_closed':True,'error':None,'cancelled':False}]
    trace.append({'kind':'engine_close','engine_pid':123,'epoch':19,'clean_exit':True,'exit_code':0,'error':None});models={'data':[{'id':'hotschmoe-dd'},{'id':client.ALIAS}]};(out/'models.json').write_text(json.dumps(models));(out/'report.json').write_text(json.dumps({'passed':True,'errors':[],'results':rows,'models':models}));self.save(root,trace);return trace
 def save(self,root,trace):(root/'engine-token-trace.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in trace))
 def test_eight_native_HTTP_original_shape(self):
    with tempfile.TemporaryDirectory()as td:self.fixture(Path(td));self.assertEqual(e.bind(td)['actual_requests'],8)
 def test_wrong_done_owner_refused(self):
    with tempfile.TemporaryDirectory()as td:r=Path(td);t=self.fixture(r);t[1]['done_observation']['call']=2;self.save(r,t);self.assertRaises(ValueError,e.bind,r)
 def test_missing_extra_foreign_calls(self):
    for mutate in(lambda t:t.pop(1),lambda t:t.append(copy.deepcopy(t[0])),lambda t:t[0].update(engine_pid=True)):
        with tempfile.TemporaryDirectory()as td:r=Path(td);t=self.fixture(r);mutate(t);self.save(r,t);self.assertRaises(ValueError,e.bind,r)
 def test_nonoverlap_clients(self):
    with tempfile.TemporaryDirectory()as td:r=Path(td);self.fixture(r);p=r/'api-client/report.json';v=json.loads(p.read_text());v['results'][7]['started_monotonic_ns']=3000;v['results'][7]['finished_monotonic_ns']=4000;p.write_text(json.dumps(v));self.assertRaises(ValueError,e.bind,r)
 def test_wrong_actual_response_usage(self):
    with tempfile.TemporaryDirectory()as td:r=Path(td);self.fixture(r);p=r/'api-client/response-0.json';v=json.loads(p.read_text());v['usage']={'prompt_tokens':2,'completion_tokens':2,'total_tokens':4};p.write_text(json.dumps(v));self.assertRaises(ValueError,e.bind,r)
 def test_historical_full_roster_mutation(self):
    with tempfile.TemporaryDirectory()as td:r=Path(td);(r/'report.json').write_text('{"passed":true}');(r/'raw.txt').write_text('original');proof=history.capture(r,{'report_sha256':history.sha(r/'report.json')},'normal43_mode0');self.assertFalse(history.rejoin(proof)['old_current_registry_gate_passed']);(r/'raw.txt').write_text('mutated');self.assertRaises(ValueError,history.rejoin,proof)
 def test_historical_missing_extra_and_alias(self):
    with tempfile.TemporaryDirectory()as td:r=Path(td)/'root';r.mkdir();(r/'report.json').write_text('{"passed":true}');proof=history.capture(r,{'report_sha256':history.sha(r/'report.json')},'normal43_mode0');(r/'extra').write_text('new');self.assertRaises(ValueError,history.rejoin,proof);alias=Path(td)/'alias';alias.symlink_to(r);self.assertRaises(ValueError,history.tree,alias)
 def test_original_owned_lifecycle_changes_exact(self):
    source=Path(ctrl.original.__file__).read_text();original={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    for name in('leased','absent','stop_lock','stop_receipt_checked','poll_owned_run','native_close_proven'):
        self.assertEqual(inspect.unwrap(ctrl._PORTS[name]).__code__.co_argcount,len(original[name].args.args))
    self.assertIn('terminal-inspection.json',ctrl._PORTS['owned_stop'].__code__.co_consts)
 def test_actual_stop_saved_inspection(self):
    with tempfile.TemporaryDirectory()as td:
        root=Path(td);(root/'launch.json').write_text(json.dumps({'container':'owned','prepared_sha256':'x'}));obj={'Config':{'Labels':{'b70.c1.prepared':'x'}},'State':{'Running':False,'ExitCode':0,'OOMKilled':False}};ns=ctrl._PORTS;before={k:ns[k]for k in('inspected','run','absent','native_close_proven')}
        try:
            ns.update(inspected=lambda _:obj,run=lambda *a,**k:None,absent=lambda _:True,native_close_proven=lambda _:True);self.assertTrue(ns['owned_stop'](root)['clean_exit']);self.assertEqual(ctrl.read(root/'terminal-inspection.json'),obj)
        finally:ns.update(before)

 def test_real_supervisor_retired_after_owned_stop_error(self):
    with tempfile.TemporaryDirectory()as td:
        root=Path(td);(root/'launch.json').write_text('{}');proc=subprocess.Popen([sys.executable,'-c','pass']);attempts=[];errors=[]
        def stop(_):
            attempts.append(1)
            if len(attempts)==1:raise InterruptedError('injected actual owned inspection uncertainty')
            (root/'stop.json').write_text('{}');return {'removed':True}
        with patch.object(actor.c,'owned_stop',side_effect=stop),patch.object(actor.c,'stop_receipt_checked',return_value={'removed':True}),patch.object(actor.time,'sleep',return_value=None):actor.retire_supervisor(root,proc,errors)
        self.assertEqual(proc.poll(),0);self.assertEqual(len(attempts),2);self.assertTrue(any('InterruptedError'in row for row in errors))
 def test_true_original_lifecycle_undo(self):
    # Exact declaration evidence: seven original functions have no body edits.
    original_tree=ast.parse(Path(ctrl.original.__file__).read_text());names=('leased','inspected','absent','stop_lock','stop_receipt_checked','poll_owned_run','native_close_proven')
    for node in original_tree.body:
        if isinstance(node,ast.FunctionDef)and node.name in names:
            expected={**vars(ctrl.original)};exec(compile(ast.fix_missing_locations(ast.Module([copy.deepcopy(node)],type_ignores=[])),'[test-original]','exec'),expected)
            a=inspect.unwrap(ctrl._PORTS[node.name]).__code__;b=inspect.unwrap(expected[node.name]).__code__;
            def normalized(code):return(code.co_code,code.co_names,tuple(normalized(v)if isinstance(v,types.CodeType)else v for v in code.co_consts))
            self.assertEqual(normalized(a),normalized(b))

if __name__=='__main__':unittest.main()
