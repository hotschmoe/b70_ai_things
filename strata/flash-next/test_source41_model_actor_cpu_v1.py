import copy,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import source41_model_cli_contract_v1 as c
import run_source41_model_cli_v1 as a
import prefill_chunk64_512_control_v2 as template

class Controls(unittest.TestCase):
    def fixture(self):
        cfg=template.expected_config('chunk64',{'inherited_math_env':{}},18082);m={'cards':[0],'alias':cfg['aliases'][0],'pack':'/owned/pack','profile':'chunk64','runtime':{'image':'sha256:'+'a'*64}}
        inputs={'short':[1,2],'prefill_about512':[3]*524};return m,cfg,inputs
    def rows(self):
        m,cfg,inputs=self.fixture();_,recipe=a.phase_recipe('source41-off',m,cfg,inputs,'a'*64);rows=[]
        for req in recipe['requests']:
            done='DONE 1 %d 0.1 0.2 stop'%len(req['ids']);rows.append({'ordinal':req['ordinal'],'raw':{'ids':req['ids'],'fresh':1,'pin':None,'cancel_requested':None,'stop_sent':False,'command':'GEN 64 fresh=1 ckpt=1 temperature=0 logprobs=20 '+' '.join(map(str,req['ids'])),'stdout':['T 7',done],'stdout_seconds':[0.1,0.2],'output_ids':[7]}})
        return rows,recipe
    def test_real_off_protocol_and_equivalence(self):
        rows,recipe=self.rows();views=c.requests(rows,recipe['requests']);self.assertTrue(c.equivalence(views,copy.deepcopy(views))['OFF_equivalence_passed'])
    def test_changed_output_finish_budget_input_refused(self):
        rows,recipe=self.rows()
        for key in('fresh','pin','output','budget','clock','command','ordinal'):
            changed=copy.deepcopy(rows);raw=changed[0]['raw']
            if key=='fresh':raw['fresh']=True
            elif key=='pin':raw['pin']=0
            elif key=='output':raw['output_ids']=[8]
            elif key=='budget':raw['stdout'][-1]='DONE 65 2 0.1 0.2 length'
            elif key=='clock':raw['stdout_seconds']=[.2,.1]
            elif key=='command':raw['command']+=' pin=0'
            else:changed[0]['ordinal']=True
            with self.subTest(key=key),self.assertRaises(ValueError):c.requests(changed,recipe['requests'])
    def test_cross_arm_and_repeat_mismatch(self):
        rows,recipe=self.rows();views=c.requests(rows,recipe['requests']);bad=copy.deepcopy(views);bad[0]['output_ids']=[9]
        with self.assertRaises(ValueError):c.equivalence(views,bad)
        with self.assertRaises(ValueError):c.equivalence(bad,bad)
    def test_true41_descriptor_never_old_runtime_view(self):
        m,cfg,inputs=self.fixture();descriptor,recipe=a.phase_recipe('source41-off',m,cfg,inputs,'a'*64)
        self.assertNotIn('combined_generation',descriptor);self.assertEqual(recipe['engine_generation'],41);self.assertEqual(recipe['native_env']['STRATA_QSA_INDEXER_MATERIALIZED_HALF'],'0');self.assertFalse(recipe['observer_enabled'])
    def test_fidelity_observer_separate_mode_and_exact_ids(self):
        m,cfg,inputs=self.fixture();descriptor,recipe=a.phase_recipe('source41-original-fidelity',m,cfg,inputs,'a'*64)
        self.assertEqual(recipe['native_env']['STRATA_QSA_INDEXER_MATERIALIZED_HALF'],'1');self.assertTrue(recipe['observer_enabled']);self.assertEqual([r['ids']for r in recipe['requests']],[[248045,8678,198,15666]]*4);self.assertTrue(all(r['pin']is None and r['max_new']==1 for r in recipe['requests']))
    def test_guardian_actual_thread_failure_latches_and_retires(self):
        calls=[];notice=threading.Event()
        def read():
            calls.append(1)
            if len(calls)>1:raise ValueError('bad page preserved')
            return {'passed':True}
        guard=c.Guardian(read,notice.set,.01);guard.start();self.assertTrue(notice.wait(1));proof=guard.close();self.assertFalse(proof['passed']);self.assertTrue(proof['thread_retired']);self.assertIn('bad page',proof['errors'][0])
    def test_guardian_normal_actual_thread_retirement(self):
        guard=c.Guardian(lambda:{'passed':True},lambda:self.fail('unexpected failure'),.01);guard.start();proof=guard.close();self.assertTrue(proof['passed']);self.assertTrue(proof['thread_retired']);self.assertEqual(len(proof['samples']),1)
    def test_guardian_initial_failure_cannot_start(self):
        def bad():raise ValueError('initial page')
        guard=c.Guardian(bad,lambda:None,.01)
        with self.assertRaises(ValueError):guard.start()
        self.assertIsNone(guard.thread)

class PipeControls(unittest.TestCase):
    def exercise(self,body,action=None):
        import sys
        from native_source41_cli_protocol_v1 import Protocol
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);active=[]
            def cleanup():
                if active and active[0].p.poll()is None:active[0].p.terminate()
            protocol=None
            try:
                protocol=Protocol([sys.executable,'-u','-c',body],root,cleanup,ready_timeout=2,active=active)
                if action:return action(protocol,root)
            finally:
                protocol=protocol or(active[0]if active else None)
                if protocol:protocol.close(failed=action is None)
    def test_actual_native_pipe_ready_request_quit_eof(self):
        body="import sys;print('READY native');\nfor line in sys.stdin:\n if line.startswith('QUIT'):break\n ids=line.split('logprobs=20 ')[1].split();print('T 7');print('DONE 1 '+str(len(ids))+' 0.1 0.2 stop',flush=True)"
        def run(p,root):
            raw=p.request('request-1',[1,2],fresh=1,max_new=64,pin=None);self.assertEqual(raw['output_ids'],[7]);self.assertEqual(p.close(),0);self.assertTrue(p.capture['passed']);self.assertTrue(p.capture['eof']);self.assertTrue(p.capture['reader_retired'])
        self.exercise(body,run)
    def test_actual_pipe_early_eof_refuses_ready(self):
        with self.assertRaises(ValueError):self.exercise("print('not ready')")
    def test_actual_pipe_oversized_line_fails_bounded_before_ready(self):
        with self.assertRaises(ValueError):self.exercise("print('x'*(1048576+2))")
    def test_actual_partial_semantic_line_refused(self):
        with self.assertRaises(ValueError):self.exercise("import sys;sys.stdout.write('READY native');sys.stdout.flush()")

if __name__=='__main__':unittest.main()
