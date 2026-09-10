"""CPU fixtures only: no HTTP, key access, serving or GPU execution."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import serve


class TrialTests(unittest.TestCase):
    def test_unfinalized(self):
        with self.assertRaisesRegex(RuntimeError, 'waits'):
            serve.verify_inputs({'schema':1,'scope':serve.SCOPE,'user_authorized_monitored_trial':True,
                'long200k_qualified':False,'unrestricted_model_quality_qualified':False})

    def test_no_quality_upgrade(self):
        with self.assertRaisesRegex(RuntimeError, 'limits'):
            serve.verify_inputs({'schema':1,'scope':serve.SCOPE,'user_authorized_monitored_trial':True,
                'long200k_qualified':True,'unrestricted_model_quality_qualified':False})

    def test_failure_exit_is_not_backend_success(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            self.assertEqual(serve.write_exit_receipt(p,0,False),1)
            self.assertEqual((p/'backend-exit.rc').read_text(),'0\n')
            self.assertEqual((p/'exit.rc').read_text(),'1\n')
            self.assertEqual(serve.write_exit_receipt(p,0,True),0)
            self.assertEqual(serve.write_exit_receipt(p,1,True),1)

    def test_recovery_required_and_forced_stop_retained(self):
        pause={'user_requested_pause':True,'long200k_qualified':False,'server_lifecycle_rc':0,
            'owned_container_absent':True,'normal_stop':False,'forced_engine_stop':True,
            'requires_additional_recovery':True}
        with patch.object(serve,'read',side_effect=[pause,{'passed':False}]):
            with self.assertRaisesRegex(RuntimeError,'recovery incomplete'):
                serve.verify_pause_recovery({'stopped200k_receipt':'p','recovery_receipt':'r'})
        pause['normal_stop']=True
        with patch.object(serve,'read',return_value=pause):
            with self.assertRaisesRegex(RuntimeError,'normal_stop'):
                serve.verify_pause_recovery({'stopped200k_receipt':'p'})

    def test_live_planned_features_and_startup_scope(self):
        base=serve.ROOT/'results/bang_recurrence_testing_20260910'
        plan=serve.read(base/'native5802-backend200k/plan.json')
        serve.verify_plan(plan,200000)
        cmd=serve.service_command(plan,Path('/tmp/trial-fixture'))
        self.assertEqual(serve.value(cmd,'--port'),'18124')
        jobs=serve.startup_jobs({'plan100k':str(base/'native5802-backend100k-v2/plan.json')},Path('/tmp/trial-fixture'),'fixture')
        self.assertEqual(len(jobs),5)
        self.assertEqual(sum('quality' in j['name'] for j in jobs),2)

if __name__=='__main__':unittest.main()
