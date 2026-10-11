import unittest,copy
from unittest.mock import patch
import test_source43_warm_pti_cpu_v1 as old
import pti_owned_temporal_control_v2 as helper
import run_source43_warm_pti_v2 as actor
import validate_source43_warm_pti_v2 as reader
import pti_temporal_fixture_binding_v2 as fixture
class Controls(old.Tests):
    def setUp(self):self.patches=[patch.object(old,'h',helper),patch.object(old,'a',actor),patch.object(old,'v',reader)];[p.start()for p in self.patches]
    def tearDown(self):[p.stop()for p in reversed(self.patches)]
class NewControls(unittest.TestCase):
    def test_real_closed_fixture_proof(self):
        result=fixture.binding();self.assertTrue(result['actual_CLI_resume_accepted']);self.assertFalse(result['model_runtime_qualification_transferred'])
    def test_wrong_command_or_state_refused(self):
        body=fixture.read_unique(fixture.FIXTURE/'inside.json');changed=copy.deepcopy(body);changed['commands'][1]['command']=['/pti/bin/unitrace','--session',body['session'],'--resume']
        with self.assertRaises(ValueError):fixture.check_inside(changed)
        changed=copy.deepcopy(body);changed['resumed']['state']=1
        with self.assertRaises(ValueError):fixture.check_inside(changed)
    def test_postrun_predicate_uses_actual_context_without_final_report(self):
        import tempfile,json,os
        from pathlib import Path
        class Watch:
            def __init__(self,*x):self.latched=False;self.errors=[]
            def start(self):pass
            def close(self):return {'passed':True}
        with tempfile.TemporaryDirectory()as d:
            out=Path(d);directory=out/actor.PHASES[1];directory.mkdir();(directory/'host-traces').mkdir();(directory/'requests.json').write_text(json.dumps([{'raw':{}},{'raw':{}}]));(directory/'recipe.json').write_text(json.dumps({'PTI_enabled':True}));(directory/'engine.combined.log').write_text('tiny');(directory/'command.json').write_text('[]')
            require_context=[]
            def control(path,recipe,terminal,context):
                self.assertFalse((out/'report.json').exists());self.assertEqual(context['producer_pid'],os.getpid());self.assertEqual(context['source_plan_sha256'],'actual-source');self.assertGreater(context['producer_start_ticks'],0);require_context.append(context)
            runtime={'intake_path':'tiny'}
            with patch.object(actor,'Watcher',Watch),patch.object(actor,'_PORTS',{'native_phase':lambda *x,**k:{'terminal':{}}}),patch.object(actor,'native_pid',return_value=1),patch.object(actor.collector,'collect',return_value={}),patch.object(actor,'_RUNTIME',runtime),patch.object(actor,'tool',type('Tool',(),{'binding':staticmethod(lambda x:runtime)})()),patch.object(actor,'publish_final',side_effect=lambda path,proof,out:proof),patch.object(reader,'graph_binding',return_value={}),patch.object(reader,'control_binding',side_effect=control),patch.object(reader,'clock_window',return_value={}):
                actor.run_phase(out,actor.PHASES[1],None,None,None,'actual-source')
            self.assertEqual(len(require_context),1)
if __name__=='__main__':unittest.main()
