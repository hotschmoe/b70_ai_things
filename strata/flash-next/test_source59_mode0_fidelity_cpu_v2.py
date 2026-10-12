"""Regression: execute exact protected parent's constructor call; mock only Popen."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_source59_mode0_fidelity_v1 as old
import run_source59_mode0_fidelity_v2 as actor
import native_source41_cli_protocol_v4 as pipe
HERE=Path(__file__).resolve().parent
class Constructor(unittest.TestCase):
 def call(self,protocol):
  tree=ast.parse((HERE/'run_source41_model_cli_v5.py').read_bytes())
  fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='run_phase')
  protected=next(x for x in fn.body if isinstance(x,ast.Try))
  calls=[x for x in ast.walk(protected)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='Protocol']
  self.assertEqual(len(calls),1);node=copy.deepcopy(calls[0]);self.assertEqual([x.arg for x in node.keywords],['active','pipe_bytes','pipe_lines'])
  arguments=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x)for x in ('cmd','directory','cleanup','active','plan')],kwonlyargs=[],kw_defaults=[],defaults=[])
  wrapper=ast.FunctionDef(name='invoke',args=arguments,body=[ast.Return(value=node)],decorator_list=[])
  ns={'Protocol':protocol};exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),'[exact-protected-parent-Protocol-call]','exec'),ns);return ns['invoke']
 def fixture(self,path):
  source=Path('/tmp/b70-source59-mode0-fidelity-current33-construction-v1.json')
  proof=json.loads(source.read_bytes());row=proof['commands'][0];plan=row['recipe'];actor.c.write(path/'recipe.json',plan);return row['native_command'],plan
 def test_old_counterexample_and_corrected_actual_constructor(self):
  for protocol,wanted in ((old.Protocol,TypeError),(actor.Protocol,OSError)):
   with tempfile.TemporaryDirectory()as temp:
    directory=Path(temp);command,plan=self.fixture(directory);active=[]
    with patch.object(pipe.subprocess,'Popen',side_effect=OSError('CPU-only exact Popen sentinel'))as popen:
     try:self.call(protocol)(command,directory,lambda:self.fail('No process exists to clean'),active,plan)
     except wanted as error:
      if protocol is old.Protocol:self.assertIn('multiple values',str(error));self.assertEqual(popen.call_count,0)
      else:
       self.assertEqual(str(error),'CPU-only exact Popen sentinel');self.assertEqual(popen.call_count,1)
       self.assertEqual(popen.call_args.args[0],command)
       failure=actor.c.read(directory/'launch-failure.json');self.assertIs(failure['actual_client_started'],False);self.assertEqual(failure['command'],command)
       tb=error.__traceback__;instance=None
       while tb:
        if 'self'in tb.tb_frame.f_locals and isinstance(tb.tb_frame.f_locals['self'],actor.Protocol):instance=tb.tb_frame.f_locals['self']
        tb=tb.tb_next
       self.assertIsNotNone(instance);self.assertEqual((instance.pipe_bytes,instance.pipe_lines),(128<<20,1000000));self.assertTrue(instance.err.closed);self.assertEqual(instance.threads,[])
      self.assertEqual(active,[])
     else:self.fail('Actual constructor should reach no-launch Popen sentinel')
 def test_narrow_successor_deltas(self):
  before=Path(old.__file__).read_text();after=Path(actor.__file__).read_text()
  normalized=after.replace('source-plan-v2.json','source-plan-v1.json').replace('source59_mode0_fidelity_raw_v2','source59_mode0_fidelity_raw_v1')
  normalized=normalized.replace('  kwargs.update(pipe_bytes=128<<20,pipe_lines=1000000)\n  super().__init__(command,directory,*args,**kwargs)','  super().__init__(command,directory,*args,pipe_bytes=128<<20,pipe_lines=1000000,**kwargs)')
  self.assertEqual(normalized,before)
  self.assertEqual((HERE/'source59_mode0_fidelity_raw_v2.py').read_text().replace('run_source59_mode0_fidelity_v2','run_source59_mode0_fidelity_v1'),(HERE/'source59_mode0_fidelity_raw_v1.py').read_text())
  self.assertEqual((HERE/'validate_source59_mode0_fidelity_v2.py').read_text().replace('run_source59_mode0_fidelity_v2','run_source59_mode0_fidelity_v1').replace('source59_mode0_fidelity_raw_v2','source59_mode0_fidelity_raw_v1').replace('source-plan-v2.json','source-plan-v1.json'),(HERE/'validate_source59_mode0_fidelity_v1.py').read_text())
