import copy,json,tempfile,unittest
from pathlib import Path
import source41_lpdisabled_contract_v1 as omission
import collect_source41_lpdisabled_host_v1 as collector
import run_source41_host_lpdisabled_v1 as actor
import test_source41_model_actor_cpu_v3 as model_fixture
import test_host_critical_path36_cpu_v1 as trace_fixture
class Controls(unittest.TestCase):
 def test_exact_omitted_command_and_not_zero(self):
  command=omission._REQUEST['request_command']([1,2],64,1,None);self.assertEqual(command,'GEN 64 fresh=1 ckpt=1 temperature=0 1 2');self.assertNotIn('logprobs',command)
 def rows(self):
  rows,recipe=model_fixture.Controls().rows()
  for row in rows:row['raw']['command']=row['raw']['command'].replace(' logprobs=20','');row['raw']['LP']=[]
  return rows,recipe
 def test_omitted_request_positive_and_explicit_zero_or_lp_refused(self):
  rows,recipe=self.rows();self.assertEqual(len(omission.requests(rows,recipe['requests'])),4)
  for kind in('zero','twenty','lp'):
   bad=copy.deepcopy(rows)
   if kind=='zero':bad[0]['raw']['command']=bad[0]['raw']['command'].replace('temperature=0 ','temperature=0 logprobs=0 ')
   elif kind=='twenty':bad[0]['raw']['command']=bad[0]['raw']['command'].replace('temperature=0 ','temperature=0 logprobs=20 ')
   else:bad[0]['raw']['LP']=['LP -1'];bad[0]['raw']['stdout'].insert(1,'LP -1');bad[0]['raw']['stdout_seconds']=[.1,.15,.2]
   with self.subTest(kind=kind),self.assertRaises(ValueError):omission.requests(bad,recipe['requests'])
 def test_actual_pipe_sends_omitted_field_and_retains_eof(self):
  import sys
  active=[]
  with tempfile.TemporaryDirectory()as t:
   root=Path(t)
   def cleanup():
    if active and active[0].p.poll()is None:active[0].p.terminate()
   body="import sys;print('READY native',flush=True);\\nfor line in sys.stdin:\\n if line.startswith('QUIT'):break\\n if 'logprobs=' in line:raise RuntimeError('unexpected explicit logprobs')\\n print('T 7');print('DONE 1 2 0.1 0.2 stop',flush=True)".replace('\\n','\n')
   p=omission.Protocol([sys.executable,'-u','-c',body],root,cleanup,active=active,ready_timeout=2)
   try:raw=p.request('request-1',[1,2],fresh=1,max_new=64,pin=None);self.assertEqual(raw['LP'],[]);self.assertNotIn('logprobs=',raw['command'])
   finally:p.close()
   sent=[json.loads(line)for line in(root/'native.stdin.jsonl').read_text().splitlines()];self.assertNotIn('logprobs=',sent[0]['packet']);self.assertTrue(p.capture['passed']);self.assertTrue(p.capture['eof'])
 def fixture(self,root):
  path,log,request=trace_fixture.HostTests().fixture(root);rows=[json.loads(line)for line in path.read_text().splitlines()];events=[e for e in rows[1:-1]if e['kind']not in(trace_fixture.K['LPReadback'],trace_fixture.K['LPCPU'])]
  for seq,event in enumerate(events):event['seq']=seq;event['ns']=100+seq*100
  rows=[dict(rows[0],pid=1,events=len(events)),*events,dict(rows[-1],records=len(events))];path.write_text('\n'.join(json.dumps(row)for row in rows)+'\n');request['LP']=[];request['stdout']=['T 16',request['done']];return path,log,request,rows
 def test_actual_fixture_no_lp_rows_or_scopes_positive(self):
  with tempfile.TemporaryDirectory()as t:
   path,log,request,rows=self.fixture(Path(t));result=collector.collect([path],log,'a'*64,[request],[[0,0,48]],1);self.assertTrue(result['passed']);self.assertTrue(result['LP_protocol_and_scope_absence_recollected'])
 def test_lp_protocol_or_scope_reintroduction_refused(self):
  for kind in('raw','stdout','scope'):
   with tempfile.TemporaryDirectory()as t:
    path,log,request,rows=self.fixture(Path(t))
    if kind=='raw':request['LP']=['LP 0']
    elif kind=='stdout':request['stdout'].insert(1,'LP 0')
    else:rows[1]['kind']=trace_fixture.K['LPReadback'];path.write_text('\n'.join(json.dumps(row)for row in rows)+'\n')
    with self.subTest(kind=kind),self.assertRaises(ValueError):collector.collect([path],log,'a'*64,[request],[[0,0,48]],1)
 def test_own_runtime_protocol_and_h36_only_toggle(self):
  m,cfg,inputs=model_fixture.Controls().fixture();left=actor.phase_recipe(actor.PHASES[0],m,cfg,inputs,'a'*64);right=actor.phase_recipe(actor.PHASES[1],m,cfg,inputs,'a'*64);self.assertIs(actor._PORTS['Protocol'],omission.Protocol);self.assertEqual(left[1]['requests'],right[1]['requests']);self.assertEqual(left[1]['native_argv'],right[1]['native_argv']);self.assertEqual(left[1]['materialized_mode'],0);self.assertFalse(left[1]['observer_enabled']);self.assertIn('omitted',left[1]['logprobs_policy'])
if __name__=='__main__':unittest.main()
