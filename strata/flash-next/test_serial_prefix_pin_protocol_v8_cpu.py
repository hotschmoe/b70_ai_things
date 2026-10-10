"""V8 client pin presence and actual source35 ceiling contracts; CPU only."""
import ast,inspect,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import serial_prefix_qualification_v8 as q
import qualify_serial_prefix_v8 as parent
import test_ple_prompt35_cpu_v1 as source35


class ProtocolTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):source35.SourceTests.setUpClass();cls.source=source35.SourceTests.source
 @classmethod
 def tearDownClass(cls):source35.SourceTests.tearDownClass()
 def keys(self,command):return {word.split('=',1)[0]:word.split('=',1)[1] for word in command.split() if '=' in word}
 def parsed_ceiling(self,command,prompt):
  keys=self.keys(command);fresh=keys['fresh']=='1';present='pin' in keys;pin=int(keys.get('pin',0))
  # Exact source-expression branch fixture, not an invented native execution.
  header=(self.source/'sycl/include/strata/core/batch_public_prefix.hpp').read_text();self.assertIn('return fresh?0:(pin_present?pin:prompt-1);',header)
  return 0 if fresh else pin if present else prompt-1
 def test_absent_default_zero_positive_are_distinct_source_contracts(self):
  ids=list(range(49));absent=q.request_command(ids);zero=q.request_command(ids,pin=0);positive=q.request_command(ids,pin=21)
  self.assertNotIn('pin',self.keys(absent));self.assertEqual(self.keys(zero)['pin'],'0');self.assertEqual(self.keys(positive)['pin'],'21');self.assertEqual([self.parsed_ceiling(c,49) for c in (absent,zero,positive)],[48,0,21])
 def test_fresh_zero_ceiling_preserved_independent_of_pin_presence(self):
  for pin in (None,0,21):self.assertEqual(self.parsed_ceiling(q.request_command(list(range(49)),fresh=1,pin=pin),49),0)
 def test_actual_parser_distinguishes_absent_from_explicitzero(self):
  text=(self.source/'sycl/src/program/generate.cpp').read_text();self.assertIn('req_pin_bad = false,req_pin_present=false;',text);start=text.index('else if (key == "pin")');self.assertIn('req_pin_present=true;',text[start:start+520]);self.assertIn('batch_public::ceiling(req_fresh!=0,req_pin_present,req_pin,n)',text)
 def test_actual_Protocol_request_does_not_reinsert_default_zero(self):
  self.assertIsNone(inspect.signature(q.Protocol.request).parameters['pin'].default)
  with tempfile.TemporaryDirectory() as name:
   p=q.Protocol.__new__(q.Protocol);p.directory=Path(name);p.stderr=[];p.ready={'CPU':'mock'};sent=[];p.send=sent.append;lines=iter([(0.,'T 42'),(0.,'DONE 1 2 0 0 stop')]);p.q=SimpleNamespace(get=lambda **kw:next(lines))
   raw=p.request('absent',[11,12],fresh=0);self.assertIsNone(raw['pin']);self.assertEqual(sent,[q.request_command([11,12])]);self.assertNotIn('pin=',raw['command'])
 def test_invalid_pin_fresh_or_tokens_fail_closed(self):
  for pin in (-1,2,True,'0'):
   with self.assertRaises(ValueError):q.request_command([11,12],pin=pin)
  for ids in ([],[True],[-1],[248320]):
   with self.assertRaises(ValueError):q.request_command(ids)
  with self.assertRaises(ValueError):q.request_command([11],fresh=2)
 def test_root_turn_emit_absent_pin_and_pin_group_positive(self):
  source=Path(q.__file__).read_text();self.assertIn("pin=plan['tokens']['pin'] if group=='pin' else None",source);self.assertIn('def req(label,key,fresh=0,pin=None,cancel=None)',source)
  ids=list(range(49));establish=q.request_command(ids,fresh=1);hit=q.request_command(ids,fresh=0);self.assertEqual(self.parsed_ceiling(establish,49),0);self.assertEqual(self.parsed_ceiling(hit,49),48)
 def test_oldV7_basic_rejected_before_model_admission(self):
  with patch.object(q,'read',return_value={'schema':7,'group':'basic','passed':True}),patch.object(q,'source35_admission',side_effect=AssertionError('No model')):
   with self.assertRaises(ValueError):q.basic_receipt_binding(Path('/tmp/CPU-old-basic/report.json'),{})
 def test_unchanged_parser_math_raw_identity_and_parent_lifecycle(self):
  here=Path(q.__file__).parent
  def functions(path):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}
  old=functions(here/'serial_prefix_qualification_v7.py');new=functions(Path(q.__file__))
  for name in ('extract_numeric','extract','le32_digest','keyed_eviction','continue_fixture','compare_vectors','compare_requests','decode_output','topology_args','observer_env_contract','source35_admission','engine_identity'):self.assertEqual(old[name],new[name],name)
  old=functions(here/'qualify_serial_prefix_v7.py');new=functions(Path(parent.__file__))
  for name in ('stat_signature','full_buffered_identity','finalizable'):self.assertEqual(old[name],new[name],name)
 def test_root21_tokens_and_gate_unchanged(self):
  source=Path(q.__file__).read_text();self.assertIn("expected=plan['tokens']['pin' if group=='pin' else 'turn_shared' if group=='turn' else 'root']",source);self.assertIn("require(hm['ledger']['actual_reused']==expected",source)
  old=(Path(q.__file__).parent/'serial_prefix_prompt_fixture_v6.py').read_text();self.assertIn("root=next(i for i in range(1,len(cases['A'])) if cases['A'][i]==turn[0])",old)


if __name__=='__main__':unittest.main()
