import ast,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_source59_mode0_fidelity_v1 as a
import validate_source59_mode0_fidelity_v1 as reader
import source59_mode0_fidelity_raw_v1 as raw
import current_mode0_fidelity_capture_recipe_v2 as recipe
from serial37_canonical_json_v3 import canonical
F17=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
class Current59(unittest.TestCase):
 def test_genuine_recipe_and_real_callback(self):
  with tempfile.TemporaryDirectory()as temp:
   old_ns,old_phases=a.parent.NS,a.parent.PHASES
   try:
    a.parent.NS=a.NS;a.parent.PHASES=a.PHASES
    settings={'generation':59,'prefill':512,'cards':2,'esimd':0,'mirror':0,'experiment':'copy','corpus':str(F17/'natural-p2048-tokenizer-v1/corpus.json'),'corpus_sha256':'61b5caabf76bfedfda334973741366a8341e9f600cf00b42ac042f6700afdbef','profile_a':None,'profile_b':None}
    a.parent.configure(settings);a.parent.set_sdk(a.SDK,a.SDK_SHA);a.NS['set_case']('prefill_about512');a.NS['set_runtime'](F17/'pti-paused-regular-loaded-fixture-v2/report.json','ead4530510357315c8d2d6874378c92a5f3522f345c333c05b6173ffd6ad8dc9')
    plan=a.c.read(F17/'source43-normal-cli-v1-prepared-v1/plan.json')
    # Metadata/source construction only. Never current payload admission or Docker.
    with patch.object(a,'closure',return_value='a'*64):proof=a.preflight(plan,Path(temp))
    command=proof['commands'][0];r=command['recipe'];self.assertEqual(r['requests'],[{'ordinal':i,'ids':[248045],'fresh':1,'pin':None,'max_new':1}for i in range(1,5)])
    self.assertEqual(r['engine_generation'],59);self.assertEqual(r['materialized_mode'],0);self.assertEqual(r['native_env']['SYCL_UR_TRACE'],'2');self.assertIn(str(a.SDK/'build')+':/build:ro',command['native_command']);self.assertNotIn('unitrace',command['native_command'][-1])
    self.assertIs(a.NS['life'].contract.native_command,a.native_command)
    self.assertEqual(len(raw.field_contract()),33)
    self.assertEqual(sum(x['bytes']for x in raw.field_contract()),7023048)
    p=dict(r['native_env']);p['STRATA_QSA_INDEXER_MATERIALIZED_HALF']='1'
    with self.assertRaises(ValueError):recipe.recipe(r['native_argv'],p,'a'*64,59)
   finally:a.parent.NS=old_ns;a.parent.PHASES=old_phases
 def test_exact_protocol_request_and_foreign_negative(self):
  want=[{'ordinal':i,'ids':[248045],'fresh':1,'pin':None,'max_new':1}for i in range(1,5)];rows=[]
  for i in range(1,5):
   data={'ids':[248045],'fresh':1,'pin':None,'cancel_requested':None,'stop_sent':False,'command':'GEN 1 fresh=1 ckpt=1 temperature=0 248045','stdout':['T 846','DONE 1 1 1.0 0.0 length'],'output_ids':[846],'stdout_seconds':[.1,.2],'LP':[]}
   rows.append({'ordinal':i,'raw':data})
  views=a.requests(rows,want);self.assertTrue(a.equivalence(views)['current59_four_fresh_prefix1_histories_equal'])
  rows[2]['raw']['command']='GEN 1 fresh=1 ckpt=1 temperature=0 0'
  with self.assertRaises(ValueError):a.requests(rows,want)
 def test_actual_namespace_and_saved_public_failure(self):
  from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
  a.NS['_PORTS']=a.parent.base.boundary_ports(a.NS)
  active=SimpleNamespace(**{**vars(a),**a.NS});ports=boundary_reader_ports(active)
  self.assertIs(ports['a'],active);self.assertIs(a.NS['run_phase'],a.run_phase);self.assertEqual(active.PHASES,('source59-mode0-fidelity',))
  source=Path(a.__file__).read_text();self.assertIn('pipe_bytes=128<<20,pipe_lines=1000000',source);self.assertIn("'ARM').write_text",source)
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);a.c.write(root/'report.json',{'passed':False,'errors':['actual original failure'],'artifacts':{},'SDK_root':str(a.SDK),'SDK_receipt_sha256':a.SDK_SHA})
   with patch.object(reader,'source_binding',return_value='b'*64):
    with self.assertRaises(ValueError):reader.finalized_binding(root,a.c.sha(root/'report.json'))
    with self.assertRaises(ValueError):reader.finalized_binding(root,'0'*64)
 def test_no_original_source44_runtime_admission(self):
  source=Path(raw.__file__).read_text()
  self.assertNotIn('source44_sdk_binding',source);self.assertNotIn('sdk.binding()',source)
  self.assertIn("'numerical_source_generation' not in f",source)
  self.assertIn("'extra44_core_RMS_fields_observed':False",source)
  self.assertIn("expected_sfd",source);self.assertIn("len(joins)==len(set(joins))==192",source)
  self.assertIn("type(f[k])is int",source)
