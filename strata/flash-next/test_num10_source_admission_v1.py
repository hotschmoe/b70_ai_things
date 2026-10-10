"""Tiny source/admission controls; no native proof or payload execution."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import layer0_numerical_qualification_v10 as q
import validate_layer0_numerical_final_v10 as v

class AdmissionTests(unittest.TestCase):
    def test_generation_refuses_old_before_source_reads(self):
        with patch.object(q.c1, 'combined_generation_gate') as gate:
            with self.assertRaises(ValueError): q.engine_binding(Path('/absent'), 'old29')
            gate.assert_not_called()
    def test_baseline_presence_and_observers(self):
        q.numerical_baseline_gate({})
        for key in q.NUMERICAL_BASELINE_OFF:
            with self.assertRaises(ValueError): q.numerical_baseline_gate({key:'1'})
        for key in ('STRATA_VERIFY_EAGER','STRATA_CKPT_REREAD','STRATA_STATE_HASH'):
            with self.assertRaises(ValueError): q.numerical_baseline_gate({key:'0'})
    def test_public_source_proof_before_model_or_provider(self):
        with patch.object(q,'engine_binding'), patch.object(q,'read',return_value={'combined_generation':{}}), patch.object(q.c113,'source_pins'), patch.object(q.c1,'metadata_admission_gate'), patch.object(q,'validate_final_source_proof',side_effect=ValueError('missing current C113')), patch.object(q.c1,'validate_prepared') as prepared:
            with self.assertRaisesRegex(ValueError,'current C113'): q.candidate_binding({'engine_root':'/tiny','prepared':'/tiny'})
            prepared.assert_not_called()
    def test_final_reader_is_readonly(self):
        tree=ast.parse(Path(v.__file__).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('finalize','write','write_text','write_bytes','run','Popen') for n in calls))
        self.assertEqual(q.sha(Path(q.__file__)),v.DRIVER_SHA)
        self.assertEqual(q.sha(Path(v.parent_ctrl.__file__)),v.PARENT_SHA)
    def test_actual_fields_physical_layout(self):
        fields={f['name']:f for f in q.field_contract()}
        for name in ('gdn_state_before','gdn_state_after'):
            self.assertEqual(fields[name]['bytes'],128*48*128*4)
            self.assertEqual(fields[name]['provenance'],'actual_buffer')
        for name in ('gdn_conv_before','gdn_conv_after','gdn_qkv','gdn_decay_beta','gdn_normalized_qkv'):
            self.assertIn(name,fields)
    def test_raw_mutation_recollection_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);plan={'args':[],'prefixes':{str(n):[7]*n for n in q.PREFIXES}}
            result={'passed':True,'error':None,'removed':True,'engine_rc':0,'state':{'ExitCode':0,'Running':False},'complete_four_prefix_roster':True}
            all_rows=[]
            for name in ('candidatecombined_off','candidatecombined_on'):
                directory=root/name;directory.mkdir();log=directory/'engine.combined.log';log.write_text('tiny producer')
                result['canonical_combined_log_sha256']=q.sha(log);(directory/'result.json').write_text(json.dumps(result))
                rows=[{'prefix':n,'raw':{'ids':[7]*n,'fresh':1,'stderr':[]},'meta':{},'external':{},'layer0':None if name.endswith('off') else {'actual':'saved'}} for n in q.PREFIXES]
                (directory/'requests.json').write_text(json.dumps(rows));all_rows.append(rows)
            child={'results':[copy.deepcopy(result),copy.deepcopy(result)],'plan_sha256':'tiny'}
            with patch.object(q.base,'extract',return_value={}), patch.object(q,'strengthen',return_value={}), patch.object(q,'expected_stage_ranges',return_value=[]), patch.object(q,'collect_frame',return_value={'actual':'mutated'}):
                with self.assertRaisesRegex(ValueError,'raw33 fields'): v.recollect(root,child,plan,{}, {})

    def test_raw_mutation_and_parent_mismatch_failclosed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'child').mkdir()
            plan={'driver_sha256':v.DRIVER_SHA}; child={'driver_sha256':v.DRIVER_SHA,'plan_sha256':'wrong'}
            parent={'controller_sha256':v.DRIVER_SHA,'wrapper_sha256':v.PARENT_SHA,'plan_sha256':'wrong'}
            for name,data in [('parent-qualification.json',parent),('child/report.json',child),('child/plan.snapshot.json',plan),('input-plan.snapshot.json',plan)]: (root/name).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'parent/child/plan'): v.finalized_binding(root)
            raw=root/'target.bin';raw.write_bytes(b'abcd');first=q.sha(raw);raw.write_bytes(b'abce');self.assertNotEqual(first,q.sha(raw))
            child['driver_sha256']='oldNUM9';(root/'child/report.json').write_text(json.dumps(child))
            with self.assertRaisesRegex(ValueError,'driver binding'): v.finalized_binding(root)

if __name__=='__main__': unittest.main()
