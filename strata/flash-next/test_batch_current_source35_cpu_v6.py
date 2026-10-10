"""Source35/C113 generation port controls; no runtime, weights, Docker or GPU."""
import ast
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import batch_numerical_proofs_v6 as proof
import batch_numerical_execution_v6 as execution
import qualify_batch_numerical_v6 as parent
import run_batch_api_controls_v6 as api
HERE = Path(__file__).resolve().parent


def function(file, name, normalize=False):
    text = file.read_text()
    if normalize:
        text = text.replace('source33', 'source35').replace('current_PLE_host_observer_source35','current_PLE_host_observer_source33').replace('source35_input_observation_runtime_qualified','source33_input_observation_runtime_qualified')
    return ast.dump(next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name), include_attributes=False)


class Source35Port(unittest.TestCase):
    def test_exact_source35_recipe_and_C113_providers(self):
        spec = proof.lane_contract('source35')
        self.assertEqual((spec['source_files'], spec['headers'], spec['patches'], spec['generation']), (63, 27, 35, 13))
        c1, final = proof.providers('source35')
        self.assertEqual(c1.__name__, 'c1_serve_controller_combined_v13')
        self.assertEqual(final.__name__, 'qualify_c1_serving_combined_v13')
        plan = parent.source_plan_binding('source35')
        self.assertEqual(len(plan['expected_patched_source_sha256']), 63)
        self.assertEqual(len(plan['added_header_payloads']), 27)
        self.assertEqual(len(plan['patches']), 35)
        self.assertEqual(len(plan['build_targets']), 8)
        self.assertEqual(len(plan['runtime_python_sources']), 6)

    def test_all_older_lanes_rejected(self):
        for lane in ('source29', 'source31', 'source32', 'source33', 'source34', 'old28'):
            with self.subTest(lane=lane), self.assertRaises(ValueError):
                proof.lane_contract(lane)

    def test_foreign_SDK_receipts_rejected_before_payload(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for digest in ('81798dff3fd012f7b95984403ea1088a1f157d37ee2d2e6e1b612711052cd34c','89ba4019dd5b9b05ed6cc2f61fcf776e35fcd97824743582aac1a42f1d419338','94aef305dbc6a31f023afa55573d52a9ce6af9443cfc0182eb7c324255159094', '01884ae392f4eb9dd1379e08de2b05ab4836ea350bc4e349bf01cedc5425fdc5'):
                (root / 'receipt.json').write_text(json.dumps({'build_rc': 0, 'external_source_unchanged': True, 'plan_snapshot_unchanged': True, 'plan_sha256': digest}))
                with self.assertRaisesRegex(ValueError, 'matching whole SDK'):
                    proof.engine_binding(root, 'source35')

    def test_input33_P30_OFF_contract(self):
        self.assertTrue(proof.source_observers_off({}))
        self.assertTrue(proof.source_observers_off({'STRATA_PLE_INPUT33': '0', 'STRATA_PREFIX30': '0'}))
        for flag in ('STRATA_PLE_INPUT33', 'STRATA_PREFIX30'):
            for value in ('1', '', 'true', 0):
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    proof.source_observers_off({flag: value})

    def test_plan_off_guard_precedes_registry_model_access(self):
        with self.assertRaisesRegex(ValueError, 'STRATA_PLE_INPUT33 OFF'):
            execution.manifest_binding({'env': {'STRATA_PLE_INPUT33': '1'}})

    def test_EAGER_presence_zero_also_rejected(self):
        for value in ('0', '1', ''):
            with self.assertRaisesRegex(ValueError, 'EAGER absent'):
                proof.source_observers_off({'STRATA_VERIFY_EAGER': value})

    def test_baseline_off_guard_precedes_C113_model_access(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'prepared.json').write_text(json.dumps({'engine_receipt': str(root / 'receipt.json')}))
            (root / 'server-config.json').write_text(json.dumps({'env': {'STRATA_PREFIX30': '1'}}))
            with patch.object(proof, 'engine_binding'), patch.object(proof, 'providers') as providers:
                with self.assertRaisesRegex(ValueError, 'STRATA_PREFIX30 OFF'):
                    proof.genuine_baseline(root)
                providers.assert_not_called()

    def test_parent_health_source4_and_teardown_AST_unchanged(self):
        for name in ('stat_signature', 'full_buffered_identity', 'finalizable'):
            self.assertEqual(function(HERE / 'qualify_batch_numerical_v5.py', name), function(HERE / 'qualify_batch_numerical_v6.py', name))

    def test_missing_compiled_source35_marker_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'prepared.json').write_text(json.dumps({'engine_receipt': str(root / 'receipt.json')}))
            (root / 'server-config.json').write_text(json.dumps({'env': {}}))
            c1 = Mock();c1.validate_prepared.return_value = {'combined_generation': {'semantic_current_PLE_gather_fix32': True}}
            final = Mock()
            with patch.object(proof, 'engine_binding'), patch.object(proof, 'providers', return_value=(c1, final)):
                with self.assertRaisesRegex(ValueError, 'source33 compiled host observer'):
                    proof.genuine_baseline(root)

    def test_each_source32_33_34_35_marker_is_required(self):
        names=('semantic_current_PLE_gather_fix32','current_PLE_host_observer_source33','final_window_PLE_observer_fix34','prompt_verifier_P30_capture35')
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'prepared.json').write_text(json.dumps({'engine_receipt':str(root/'receipt.json')}));(root/'server-config.json').write_text(json.dumps({'env':{}}))
            for marker in names:
                generation={n:True for n in names};generation.pop(marker);c1=Mock();c1.validate_prepared.return_value={'combined_generation':generation};final=Mock()
                with patch.object(proof,'engine_binding'),patch.object(proof,'providers',return_value=(c1,final)):
                    with self.assertRaises(ValueError):proof.genuine_baseline(root,'source35')
    def test_synthetic_newbaseline_binding_preserves_unqualified_claims(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'receipt.json').write_text('CPU mock SDK receipt only');prepared={'engine_receipt':str(root/'receipt.json'),'engine_receipt_sha256':proof.sha(root/'receipt.json'),'combined_generation':{n:True for n in ('semantic_current_PLE_gather_fix32','current_PLE_host_observer_source33','final_window_PLE_observer_fix34','prompt_verifier_P30_capture35')},'upload_lifecycle':{'runner_generation':2,'post_full4_source_qualified':True}};(root/'prepared.json').write_text(json.dumps(prepared));(root/'server-config.json').write_text(json.dumps({'env':{}}));c1=Mock();c1.validate_prepared.return_value=prepared;final=Mock();final.validate_final_source_proof.return_value={'CPU_mock_only':True}
            with patch.object(proof,'engine_binding'),patch.object(proof,'providers',return_value=(c1,final)):actual,binding=proof.genuine_baseline(root,'source35')
            self.assertEqual(binding['source_lane'],'source35');self.assertTrue(binding['final_window_PLE_observer_fix34']);self.assertTrue(binding['prompt_verifier_P30_capture35']);self.assertFalse(binding['source33_input_observation_runtime_qualified']);self.assertEqual(binding['C1_controller_sha256'],'f2057108c8ed7992ddbb71facdc81f78302ccf142090f4d73b3a5063c6d8c149')
    def test_actual_pinned_C113_validator_refuses_C111_C112_fixture(self):
        import test_qualify_c1_serving_combined_v13 as fixtures
        fixture=fixtures.FinalTests('test_mock_complete_current_positive');fixture.setUp();self.addCleanup(fixture.doCleanups)
        for generation in (11,12):
            fixture.final['c1_parent_generation']=generation
            with self.assertRaises(AssertionError):fixture.validate()
    def test_numerical_private_state_cancel_migration_guards_unchanged(self):
        names = ('exact_producer_counters', 'native_row_event', 'terminal_api_join', 'passing_suite', 'off_on_histories', 'owner_proofs', 'raw_case', 'confined_file', 'artifact_bindings', 'validate_artifacts')
        for name in names:
            self.assertEqual(function(HERE / 'batch_numerical_proofs_v5.py', name, True), function(HERE / 'batch_numerical_proofs_v6.py', name), name)

    def test_final_raw_native_API_serial_recollection_unchanged(self):
        for name in ('native_histories', 'api_histories', 'vectors', 'serial_vectors'):
            self.assertEqual(function(HERE / 'audit_batch_numerical_suite_v5.py', name), function(HERE / 'audit_batch_numerical_suite_v6.py', name))

    def test_counter_negative_controls_remain_exact(self):
        policy = {(1, 'admission'): {'reused': 0, 'read_from': 0, 'reread_to': -1}}
        trace = 'SBF resume rid=1 slot=0 reused=0 read_from=0 reread_to=-1'
        self.assertTrue(proof.exact_producer_counters(trace, {1}, policy))
        for key, old, new in (('reused', '0', '1'), ('read_from', '0', '1'), ('reread_to', '-1', '0')):
            with self.assertRaises(ValueError):
                proof.exact_producer_counters(trace.replace(key + '=' + old, key + '=' + new), {1}, policy)

    def test_source35_cases_preserve_original_corpus_and_policy(self):
        plan = json.loads((HERE / 'current-ple-prompt35-engine-build-plan-v1.json').read_text())
        for count in (2, 4, 6):
            old = json.loads((HERE / ('batch-numerical-case%d-source33-v1.json' % count)).read_text())
            new = json.loads((HERE / ('batch-numerical-case%d-source35-v1.json' % count)).read_text())
            self.assertEqual(new['source_lane'], 'source35')
            for name in ('tokens', 'messages', 'api_token_ids', 'tokenizer_sha256', 'actual_counter_policy', 'max_new_by_request', 'cancel_index'):
                self.assertEqual(old[name], new[name])
            self.assertTrue(new['source32_semantic_current_PLE_required'])
            self.assertTrue(new['source33_host_observer_compiled_OFF_required'])
            self.assertTrue(new['source34_final_window_PLE_observer_compiled_OFF_required'])
            self.assertTrue(new['source35_normal_prompt_verifier_capture_compiled_OFF_required'])
            for name, digest in new['source_sha256'].items():
                self.assertEqual(digest, plan['expected_patched_source_sha256'][name])

    def test_registry_alias_closure_preserves_primary_name(self):
        text = (HERE / 'batch-numerical-model-registry-proposal-v6.yaml').read_text()
        names = re.findall(r'^\s*served_model_id:\s*(\S+)\s*$', text, re.M)
        expected = []
        for cards in ('0', '0,1'):
            cfg = {'env': {'ZE_AFFINITY_MASK': cards, 'STRATA_STAGE_MIRRORS': '1', 'STRATA_STAGE_MIRROR_SEGMENT_MIB': '1024'}, 'args': ['--layer-split', '32'] if cards == '0,1' else []}
            for count in (2, 4, 6):
                for diagnostic in (False, True):
                    expected.append(api.experimental_alias(cfg, count, diagnostic))
        self.assertEqual(set(names), set(expected))
        self.assertEqual(len(names), 12)
        self.assertEqual(text.count('primary_client_id: hotschmoe-dd'), 12)

    def test_runtime_clients_guard_source_observers_OFF(self):
        for name in ('run_batch_numerical_pilot_v6.py', 'run_batch_api_controls_v6.py', 'run_batch_serial_controls_v6.py', 'audit_batch_numerical_suite_v6.py'):
            tree = ast.parse((HERE / name).read_text())
            self.assertTrue(any(isinstance(node, ast.Call) and ast.unparse(node.func) == 'source_observers_off' for node in ast.walk(tree)), name)
        for name in ('batch_numerical_execution_v6.py', 'run_batch_api_controls_v6.py'):
            text = (HERE / name).read_text()
            self.assertIn("env.update(STRATA_PLE_INPUT33='0',STRATA_PREFIX30='0')", text)
            self.assertNotIn("'STRATA_LAYER0_Q8_DIAG','STRATA_VERIFY_EAGER'", text)

    def test_actual_derived_config_has_EAGER_absent_and_source_observers_OFF(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name);prepared = root / 'prepared';prepared.mkdir();output = root / 'derived';output.mkdir()
            cfg = {'env': {'ZE_AFFINITY_MASK': '0', 'STRATA_STAGE_MIRRORS': '1', 'STRATA_STAGE_MIRROR_SEGMENT_MIB': '1024'}, 'args': ['--no-prefill-borrow']}
            (prepared / 'server-config.json').write_text(json.dumps(cfg))
            (prepared / 'artifact-identity.json').write_text(json.dumps({'runtime': {}}))
            derived, manifest = api.derived_config(prepared, output, 2, 18522, True)
            self.assertNotIn('STRATA_VERIFY_EAGER', derived['env'])
            self.assertEqual(derived['env']['STRATA_PLE_INPUT33'], '0')
            self.assertEqual(derived['env']['STRATA_PREFIX30'], '0')
            self.assertEqual(derived['model_name'], 'hotschmoe-dd')
            self.assertEqual(manifest['runtime']['env']['STRATA_PLE_INPUT33'], '0')

    def test_immutable_new_and_shared_source_closure(self):
        plan = json.loads((HERE / 'batch-numerical-harness-source-plan-v6.json').read_text())
        for name, digest in {**plan['source_sha256'], **plan['immutable_dependency_sha256']}.items():
            self.assertEqual(proof.sha(HERE.parents[1] / name), digest, name)


if __name__ == '__main__':
    unittest.main()
