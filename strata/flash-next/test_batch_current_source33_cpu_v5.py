"""Source33/C111 generation port controls; no runtime, weights, Docker or GPU."""
import ast
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import batch_numerical_proofs_v5 as proof
import batch_numerical_execution_v5 as execution
import qualify_batch_numerical_v5 as parent
import run_batch_api_controls_v5 as api
HERE = Path(__file__).resolve().parent


def function(file, name, normalize=False):
    text = file.read_text()
    if normalize:
        text = text.replace('source32', 'source33')
    return ast.dump(next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name), include_attributes=False)


class Source33Port(unittest.TestCase):
    def test_exact_source33_recipe_and_C111_providers(self):
        spec = proof.lane_contract('source33')
        self.assertEqual((spec['source_files'], spec['headers'], spec['patches'], spec['generation']), (63, 27, 33, 11))
        c1, final = proof.providers('source33')
        self.assertEqual(c1.__name__, 'c1_serve_controller_combined_v11')
        self.assertEqual(final.__name__, 'qualify_c1_serving_combined_v11')
        plan = parent.source_plan_binding('source33')
        self.assertEqual(len(plan['expected_patched_source_sha256']), 63)
        self.assertEqual(len(plan['added_header_payloads']), 27)
        self.assertEqual(len(plan['patches']), 33)
        self.assertEqual(len(plan['build_targets']), 8)
        self.assertEqual(len(plan['runtime_python_sources']), 6)

    def test_all_older_lanes_rejected(self):
        for lane in ('source29', 'source31', 'source32', 'old28'):
            with self.subTest(lane=lane), self.assertRaises(ValueError):
                proof.lane_contract(lane)

    def test_foreign_SDK_receipts_rejected_before_payload(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for digest in ('94aef305dbc6a31f023afa55573d52a9ce6af9443cfc0182eb7c324255159094', '01884ae392f4eb9dd1379e08de2b05ab4836ea350bc4e349bf01cedc5425fdc5'):
                (root / 'receipt.json').write_text(json.dumps({'build_rc': 0, 'external_source_unchanged': True, 'plan_snapshot_unchanged': True, 'plan_sha256': digest}))
                with self.assertRaisesRegex(ValueError, 'matching whole SDK'):
                    proof.engine_binding(root, 'source33')

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

    def test_baseline_off_guard_precedes_C111_model_access(self):
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
            self.assertEqual(function(HERE / 'qualify_batch_numerical_v4.py', name), function(HERE / 'qualify_batch_numerical_v5.py', name))

    def test_missing_compiled_source33_marker_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'prepared.json').write_text(json.dumps({'engine_receipt': str(root / 'receipt.json')}))
            (root / 'server-config.json').write_text(json.dumps({'env': {}}))
            c1 = Mock();c1.validate_prepared.return_value = {'combined_generation': {'semantic_current_PLE_gather_fix32': True}}
            final = Mock()
            with patch.object(proof, 'engine_binding'), patch.object(proof, 'providers', return_value=(c1, final)):
                with self.assertRaisesRegex(ValueError, 'source33 compiled host observer'):
                    proof.genuine_baseline(root)

    def test_numerical_private_state_cancel_migration_guards_unchanged(self):
        names = ('exact_producer_counters', 'native_row_event', 'terminal_api_join', 'passing_suite', 'off_on_histories', 'owner_proofs', 'raw_case', 'confined_file', 'artifact_bindings', 'validate_artifacts')
        for name in names:
            self.assertEqual(function(HERE / 'batch_numerical_proofs_v4.py', name, True), function(HERE / 'batch_numerical_proofs_v5.py', name), name)

    def test_final_raw_native_API_serial_recollection_unchanged(self):
        for name in ('native_histories', 'api_histories', 'vectors', 'serial_vectors'):
            self.assertEqual(function(HERE / 'audit_batch_numerical_suite_v4.py', name), function(HERE / 'audit_batch_numerical_suite_v5.py', name))

    def test_counter_negative_controls_remain_exact(self):
        policy = {(1, 'admission'): {'reused': 0, 'read_from': 0, 'reread_to': -1}}
        trace = 'SBF resume rid=1 slot=0 reused=0 read_from=0 reread_to=-1'
        self.assertTrue(proof.exact_producer_counters(trace, {1}, policy))
        for key, old, new in (('reused', '0', '1'), ('read_from', '0', '1'), ('reread_to', '-1', '0')):
            with self.assertRaises(ValueError):
                proof.exact_producer_counters(trace.replace(key + '=' + old, key + '=' + new), {1}, policy)

    def test_source33_cases_preserve_original_corpus_and_policy(self):
        plan = json.loads((HERE / 'current-ple-input33-engine-build-plan-v1.json').read_text())
        for count in (2, 4, 6):
            old = json.loads((HERE / ('batch-numerical-case%d-source32-v1.json' % count)).read_text())
            new = json.loads((HERE / ('batch-numerical-case%d-source33-v1.json' % count)).read_text())
            self.assertEqual(new['source_lane'], 'source33')
            for name in ('tokens', 'messages', 'api_token_ids', 'tokenizer_sha256', 'actual_counter_policy', 'max_new_by_request', 'cancel_index'):
                self.assertEqual(old[name], new[name])
            self.assertTrue(new['source32_semantic_current_PLE_required'])
            self.assertTrue(new['source33_host_observer_compiled_OFF_required'])
            for name, digest in new['source_sha256'].items():
                self.assertEqual(digest, plan['expected_patched_source_sha256'][name])

    def test_registry_alias_closure_preserves_primary_name(self):
        text = (HERE / 'batch-numerical-model-registry-proposal-v5.yaml').read_text()
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
        for name in ('run_batch_numerical_pilot_v5.py', 'run_batch_api_controls_v5.py', 'run_batch_serial_controls_v5.py', 'audit_batch_numerical_suite_v5.py'):
            tree = ast.parse((HERE / name).read_text())
            self.assertTrue(any(isinstance(node, ast.Call) and ast.unparse(node.func) == 'source_observers_off' for node in ast.walk(tree)), name)
        for name in ('batch_numerical_execution_v5.py', 'run_batch_api_controls_v5.py'):
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
        plan = json.loads((HERE / 'batch-numerical-harness-source-plan-v5.json').read_text())
        for name, digest in {**plan['source_sha256'], **plan['immutable_dependency_sha256']}.items():
            self.assertEqual(proof.sha(HERE.parents[1] / name), digest, name)


if __name__ == '__main__':
    unittest.main()
