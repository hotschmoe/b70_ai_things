"""Fail-closed source33 admission and tiny observation fixtures; no weights/GPU."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import explore_full48_original_prefix1_v4 as explorer
import validate_prefix_residual30_final_v4 as admission


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='ascii')


class AdmissionFailures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = {'driver_sha256': admission.DRIVER_SHA}
        put(self.root / 'child/plan.snapshot.json', self.plan)
        put(self.root / 'input-plan.snapshot.json', self.plan)
        binding = admission.sha(self.root / 'child/plan.snapshot.json')
        self.parent = {'controller_sha256': admission.DRIVER_SHA, 'wrapper_sha256': admission.PARENT_SHA, 'plan_sha256': binding, 'passed': False, 'errors': ['synthetic CPU negative only']}
        self.child = {'driver_sha256': admission.DRIVER_SHA, 'plan_sha256': binding}
        put(self.root / 'parent-qualification.json', self.parent)
        put(self.root / 'child/report.json', self.child)
        put(self.root / 'post-health.json', {})

    def tearDown(self):
        self.temp.cleanup()

    def test_failed_parent_rejected_before_model_admission(self):
        with patch.object(admission.ctrl, 'manifest_binding') as gate:
            with self.assertRaisesRegex(ValueError, 'parent lifecycle'):
                admission.finalized_binding(self.root)
            gate.assert_not_called()

    def test_wrong_source_driver_rejected(self):
        self.parent['controller_sha256'] = '0' * 64
        put(self.root / 'parent-qualification.json', self.parent)
        with self.assertRaisesRegex(ValueError, 'driver differs'):
            admission.finalized_binding(self.root)

    def test_old_parent_wrapper_rejected(self):
        self.parent['wrapper_sha256'] = '0' * 64
        put(self.root / 'parent-qualification.json', self.parent)
        with self.assertRaisesRegex(ValueError, 'parent differs'):
            admission.finalized_binding(self.root)

    def test_changed_plan_rejected(self):
        put(self.root / 'input-plan.snapshot.json', {'driver_sha256': admission.DRIVER_SHA, 'foreign': True})
        with self.assertRaisesRegex(ValueError, 'plan/report binding'):
            admission.finalized_binding(self.root)

    def test_admission_is_read_only_and_recollects_public_components(self):
        source = Path(admission.__file__).read_text()
        tree = ast.parse(source)
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for name in ('ctrl.manifest_binding', 'ctrl.base.extract', 'ctrl.strengthen', 'ctrl.collect', 'ctrl.cross_sfd', 'ctrl.validate_original_input33', 'audit_text'):
            self.assertIn(name, calls)
        for name in calls:
            self.assertFalse(name.endswith(('.write_text', '.write_bytes', '.finalize')))
        self.assertNotIn('reference21', source)

    def test_current_page_timestamp_is_not_source_identity(self):
        recorded = {'sha256': 'a' * 64, 'current_known_pages': {'epoch': 1, 'rows': [{'sha256': 'b' * 64}], 'stat_after': [1, 2, 3, 4, 5]}}
        current = json.loads(json.dumps(recorded));current['current_known_pages']['epoch'] = 2
        self.assertTrue(admission.same_current_identity(current, recorded))
        current['current_known_pages']['rows'][0]['sha256'] = 'c' * 64
        self.assertFalse(admission.same_current_identity(current, recorded))

    def test_admission_requires_actual_finalwindow34_marker(self):
        text = Path(admission.__file__).read_text()
        self.assertIn("chain['generation'].get('final_window_PLE_observer_fix34') is True", text)
        self.assertEqual(admission.ctrl.PLAN_SOURCE_SHA, '82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7')

    def test_new_and_owned_frozen_dependency_closure(self):
        binding = explorer.dependency_binding()
        self.assertIn('strata/flash-next/current-ple-prompt35-engine-build-plan-v1.json', binding)
        self.assertIn('strata/flash-next/validate_prefix_residual30_final_v4.py', binding)


class ObservationFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / 'child/p30_on'
        self.fields = []
        self.values = {}
        for layer in range(48):
            for phase in ('input', 'attention', 'ffn'):
                value = np.full((4, 2560), layer + {'input': 0, 'attention': .25, 'ffn': .5}[phase], dtype='<f4')
                path = self.directory / 'p30' / ('l%d-%s.f32' % (layer, phase))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value.tobytes())
                self.fields.append({'layer': layer, 'phase': phase, 'path': str(path), 'sha256': explorer.sha(path), 'bytes': 40960})
                self.values['layer%d_%s' % (layer, phase)] = value
        headpath = self.directory / 'captures/head.f32'
        headpath.parent.mkdir()
        headpath.write_bytes(np.zeros(248320, dtype='<f4').tobytes())
        self.head = {'pid': 9, 'request': 1, 'path': str(headpath), 'sha256': explorer.sha(headpath)}
        put(self.directory / 'requests.json', [{'prefix': n, 'raw': {'ids': [19] * n}, 'meta': {'logits': [self.head], 'coverage': {'required_stage_ranges': {'0': [0, 48]}}}} for n in (1, 2, 4, 8)])
        self.capture = {'frames': [{'binding': {'schema':2,'nonce':explorer.window_nonce(9,1,0,1,'verifier'),'request': 1, 'pid': 9, 'gen_ids': [19], 'route': 'verifier', 'rows': 1, 'first_position': 0}, 'fields': self.fields}]}
        self.plan = {'prefixes': {'1': [19]}}
        # Mock only admission boundary. These are observation matrices, never
        # positive GPU receipts or inputs to original-state computation.
        self.mock = patch.object(admission, 'finalized_binding', return_value=({}, {}, self.plan, {'capture': self.capture}))
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.temp.cleanup()

    def test_all144_phase_shapes_and_actual_head(self):
        ids, values, binding = explorer.native_prefix1(self.root, self.plan)
        self.assertEqual(ids, [19])
        self.assertEqual(len(values), 145)
        self.assertEqual(values['layer1_input'].shape, (4, 2560))
        self.assertEqual(values['head'].shape, (248320,))
        self.assertTrue(binding['all144_P30_phases'])
        self.assertFalse(binding['native_inputs_used_for_own_computation'])

    def test_raw_mutation_rejected(self):
        path = Path(self.fields[0]['path'])
        path.write_bytes(b'\x00' * 40960)
        # l0/input is all zeros; mutate a different byte explicitly.
        raw = bytearray(path.read_bytes());raw[0] = 1;path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'raw path/SHA/extent'):
            explorer.native_prefix1(self.root, self.plan)

    def test_missing_phase_rejected(self):
        self.fields.pop()
        with self.assertRaisesRegex(ValueError, 'All144'):
            explorer.native_prefix1(self.root, self.plan)

    def test_duplicate_phase_rejected(self):
        self.fields.append(dict(self.fields[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate P30'):
            explorer.native_prefix1(self.root, self.plan)

    def test_schema1_and_stale_final_nonce_rejected(self):
        binding=self.capture['frames'][0]['binding']
        for key,value in [('schema',1),('nonce',explorer.window_nonce(9,1,0,1,'prompt_verifier'))]:
            old=binding[key];binding[key]=value
            with self.assertRaisesRegex(ValueError,'schema2 final verifier window nonce'):
                explorer.native_prefix1(self.root,self.plan)
            binding[key]=old

    def test_wrong_row_or_pid_rejected(self):
        self.capture['frames'][0]['binding']['pid'] = 10
        with self.assertRaises(ValueError):
            explorer.native_prefix1(self.root, self.plan)

    def test_145_comparisons_are_exploratory(self):
        _, native, _ = explorer.native_prefix1(self.root, self.plan)
        owned = {'ids': [19], 'captured_inputs_used': False, 'captured_states_or_selected_ids_used': False, 'full_model_math_qualified': False, 'trace': [{'route': 'verifier', 'layers': [{'layer': l, **{phase: self.values['layer%d_%s' % (l, phase)] for phase in ('input', 'attention', 'ffn')}} for l in range(48)]}], 'first_generated_logits': np.zeros(248320), 'gdn_states_owned': {}, 'qsa_states_owned': {}, 'ple_history_owned': np.zeros((9, 10240)), 'last_two_owned': [-1, 19]}
        output = self.root / 'own';output.mkdir()
        result = explorer.save_owned_and_compare(output, owned, native)
        self.assertEqual(len(result['comparisons']), 145)
        self.assertFalse(result['numeric_pass_claim'])
        self.assertIsNone(result['tolerance_gate'])
        self.assertFalse(result['full_model_math_qualified'])

    def test_original_composition_receives_only_ids(self):
        tree = ast.parse(Path(explorer.__file__).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'composition.tokens']
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[0]), 'ids')
        self.assertEqual(calls[0].keywords, [])


if __name__ == '__main__':
    unittest.main()
