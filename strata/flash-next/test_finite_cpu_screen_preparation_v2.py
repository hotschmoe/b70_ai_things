"""Tiny metadata-only new-fixture admission; no original weights or runtime."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import finite_cpu_screen_preparation_v2 as prep


class Controls(unittest.TestCase):
    def fixture(self):
        seed = prep.export.read(prep.export.SEED)
        fixture = {'fixtures': {}}
        for phase in ('warm', 'target'):
            fixture['fixtures'][phase] = [dict(logical_index=i, messages=messages, rendered='tiny synthetic metadata', ids=[20] * 100)
                                          for i, messages in enumerate(seed['messages'][phase])]
        recipe = {'schema': 'api-positive-overlap-finite-screen-source-v2',
                  'actual_model_inference': False, 'source_bindings': {},
                  'fixture_root': None, 'preregistered_screen_plan': None,
                  'corpus_messages': seed['messages']['warm'] + seed['messages']['target'],
                  'memory_cap_bytes': 116 << 30, 'minimum_start_available_bytes': 112 << 30,
                  'minimum_available_bytes': 6 << 30, 'runner_sha256': 'CPU-test-only'}
        return fixture, recipe

    def test_exact_new_prepared_plan_no_undeclared_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, recipe = self.fixture()
            binding = {'actual_new_fixture_generation': 'CPU-test-only'}
            request_path = root / 'prereg.json'
            recipe_path = root / 'source.json'
            recipe_path.write_text(json.dumps(recipe))
            with patch.object(prep.requests.producer, 'source_binding', return_value={'SOURCE_CPU_TEST_ONLY': True}), patch.object(prep, 'RECIPE', recipe_path):
                request_path.write_text(json.dumps(prep.requests.screen_plan(fixture, binding, prep.export.read(prep.export.SEED))))
                plan = prep.expected_plan(recipe, root / 'fixture', fixture, binding, request_path)
                path = root / 'plan.json'
                path.write_text(json.dumps(plan))
                with patch.object(prep.export, 'finalized_binding', return_value=(fixture, binding)):
                    self.assertEqual(prep.admit_prepared(path), plan)
                    for key, value in (('memory_cap_bytes', 1), ('minimum_available_bytes', 0),
                                       ('runner_sha256', 'foreign'), ('actual_model_inference', True)):
                        bad = copy.deepcopy(plan)
                        bad[key] = value
                        path.write_text(json.dumps(bad))
                        self.assertRaises(ValueError, prep.admit_prepared, path)

    def test_declared_all8_requests_and_seed_no_old_tokens(self):
        fixture, _ = self.fixture()
        with patch.object(prep.requests.producer, 'source_binding', return_value={}):
            plan = prep.requests.screen_plan(fixture, {}, prep.export.read(prep.export.SEED))
        self.assertEqual(len(plan['requests']), 8)
        self.assertTrue(all(r['request']['seed'] == 1234 and r['request']['n_predict'] == 64
                            and r['request']['temperature'] == 0 and r['fresh_process_repeats'] == 2
                            for r in plan['requests']))
        self.assertIsNone(plan['selected_candidate'])
        self.assertFalse(plan['actual_CPU_screen_observed'])

    def test_runner_new_plan_and_old_safety_helpers_exact(self):
        import ast
        import qualify_api_positive_overlap_cpu_screen_v2 as runner
        old = Path(runner.__file__).with_name('qualify_api_positive_overlap_cpu_screen_v1.py')
        old_tree = ast.parse(old.read_text())
        new_tree = ast.parse(Path(runner.__file__).read_text())
        old_helpers = {n.name: n for n in old_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new_helpers = {n.name: n for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in old_helpers.keys() - {'main', 'finalized_binding'}:
            self.assertEqual(ast.dump(old_helpers[name], include_attributes=False),
                             ast.dump(new_helpers[name], include_attributes=False), name)
        source = Path(runner.__file__).read_text()
        self.assertIn('admit_prepared(a.screen_plan)', source)
        self.assertIn('admit_prepared(report[\'prepared_plan_path\'])', source)
