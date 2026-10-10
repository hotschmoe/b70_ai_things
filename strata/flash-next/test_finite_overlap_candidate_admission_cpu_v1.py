import copy
import unittest
from finite_overlap_candidate_admission_v1 import selected_inputs


class Controls(unittest.TestCase):
    def setUp(self):
        self.fixture = {'fixtures': {
            'warm': [self.row(i) for i in range(2)],
            'target': [self.row(i + 2) for i in range(6)]}}
        self.proof = {'fixture_sha256': 'a' * 64}
        self.binding = {'continuation_screen_passed': True,
                        'complete_measurement_rows': 16,
                        'fixture_binding': self.proof,
                        'selected_candidate': 'water-cycle'}
        self.candidates = [{'id': name, 'target_indices': [2 * i, 2 * i + 1]}
                           for i, name in enumerate(('water-cycle', 'library-returns', 'trail-signs'))]

    @staticmethod
    def row(i):
        return {'messages': [{'role': 'user', 'content': str(i)}],
                'rendered': str(i), 'ids': [i]}

    def test_returns_only_selected_authentic_inputs_without_mutating_roster(self):
        result = selected_inputs(self.binding, self.fixture, self.proof, self.candidates)
        self.assertEqual([r['ids'] for r in result['target']], [[2], [3]])
        result['target'][0]['ids'][0] = 999
        self.assertEqual(self.fixture['fixtures']['target'][0]['ids'], [2])
        self.assertFalse(result['CPU_outputs_or_state_imported'])
        self.assertFalse(result['actual_GPU_positive_overlap_observed'])

    def test_incomplete_or_foreign_screen_and_nonregistered_selection_refused(self):
        for field, value in (('complete_measurement_rows', 15),
                             ('complete_measurement_rows', 16.0),
                             ('continuation_screen_passed', False),
                             ('fixture_binding', {'fixture_sha256': 'b' * 64}),
                             ('selected_candidate', 'invented')):
            binding = copy.deepcopy(self.binding); binding[field] = value
            with self.assertRaises(ValueError):
                selected_inputs(binding, self.fixture, self.proof, self.candidates)

    def test_changed_candidate_indices_and_bool_token_refused(self):
        candidates = copy.deepcopy(self.candidates); candidates[0]['target_indices'] = [0, 2]
        with self.assertRaises(ValueError):
            selected_inputs(self.binding, self.fixture, self.proof, candidates)
        self.fixture['fixtures']['target'][0]['ids'] = [True]
        with self.assertRaises(ValueError):
            selected_inputs(self.binding, self.fixture, self.proof, self.candidates)


if __name__ == '__main__':
    unittest.main()
