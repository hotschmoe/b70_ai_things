"""Source-only finite prompt policy; no tokenizer or model execution."""
import copy
import json
import unittest
from unittest.mock import patch
import api_positive_overlap_finite_corpus_v2 as corpus


class Controls(unittest.TestCase):
    def test_complete_preregistered_roster_no_actual_counts(self):
        result = corpus.binding()
        self.assertEqual(result['screen_cases_required'], 16)
        self.assertEqual(result['natural_completion_tokens_required'], [8, 63])
        self.assertIsNone(result['selected_candidate'])
        self.assertFalse(result['actual_token_counts_observed'])

    def test_original_contexts_preserved_questions_are_short_useful(self):
        old = json.loads((corpus.HERE / 'api-positive-overlap-corpus-seed-v1.json').read_bytes())
        new = json.loads(corpus.SEED.read_bytes())
        for before, after in zip(old['messages']['target'], new['messages']['target']):
            self.assertEqual(before[1]['content'].split('\n\n')[0], after[1]['content'].split('\n\n')[0])
            self.assertIn('?', after[1]['content'])
            self.assertNotIn('Write three complete sentences', after[1]['content'])
        self.assertTrue(all('one sentence of 12-20 words' in row[1]['content'] for row in new['messages']['warm']))

    def test_no_future_runtime_or_length_waiver(self):
        original = json.loads(corpus.SEED.read_bytes())
        for field, value in (('actual_CPU_continuation_observed', True),
                             ('prior_token_counts_or_responses_transferred', True),
                             ('forced_no_EOS_or_length_extension', True), ('warm_max_new', 80)):
            bad = copy.deepcopy(original)
            bad[field] = value
            with patch.object(corpus.json, 'loads', return_value=bad):
                self.assertRaises(ValueError, corpus.binding)


if __name__ == '__main__':
    unittest.main()
