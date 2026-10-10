"""Preregistered semantic corpus only; actual tokenizer/screen are future gates."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SEED = HERE / 'api-positive-overlap-corpus-seed-v2.json'


def binding():
    seed = json.loads(SEED.read_bytes())
    if seed['schema'] != 'api-positive-overlap-corpus-seed-v2':
        raise ValueError('Exact new finite corpus required')
    if seed['requested_sentence_word_range'] != [12, 20] or seed['warm_max_new'] != 64 or seed['target_max_new'] != 64:
        raise ValueError('Preregistered request/cap changed')
    if not seed['natural_completion_required'] or seed['forced_no_EOS_or_length_extension']:
        raise ValueError('Natural completion cannot be forced or extended')
    for field in ('actual_tokenizer_fixture_observed', 'actual_CPU_continuation_observed',
                  'actual_GPU_positive_overlap_observed', 'prior_token_counts_or_responses_transferred'):
        if seed[field] is not False:
            raise ValueError('Source prompts cannot transfer runtime evidence')
    if [c['target_indices'] for c in seed['candidates']] != [[0, 1], [2, 3], [4, 5]]:
        raise ValueError('Preregistered complete candidate pairs changed')
    for phase, count in (('warm', 2), ('target', 6)):
        rows = seed['messages'][phase]
        if len(rows) != count or len({json.dumps(r, sort_keys=True) for r in rows}) != count:
            raise ValueError('Exact distinct eight-input roster required')
        for messages in rows:
            if [m['role'] for m in messages] != ['system', 'user']:
                raise ValueError('Exact simple conversation required')
            text = messages[1]['content']
            if not text.isascii() or 'one sentence of 12-20 words' not in text:
                raise ValueError('Explicit useful finite sentence request required')
    return {'schema': 2, 'input_count': 8, 'fresh_repeats_required': 2,
            'screen_cases_required': 16, 'candidate_order': ['water-cycle', 'library-returns', 'trail-signs'],
            'sampling_temperature': 0, 'sampling_seed': 1234,
            'natural_completion_tokens_required': [8, 63], 'max_new_tokens': 64,
            'selected_candidate': None, 'actual_token_counts_observed': False,
            'actual_GPU_positive_overlap_observed': False,
            'prior_response_or_token_count_transfer': False}
