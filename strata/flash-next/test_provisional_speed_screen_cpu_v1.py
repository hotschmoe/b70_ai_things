import io
import json
import unittest
import provisional_speed_screen_v1 as s


def wire(content='water', count=128, finish='length', cached=0):
    rows = [': keepalive\n\n', 'data: '+json.dumps({'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None}]})+'\n\n']
    for part in (content, ' and soil'):
        rows.append('data: '+json.dumps({'choices': [{'delta': {'content': part}, 'finish_reason': None}]}, ensure_ascii=False)+'\n\n')
    rows.append('data: '+json.dumps({'choices': [{'delta': {}, 'finish_reason': finish}],
        'usage': {'prompt_tokens': 512, 'completion_tokens': count, 'total_tokens': 512+count,
                  'prompt_tokens_details': {'cached_tokens': cached}}, 'timings': {'prompt_ms': 10}})+'\n\n')
    rows.append('data: [DONE]\n\n')
    return ''.join(rows).encode()


class Controls(unittest.TestCase):
    def parse(self, raw):
        n = iter(range(1000000, 100000000, 1000000))
        return s.parse_stream(io.BytesIO(raw), 0, lambda: next(n))

    def test_chunks_are_not_tokens(self):
        v = self.parse(wire())
        self.assertEqual(v['usage']['completion_tokens'], 128)
        self.assertEqual(v['text_chunk_count'], 2)
        self.assertFalse(v['true_per_token_gaps_observed'])
        self.assertFalse(v['device_prefill_rate_observed'])
        self.assertEqual(len(v['client_text_chunk_gaps_s']), 1)

    def test_unicode_raw_preserved(self):
        raw = wire('caf\u00e9')
        v = self.parse(raw)
        self.assertEqual(b''.join(bytes.fromhex(x['line_hex']) for x in v['raw_lines']), raw)
        self.assertEqual(v['output_text'], 'caf\u00e9 and soil')

    def test_natural_eos_short_is_not_128(self):
        v = self.parse(wire(count=18, finish='stop'))
        self.assertFalse(v['completion_tokens_at_least128'])
        self.assertEqual(v['finish_reason'], 'stop')

    def test_missing_done_reject(self):
        with self.assertRaises(ValueError): self.parse(wire().replace(b'data: [DONE]\n\n', b''))

    def test_cached_reject(self):
        with self.assertRaises(ValueError): self.parse(wire(cached=1))

    def test_error_reject(self):
        with self.assertRaises(ValueError): self.parse(b'data: {"error":"engine died"}\n\ndata: [DONE]\n\n')

    def test_truncated_reject(self):
        with self.assertRaises(ValueError): self.parse(wire()[:-1])

    def test_event_after_done_reject(self):
        with self.assertRaises(ValueError): self.parse(wire()+b'data: {}\n\n')

    def test_usage_bool_reject(self):
        with self.assertRaises(ValueError): self.parse(wire().replace(b'"completion_tokens": 128', b'"completion_tokens": true'))

    def test_prompt_corpus_identical_and_useful(self):
        p = s.prompts()
        self.assertTrue(p['prefill_about512'].endswith(p['short']))
        self.assertNotIn('ignore_eos', p['short'])

    def test_failure_retains_arrived_raw(self):
        retained = []
        with self.assertRaises(ValueError):
            s.parse_stream(io.BytesIO(b'data: {"error":"native failure"}\n\n'), 0,
                           lambda: 100, retained.append)
        self.assertEqual(bytes.fromhex(retained[0]['line_hex']), b'data: {"error":"native failure"}\n')


if __name__ == '__main__': unittest.main()
