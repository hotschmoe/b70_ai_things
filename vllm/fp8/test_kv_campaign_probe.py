import unittest
from kv_campaign_probe import generation_issue


class GenerationGateTest(unittest.TestCase):
    def row(self, text, finish='length'):
        return {'text': text, 'finish_reason': finish, 'error': None}

    def test_valid_markdown_ruler(self):
        self.assertIsNone(generation_issue(self.row('A detailed guide to cache testing.\n' + '-' * 100)))

    def test_observed_malformed_line_loop(self):
        row = self.row('A detailed guide.\n' + '    def __contains__(self, self, self._map)\n' * 8)
        self.assertEqual(generation_issue(row), 'repeated-line loop')

    def test_stopped_unclosed_code(self):
        row = self.row('A detailed guide. ' * 10 + '\n```python\ndef broken() ->, |', 'stop')
        self.assertEqual(generation_issue(row), 'unclosed code fence at EOS')

    def test_budget_truncation_is_distinct(self):
        row = self.row('A detailed guide. ' * 10 + '\n```python\ndef partial(')
        self.assertIsNone(generation_issue(row))


if __name__ == '__main__':
    unittest.main()
