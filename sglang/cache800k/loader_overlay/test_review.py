import copy
import json
from pathlib import Path
import unittest
from review_fresh import check_rows
ROOT=Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910/fresh-calibration-plan/record-eager-prefixoff/02-calibrate')

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.corpus=json.loads((ROOT/'corpus.json').read_bytes())
        self.rows=[json.loads(x) for x in (ROOT/'responses.jsonl').read_bytes().splitlines()]
    def test_actual_completed_corpus(self):check_rows(self.corpus,self.rows)
    def test_mutation_rejections(self):
        mutations=[lambda x:x.pop(),lambda x:x.__setitem__(-1,copy.deepcopy(x[-2])),
                   lambda x:x[0].update(prompt_sha256='0'*64),
                   lambda x:x[0].update(client_cancelled=True),
                   lambda x:x[0].update(finish_reason='error'),
                   lambda x:x[0]['usage'].update(completion_tokens=True)]
        for mutate in mutations:
            rows=copy.deepcopy(self.rows);mutate(rows)
            with self.assertRaises(AssertionError):check_rows(self.corpus,rows)
if __name__=='__main__':unittest.main(verbosity=2)
