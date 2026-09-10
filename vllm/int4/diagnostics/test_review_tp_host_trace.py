import copy
import json
from pathlib import Path
import tempfile
import unittest

from review_tp_host_trace import review


class TraceReviewTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for rank in (0, 1):
            base = dict(pid=rank + 100, id=0, rank=rank, phase='profile',
                        arguments={'num_tokens': 32768}, counts={})
            rows.append(dict(base, event='dummy_enter'))
            comm = dict(rank_in_group=rank, world_size=2, unique_name='tp:0')
            call = dict(pid=rank + 100, scope=0, call=1, rank=rank,
                        op='all_reduce', communicator=comm)
            rows.append(dict(call, event='collective_enter'))
            rows.append(dict(call, event='collective_return'))
            shape = json.dumps(dict(op='all_reduce', input=dict(
                shape=[32768, 5120], stride=[5120, 1], dtype='torch.float16',
                device='xpu:' + str(rank))), sort_keys=True)
            rows.append(dict(base, event='dummy_return',
                             counts={'all_reduce_enter': 1, 'all_reduce_return': 1},
                             shape_counts={shape: 1}))
        return rows

    def run_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'host-test.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            return review([path])

    def test_complete_different_device_ordinals(self):
        self.assertTrue(self.run_rows(self.fixture())['passed'])

    def test_incomplete_and_truncated_rejected(self):
        rows = self.fixture()
        for index in (0, 1, 2, 3, 7):
            self.assertFalse(self.run_rows(rows[:index] + rows[index + 1:])['passed'])
        self.assertFalse(self.run_rows(rows + [dict(pid=100, event='truncated')])['passed'])
        self.assertFalse(self.run_rows([])['passed'])
        self.assertFalse(self.run_rows([r for r in rows if not r['event'].startswith('collective_')])['passed'])

    def test_mismatched_rank_counts_shapes_and_identity(self):
        for field in ('counts', 'shape_counts', 'communicator'):
            rows = copy.deepcopy(self.fixture())
            if field == 'counts':
                rows[-1]['counts']['all_reduce_enter'] = 2
            elif field == 'shape_counts':
                rows[-1]['shape_counts'] = {}
            else:
                rows[-2]['communicator'] = dict(rows[-2]['communicator'], world_size=3)
            self.assertFalse(self.run_rows(rows)['passed'])


if __name__ == '__main__':
    unittest.main()
