"""CPU tests: failure accounting and actual FP16 arithmetic without XPU."""
import unittest
from unittest.mock import Mock

from collective_control import check_environment, fenced_reduce, make_input


class ControlTests(unittest.TestCase):
    def test_environment_including_absent_keys(self):
        expected = {'CCL_TOPO_P2P_ACCESS': '0', 'LD_PRELOAD': None}
        check_environment(expected, {'CCL_TOPO_P2P_ACCESS': '0'})
        for actual in ({'CCL_TOPO_P2P_ACCESS': '1'},
                       {'CCL_TOPO_P2P_ACCESS': '0', 'LD_PRELOAD': 'other'}):
            with self.assertRaises(RuntimeError):
                check_environment(expected, actual)

    def test_success_order_and_explicit_subgroup(self):
        stages = []
        reduction = Mock()
        fenced_reduce('tensor', 'subgroup', Mock(), reduction, stages.append)
        reduction.assert_called_once_with('tensor', group='subgroup')
        self.assertEqual(stages, ['pre_fence_entry', 'pre_fence_return',
                                 'collective_host_entry', 'collective_host_return_not_completion',
                                 'post_fence_entry', 'post_fence_return'])

    def test_failures_never_claim_completion(self):
        for failing_stage in ('pre', 'reduce', 'post'):
            with self.subTest(stage=failing_stage):
                stages = []
                sync = Mock(side_effect=[RuntimeError('pre')] if failing_stage == 'pre'
                            else [None, RuntimeError('post')] if failing_stage == 'post'
                            else [None, None])
                reduction = Mock(side_effect=RuntimeError('reduce')
                                 if failing_stage == 'reduce' else None)
                with self.assertRaises(RuntimeError):
                    fenced_reduce('tensor', 'group', sync, reduction, stages.append)
                self.assertNotIn('post_fence_return', stages)
                if failing_stage == 'pre':
                    reduction.assert_not_called()
                if failing_stage == 'reduce':
                    self.assertNotIn('post_fence_entry', stages)

    def test_full_shape_cpu_fp16_oracle(self):
        import torch
        for rows in (4, 2048):
            for iteration in range(3):
                rank0 = make_input(torch, rows, 0, iteration, 'cpu')
                rank1 = make_input(torch, rows, 1, iteration, 'cpu')
                self.assertEqual(rank0.stride(), (5120, 1))
                self.assertEqual(rank0.numel() * rank0.element_size(), rows * 5120 * 2)
                self.assertTrue(torch.equal(rank0 + rank1, rank0 * 2 + 3))
                bad = rank0 + rank1
                bad[-1, -1] += 1
                self.assertFalse(torch.equal(bad, rank0 * 2 + 3))


if __name__ == '__main__':
    unittest.main()
