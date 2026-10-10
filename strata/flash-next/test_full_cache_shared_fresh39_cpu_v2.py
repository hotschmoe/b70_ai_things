"""Tiny source/roster/profile controls; no synthetic positive runtime proof."""
import copy
import unittest
from unittest.mock import patch
import full_cache_shared_fresh39_v2 as fresh
import validate_full_cache_shared_fresh39_v2 as admission


class Controls(unittest.TestCase):
    def group(self, rid, role, ids):
        return {'rid': rid, 'role': role, 'input_ids': ids,
                'vectors': {str(l): {'CPU_TEST_ONLY': True} for l in range(-1, 48)}}

    def test_full_actor_job_roster_grouped_not_shrunk(self):
        groups = [self.group(i, 'direct_gen', [10, i]) for i in range(1, 15)]
        result = fresh.job_roster(groups)
        self.assertEqual([len(g) for g in result['groups']], [6, 6, 2])
        self.assertEqual(len(result['cached_group_associations']), 14)
        self.assertFalse(result['cached_state_imported'])
        self.assertFalse(result['actual_fresh_execution_observed'])

    def test_equal_prefix_reuses_only_same_fresh_math_control(self):
        groups = [self.group(3, 'admission', [10, 11]), self.group(4, 'direct_gen', [10, 11])]
        result = fresh.job_roster(groups)
        self.assertEqual(len(result['jobs']), 1)
        self.assertEqual(len(result['cached_group_associations']), 2)
        self.assertIsNone(result['jobs'][0]['pin'])
        self.assertEqual(result['jobs'][0]['fresh'], 1)

    def test_partial49_and_invalid_actual_ids_fail_closed(self):
        group = self.group(3, 'admission', [10, 11])
        group['vectors'].pop('47')
        self.assertRaises(ValueError, fresh.job_roster, [group])
        self.assertRaises(ValueError, fresh.job_roster, [self.group(3, 'admission', [True])])
        self.assertRaises(ValueError, fresh.job_roster, [])

    def test_source39_fresh_profile_preserves_math_and_disables_cache(self):
        actor = {'args': ['--batch', '2', '--prompt-cache', '3', '--conversation-cache-mib', '512',
                          '--suffix-draft', '0', '--lookup-chain', '0', '--prefill', '64', '--max-context', '2048'],
                 'env': {'STRATA_FULL_CACHE_OBSERVER38': '1', 'STRATA_FULL_CACHE_CAPTURE_RIDS38': '3,4',
                         'STRATA_BATCH_FULL_STATE_CHAIN': '1', 'STRATA_BATCH_PUBLIC_PREFIX': '1'}}
        original = copy.deepcopy(actor)
        args, env = fresh.fresh_recipe(actor)
        self.assertEqual(actor, original)
        self.assertEqual(args[args.index('--batch') + 1], '0')
        self.assertEqual(args[args.index('--prefill') + 1], '64')
        self.assertEqual(env['STRATA_FULL_CACHE_OBSERVER38'], '0')
        self.assertNotIn('STRATA_FULL_CACHE_CAPTURE_RIDS38', env)
        self.assertEqual(env['STRATA_FIDELITY_DIAG'], '1')
        actor['env']['STRATA_VERIFY_EAGER'] = '0'
        self.assertRaises(ValueError, fresh.fresh_recipe, actor)

    def test_controls_require_every_independent_actor_not_caller_arrays(self):
        roster = fresh.job_roster([self.group(i, 'direct_gen', [10, i]) for i in range(1, 8)])
        with patch.object(fresh, 'actual_groups', return_value=[self.group(i, 'direct_gen', [10, i]) for i in range(1, 8)]), patch.object(fresh, 'read', return_value={'engine_receipt_sha256': 'CPU_ONLY'}):
            self.assertRaises(ValueError, admission.complete_controls, '/tmp/nonexistent-actor', ['/tmp/one-control'])
            self.assertRaises(ValueError, admission.complete_controls, '/tmp/nonexistent-actor', ['/tmp/same', '/tmp/same'])
        self.assertEqual(len(roster['groups']), 2)
