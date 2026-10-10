"""Tiny source-shaped CPU records only; no actual runtime proof."""
import copy
import hashlib
import json
import struct
import unittest
import full_cache_shared_metadata_v2 as m


class Controls(unittest.TestCase):
    def fixture(self):
        begin = dict(kind='request_begin', pid=123, command=1, rid=3,
                     slot=-1, slotgen=0, tokens=2,
                     input_sha256_le32=hashlib.sha256(struct.pack('<II', 4, 5)).hexdigest(),
                     fresh=False, pin_present=False, pin=-1, raw_requested=True,
                     source_normal_dispatch='GEN')
        work = dict(kind='request_work', pid=123, command=1, rid=3, slot=-1,
                    slotgen=0, reused=0, read_from=0, reread_to=-1,
                    prompt_reached=2, prompt=2, generated=1, cancelled=False,
                    finish='length')
        text = '\n'.join('FC38 ' + json.dumps(r) for r in (begin, work))
        return m.records(text, 123), 'GEN 1 fresh=0 rid=3 4,5'

    def test_direct_gen_one_truthful_scope(self):
        rows, command = self.fixture()
        result = m.request_binding(rows, command, [3])
        self.assertEqual(result['actual_new_prompt_tokens'], 2)
        self.assertTrue(result['main_body_only'])
        self.assertFalse(result['actual_client_terminal_qualified'])
        self.assertFalse(result['actual_raw_capture_qualified'])

    def test_partial_work_not_whole_prompt(self):
        rows, command = self.fixture()
        rows[1].update(prompt_reached=1, generated=0, cancelled=True, finish='cancel')
        self.assertEqual(m.request_binding(rows, command, [3])['actual_new_prompt_tokens'], 1)
        rows[1]['cancelled'] = False
        self.assertRaises(ValueError, m.request_binding, rows, command, [3])

    def test_selection_input_owner_flags_refused(self):
        rows, command = self.fixture()
        for index, key, value in ((0, 'raw_requested', False), (0, 'fresh', 0),
                                  (0, 'input_sha256_le32', '0' * 64),
                                  (1, 'slotgen', 1), (1, 'generated', 2),
                                  (1, 'read_from', True), (0, 'pin_present', True)):
            bad = copy.deepcopy(rows)
            bad[index][key] = value
            with self.subTest(key=key):
                self.assertRaises(ValueError, m.request_binding, bad, command, [3])

    def test_metadata_parser_unique_pid_event_original_index(self):
        text = 'unrelated\nFC39 {"kind":"sample","pid":123,"event":4}\nnoise\nFC38 {"kind":"sample","pid":123,"event":5}'
        rows = m.records(text, 123)
        self.assertEqual([r['original_line_index'] for r in rows], [1, 3])
        self.assertRaises(ValueError, m.records, text.replace('"event":5', '"event":4'), 123)
        self.assertRaises(ValueError, m.records, text, 124)
        self.assertRaises(ValueError, m.records, 'FC38 {"kind":"x","pid":123,"pid":123}', 123)

    def test_unpaired_victim_and_absent_memory_fail_closed(self):
        self.assertRaises(ValueError, m.resource_binding, [])

    def test_same_rid_solo_main_body_is_command_local(self):
        rows, command = self.fixture()
        admission = copy.deepcopy(rows)
        for row in admission:
            row['slot'] = 0
            row['slotgen'] = 7
            row['command'] = 1
        admission[0]['source_normal_dispatch'] = 'BGEN'
        for row in rows:
            row['command'] = 2
            row['original_line_index'] += 2
        result = m.request_binding(admission + rows, command, [3])
        self.assertEqual(result['begin']['command'], 2)
        self.assertEqual(result['work']['slotgen'], 0)
        self.assertEqual(m.request_binding(admission + rows, 'BGEN 0 1 fresh=0 rid=3 4,5', [3])['work']['slotgen'], 7)

    def test_combined_fresh_pin_is_explicit_source_input(self):
        from full_cache_shared_http_client_v2 import validate_policy
        self.assertEqual(validate_policy({'strata_fresh': True, 'strata_shared_prefix': {'tokens': 272}})['strata_fresh'], True)
        self.assertRaises(ValueError, validate_policy, {'strata_fresh': True, 'strata_shared_prefix': {'tokens': True}})

    def test_logical_resources_and_real_victim_are_separate_scopes(self):
        from full_cache_memory39_contract_v1 import logical_sample
        row = logical_sample(budget_bytes=100, parked_bytes=8, reusable_bytes=0,
                             held_snapshot_bytes=4, incoming_estimate_bytes=10,
                             local_reuse_bytes=3, allocated_snapshot_bytes=2)
        row.update(kind='logical_memory', pid=123, cache_scope=1, event=1,
                   observed_owned_snapshot_peak_bytes=17,
                   logical_reservation_peak_bytes=22,
                   incoming_estimate_overlaps_owned_pending=True,
                   whole_process_peak_qualified=False, physical_reclamation_qualified=False,
                   device_physical_memory_qualified=False)
        before = dict(kind='parked_inventory', phase='before_eviction', pid=123,
                      command=1, rid=3, event=2, actual_victim=0,
                      retained_bytes=8, physical_allocator_reclamation_qualified=False,
                      entries=[dict(index=0, pinned=False, snapshot_bytes=8, instance=9)])
        after = dict(before, phase='after_eviction', event=3, actual_victim=-1,
                     retained_bytes=0, entries=[])
        result = m.resource_binding([row, before, after])
        self.assertTrue(result['real_parked_victim_observed'])
        self.assertFalse(result['actual_full_cache_qualified'])
        self.assertFalse(result['physical_device_memory_qualified'])
        self.assertRaises(ValueError, m.resource_binding, [row, before])
        changed = copy.deepcopy(after)
        changed['retained_bytes'] = 8
        self.assertRaises(ValueError, m.resource_binding, [row, before, changed])

    def test_unbound_fresh_arrays_refused_before_parent_source_access(self):
        import validate_full_cache_shared_runtime_v2 as reader
        self.assertRaisesRegex(ValueError, 'Caller-supplied fresh arrays', reader.finalized_binding,
                               '/tmp/no-actual-shared-proof', {'fabricated': {}})

    def test_source39_ownership_requires_all_three_actual_traces(self):
        from unittest.mock import patch
        import full_cache_shared_source39_ownership_v2 as own
        with patch.object(own, 'snapshot_audit', return_value={'passed': True}), patch.object(own, 'slot_ownership', return_value={'passed': True}), patch.object(own, 'mirror_audit', return_value={'passed': True}) as mirrors:
            result = own.ownership('CPU-only mock trace', [(0, 0, 32), (1, 32, 48)], 2)
            self.assertEqual(result['source_lane'], 'source39')
            self.assertFalse(result['historical_source37_runtime_proof_transferred'])
            mirrors.return_value = {'passed': False}
            self.assertRaises(ValueError, own.ownership, 'CPU-only mock trace', [(0, 0, 48)], 2)
            self.assertRaises(ValueError, own.ownership, 'CPU-only mock trace', [(0, 0, 48), (1, 0, 48)], 2)
            self.assertRaises(ValueError, own.ownership, 'CPU-only mock trace', [(0, 0, 47)], 2)


if __name__ == '__main__':
    unittest.main()
