"""Synthetic full-size zero arrays exercise admission only, never GPU proof."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import full_cache_shared_raw49_v2 as raw


class Controls(unittest.TestCase):
    def test_source39_direct_geometry_refuses_body_and_solo_events(self):
        row = dict(rows='1', active_mask='64', selected_mask='1', admission='3', event='0')
        self.assertEqual(raw.replay_geometry(row), 3)
        for field, value in (('event', '1'), ('rows', '2'), ('active_mask', '1'), ('admission', '4')):
            bad = dict(row)
            bad[field] = value
            self.assertRaises(ValueError, raw.replay_geometry, bad)
        self.assertEqual(raw.replay_role(3), 'direct_gen')
        self.assertEqual(raw.vector_role('direct_gen_logits_before_sampler'), 'direct_gen')
        self.assertRaises(ValueError, raw.vector_role, 'arbitrary_logits')

    def test_actual_shape_direct_route_complete49_cpu_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = 'pid=123 rid=3 enginegen=1 requestgen=1 slotgen=0'
            lines = ['SBF direct_gen_begin ' + identity + ' slot=6 prompt=2 event=0 normal_dispatch=GEN observed_only=1',
                     'SBF resume ' + identity + ' slot=6 reused=0 read_from=0 reread_to=-1']
            for begin, end, phase in ((0, 1, 'prefill_chunk'), (1, 2, 'direct_gen_target_verify')):
                lines.append('SBF span ' + identity + ' slot=6 stage=0 lb=0 le=48 completed=1 begin=' + str(begin) + ' end=' + str(end) + ' phase=' + phase)
            proof = 'pid=123 stage=0 epoch=1 enginegen=1 graph=0 event=0 admission=3 lb=0 le=48'
            lines.append('SBF replay ' + proof + ' rows=1 active_mask=64 selected_mask=1 full_roster=1 input_output_verified=1 stage_context_verified=1 geometry=48x4x2560x248320')
            row = identity + ' stage=0 epoch=1 graph=0 event=0 admission=3 slot=6 pos=1 token=5 row=0 batchrows=1 selected=1'
            lines.append('SBF row ' + row)
            for layer in range(-1, 48):
                count = 248320 if layer == -1 else 10240
                path = root / ('field' + str(layer) + '.f32')
                path.write_bytes(b'\0' * (count * 4))
                phase = 'direct_gen_logits_before_sampler' if layer == -1 else 'direct_gen_residual'
                lines.append('SBF vector ' + row + ' lb=0 le=48 layer=' + str(layer) + ' phase=' + phase + ' canonical=le_f32 bytes=' + str(count * 4) + ' floats=' + str(count) + ' file=' + path.name)
            work = [dict(call=1, rid=3, pid=123, slotgen=0, input_ids=[4, 5], actual_reused=0)]
            jobs = {'jobs': [dict(call=1, rid=3, role='direct_gen', ids=[4, 5], position=1, token=5)]}
            with patch.object(raw, 'prefix_jobs', return_value=jobs):
                result = raw.collect('\n'.join(lines), [], work, [(0, 0, 48)], root, {3: ['direct_gen']})
                self.assertEqual(result['actual_raw_fields'], 49)
                self.assertFalse(result['full_cache_runtime_qualified'])
                self.assertTrue(result['fresh_numerical_control_still_required'])
                self.assertRaises(ValueError, raw.collect, '\n'.join(lines[:-1]), [], work,
                                  [(0, 0, 48)], root, {3: ['direct_gen']})


if __name__ == '__main__':
    unittest.main()
