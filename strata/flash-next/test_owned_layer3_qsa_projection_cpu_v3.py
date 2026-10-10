import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from unittest.mock import patch
import full48_owned_layer3_qsa_projection_v3 as composition
from owned_layer3_qsa_helper_contract_v3 import recollect, device_contract
import copy
from qsa_owned_state_storage_v1 import QsaConditionalProjection, QsaOwnedState
from owned_layer3_qsa_projection_v3 import OwnedLayer3ProjectionTap, save_owned


class Provider:
    actual_source = False
    def __init__(self):
        self.reader = SimpleNamespace(tensors={
            'blk.3.'+r: (None, {'type': k, 'shape_ggml_order': list(s)}, None)
            for r, (k, s) in QsaConditionalProjection.ROLES.items()})
    def rows(self, name, rows):
        shape = QsaConditionalProjection.ROLES[name.removeprefix('blk.3.')][1]
        return np.ones((len(list(rows)), shape[0]))


class Projector:
    tile = 64 << 20
    def __init__(self):
        self.calls = []
    def project(self, name, x, weight, activation):
        self.calls.append((name, weight, activation))
        shape = QsaConditionalProjection.ROLES[name.removeprefix('blk.3.')][1]
        return np.zeros(shape[1])


class Controls(unittest.TestCase):
    def producer(self):
        obj = composition.OwnedLayer3ProjectionProducer.__new__(composition.OwnedLayer3ProjectionProducer)
        obj._projection_producer_started = False; obj.identity = 'a'*64
        obj.p = SimpleNamespace(actual_source=True, source_identity_sha256='a'*64)
        obj.qsa = {3: SimpleNamespace(own_records=[], scope=lambda: {})}
        return obj

    def test_composition_exact_ids_identity_and_single_use_before_math(self):
        with patch.object(composition.OwnedHcDeviceRsReference, 'tokens', return_value={}) as compute:
            obj = self.producer()
            for ids in ([1, 2, 3, 4], composition.ACCEPTED_IDS[:3], [True, 8678, 198, 15666]):
                with self.assertRaises(ValueError):
                    obj.tokens(ids)
            self.assertEqual(compute.call_count, 0)
            obj.p.source_identity_sha256 = 'b'*64
            with self.assertRaises(ValueError):
                obj.tokens(composition.ACCEPTED_IDS)
            self.assertEqual(compute.call_count, 0)
            obj.p.source_identity_sha256 = 'a'*64
            obj.tokens(composition.ACCEPTED_IDS)
            self.assertEqual(compute.call_count, 1)
            with self.assertRaises(ValueError):
                obj.tokens(composition.ACCEPTED_IDS)
            self.assertEqual(compute.call_count, 1)

    def test_failed_own_compute_cannot_reuse_producer(self):
        with patch.object(composition.OwnedHcDeviceRsReference, 'tokens', side_effect=RuntimeError('synthetic')) as compute:
            obj = self.producer()
            with self.assertRaises(RuntimeError):
                obj.tokens(composition.ACCEPTED_IDS)
            with self.assertRaises(ValueError):
                obj.tokens(composition.ACCEPTED_IDS)
            self.assertEqual(compute.call_count, 1)

    def test_four_owned_rows_publish_outputs_without_changed_detail(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Provider(); projector = Projector()
            tap = OwnedLayer3ProjectionTap(p, projector, Path(directory)/'own', 'a'*64)
            state = QsaOwnedState(indexer_key_gamma=np.ones(128), max_cells=8)
            for pos in range(4):
                detail = tap.step(state, composition.ACCEPTED_IDS[pos], np.arange(2560, dtype='<f4'))
                self.assertEqual(list(detail['selected_ids_owned']), list(range(pos+1)))
                self.assertTrue(np.array_equal(detail['block_output'], np.zeros(2560)))
            self.assertEqual(len(tap.own_records), 4)
            self.assertEqual(len(projector.calls), 24)
            for row in tap.own_records:
                self.assertEqual(len(row['projections']), 6)
                self.assertEqual(len(row['vectors']), 4)
                for binding in row['projections']+row['vectors']+[row['own_HC_mixed_input']]:
                    raw = Path(binding['path']).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), binding['sha256'])
                    self.assertEqual(len(raw), binding['bytes'])
                    self.assertFalse(binding['captured_operand_used'])
                    self.assertFalse(binding['snapshot_is_math_input'])
            self.assertFalse(tap.scope()['full_model_math_qualified'])
            self.assertFalse(tap.scope()['original_projection_and_QSA_math_changed'])
            arrays = recollect(tap.own_records, tap.output, 'a'*64, False)
            self.assertEqual(arrays[0]['projections']['attn_q.weight'].shape, (12288,))
            with self.assertRaises(ValueError):
                recollect(tap.own_records, tap.output, 'a'*64, True)
            bad = copy.deepcopy(tap.own_records); bad[1]['token'] = 1
            with self.assertRaises(ValueError):
                recollect(bad, tap.output, 'a'*64, False)
            bad = copy.deepcopy(tap.own_records); bad[1]['predecessor_cells'] = True
            with self.assertRaises(ValueError):
                recollect(bad, tap.output, 'a'*64, False)
            path = Path(tap.own_records[0]['projections'][0]['path'])
            path.write_bytes(bytes(path.stat().st_size))
            # The synthetic projection was zero; mutate one consumed byte.
            raw = bytearray(path.read_bytes()); raw[0] = 1; path.write_bytes(raw)
            with self.assertRaises(ValueError):
                recollect(tap.own_records, tap.output, 'a'*64, False)

    def test_helper_contract_distinguishes_outputs_from_internal_witnesses(self):
        contract = device_contract()
        self.assertFalse(contract['runtime_ready'])
        self.assertFalse(contract['standalone_norm_is_fused_internal_witness'])
        self.assertFalse(contract['score_softmax_internal_witness_observed'])
        self.assertEqual(contract['query']['input_stride'], 512)
        self.assertEqual([r['rows'] for r in contract['windows']], [2, 1, 1])

    def test_nonzero_history_wrong_route_and_fifth_row_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            tap = OwnedLayer3ProjectionTap(Provider(), Projector(), Path(directory)/'own', 'b'*64)
            state = QsaOwnedState(max_cells=8)
            with self.assertRaises(ValueError):
                tap.step(state, 1, np.zeros(2560), 'prefill')
            state.keys.append(np.zeros((2, 256)))
            with self.assertRaises(ValueError):
                tap.step(state, 1, np.zeros(2560))
            state.reset()
            state.dead[0] = 1
            with self.assertRaises(ValueError):
                tap.step(state, 1, np.zeros(2560))
            state.reset()
            for n in range(4):
                tap.step(state, n, np.zeros(2560))
            with self.assertRaises(ValueError):
                tap.step(state, 4, np.zeros(2560))

    def test_snapshots_exclusive_finite_confined_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_owned(root, 'own.f32', np.ones(128), 'own', 0, 'a'*64, False)
            with self.assertRaises(FileExistsError):
                save_owned(root, 'own.f32', np.zeros(128), 'own', 0, 'a'*64, False)
            for name, values in (('../escaped.f32', np.zeros(128)),
                                 ('nan.f32', np.full(128, np.nan)),
                                 ('large.f32', np.zeros(12289))):
                with self.assertRaises(ValueError):
                    save_owned(root, name, values, 'own', 0, 'a'*64, False)


if __name__ == '__main__':
    unittest.main()
