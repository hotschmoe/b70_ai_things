import unittest
import numpy as np
from owned_layer3_qsa_decode_candidate_v2 import OwnOps, decode_owned, score_candidate


def synthetic(operation, operands, role):
    x = operands.astype(np.float64)
    if operation == 'native_exp':
        return np.float32(np.exp(x[0]))
    if operation == 'fma':
        return np.float32(x[0] * x[1] + x[2])
    return np.float32(x[0] / x[1])


class CandidateTests(unittest.TestCase):
    def test_score_known_and_dimension_order(self):
        self.assertEqual(score_candidate(np.ones(256), np.ones(256)), 16)
        q = np.zeros(256); k = np.ones(256); q[0] = 16
        self.assertEqual(score_candidate(q, k), 1)

    def test_one_cell_source_stages_and_scope(self):
        ops = OwnOps(synthetic, True)
        result = decode_owned(np.ones((24, 256)), np.ones((1, 2, 256)),
                              np.ones((1, 2, 256)) * 4, np.zeros((24, 256)), ops)
        self.assertTrue(np.array_equal(result['attention_own'], np.full((24, 256), 4)))
        self.assertTrue(np.array_equal(result['attention_gated_own'], np.full((24, 256), 2)))
        self.assertEqual(result['selected_ids_owned'], [0])
        self.assertFalse(result['score_contraction_observed'])
        self.assertFalse(result['device_intrinsics_qualified'])
        self.assertFalse(result['full_model_math_qualified'])
        self.assertTrue(any(r['role'].endswith('.chunk0.weight') for r in ops.records))
        self.assertEqual(sum(r['operation'] == 'fma' for r in ops.records), 24 * (256 * 2 + 1))
        self.assertTrue(all(r['synthetic'] and not r['captured_operand_used'] for r in ops.records))

    def test_two_equal_scores_owned_fp16_and_value_average(self):
        ops = OwnOps(synthetic, True)
        values = np.stack([np.ones((2, 256)), np.ones((2, 256)) * 3])
        result = decode_owned(np.zeros((24, 256)), np.zeros((2, 2, 256)),
                              values, np.zeros((24, 256)), ops)
        self.assertTrue(np.array_equal(result['attention_own'], np.full((24, 256), 2)))
        self.assertEqual(result['selected_ids_owned'], [0, 1])
        self.assertTrue(np.array_equal(result['softmax_denominators_own'], np.full(24, 2)))

    def test_invalid_history_nan_and_provider_refusal(self):
        args = [np.zeros((24, 256)), np.zeros((1, 2, 256)),
                np.zeros((1, 2, 256)), np.zeros((24, 256))]
        for index, replacement in ((0, np.zeros((23, 256))),
                                   (1, np.zeros((5, 2, 256))),
                                   (2, np.full((1, 2, 256), np.nan))):
            bad = list(args); bad[index] = replacement
            with self.assertRaises(ValueError):
                decode_owned(*bad, OwnOps(synthetic, True))
        with self.assertRaises(ValueError):
            decode_owned(*args, OwnOps(lambda *x: np.nan, True))

    def test_operation_records_bind_exact_arguments_and_result(self):
        ops = OwnOps(synthetic, True)
        value = ops.call('fma', [1, 2, 3], 'own.test')
        self.assertEqual(value, 5)
        self.assertEqual(ops.records[0]['operands_hex'], np.asarray([1, 2, 3], dtype='<f4').tobytes().hex())
        self.assertEqual(ops.records[0]['result_hex'], np.float32(5).tobytes().hex())
        with self.assertRaises(ValueError):
            ops.call('divide', [1, 0], 'own.test')


if __name__ == '__main__':
    unittest.main()
