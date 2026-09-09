import unittest
from full_feature_validate import check_features, counter


class FullFeatureGate(unittest.TestCase):
    def test_refuses_feature_disabled(self):
        args = dict(mtp=3, eager=False, prefix_off=False, kv_dtype='fp8_e4m3', offload_gib=0)
        m = dict(args=args, command=['VLLM_XPU_ENABLE_XPU_GRAPH=1', '--compilation-config',
                                    '{"mode":3,"cudagraph_mode":"PIECEWISE"}'])
        check_features(m)
        for key, value in [('mtp', 0), ('eager', True), ('prefix_off', True), ('kv_dtype', 'auto')]:
            with self.assertRaises(RuntimeError):
                check_features(dict(m, args=dict(args, **{key: value})))

    def test_counter_requires_evidence_and_sums_ranks(self):
        self.assertEqual(counter('x{rank="0"} 2.0\nx{rank="1"} 3\n', 'x'), 5)
        with self.assertRaises(RuntimeError):
            counter('# HELP x count\n', 'x')
