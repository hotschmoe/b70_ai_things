"""Bounded own QSA decoder candidate; device lowering remains unobserved.

All nonlinear operations and explicit FMA/division enter a tagged operation
provider. CPU substitutes are synthetic controls only. No captured values,
selected IDs, or device output lookup are accepted by this module.
"""
import numpy as np
from owned_layer3_qsa_contract_v1 import f32, xor32, require, all_cell_ids

SCORE_CANDIDATE = 'eight products with separate F32 adds, then XOR32'


def finite(value, shape, label):
    a = f32(value)
    require(a.shape == shape and np.isfinite(a).all(), label + ' geometry/finite values differ')
    return a


class OwnOps:
    def __init__(self, provider, synthetic):
        require(type(synthetic) is bool and callable(provider), 'Explicit operation provider required')
        self.provider = provider
        self.synthetic = synthetic
        self.records = []

    def call(self, operation, operands, role):
        require(operation in ('native_exp', 'fma', 'divide') and type(role) is str and role,
                'Explicit source operation/role required')
        args = f32(operands)
        require(args.ndim == 1 and len(args) == {'native_exp': 1, 'fma': 3, 'divide': 2}[operation]
                and np.isfinite(args).all(), 'Own operation operands differ')
        if operation == 'divide':
            require(args[1] > 0, 'Own positive denominator required')
        result = np.float32(self.provider(operation, args.copy(), role))
        require(np.isfinite(result), 'Own operation result must be finite')
        if operation == 'native_exp':
            require(result >= 0, 'Own exp result cannot be negative')
        self.records.append({'ordinal': len(self.records) + 1, 'operation': operation,
                             'role': role, 'operands_hex': args.tobytes().hex(),
                             'result_hex': result.tobytes().hex(),
                             'synthetic': self.synthetic, 'captured_operand_used': False})
        return result


def score_candidate(query, key):
    """One explicit candidate, not a claim about compiler contraction."""
    q = finite(query, (256,), 'Own query')
    k = finite(key, (256,), 'Own key')
    lane = np.empty(32, dtype='<f4')
    for i in range(32):
        products = f32(q[8*i:8*i+8] * k[8*i:8*i+8])
        total = products[0]
        for x in products[1:]:
            total = np.float32(total + x)
        lane[i] = total
    return np.float32(np.float32(xor32(lane)) * np.float32(0.0625))


def decode_owned(query, keys, values, gate, ops):
    """One chunk, 1..4 cells, original default FP16 pools and all own IDs.

The second chunk-reduction stage is retained even for one chunk: its exp(0),
FMA, and division are real operations in the dispatched source contract.
"""
    require(isinstance(ops, OwnOps), 'Own tagged operation provider required')
    q = finite(query, (24, 256), 'Own query')
    g = finite(gate, (24, 256), 'Own gate')
    k = f32(keys); v = f32(values)
    require(k.ndim == 3 and 1 <= k.shape[0] <= 4 and k.shape[1:] == (2, 256),
            'Own bounded KV history required')
    k = finite(k, k.shape, 'Own keys'); v = finite(v, k.shape, 'Own values')
    k = k.astype('<f2').astype('<f4'); v = v.astype('<f2').astype('<f4')
    require(np.isfinite(k).all() and np.isfinite(v).all(), 'FP16 pool overflow')
    ids = all_cell_ids(len(k))
    scores = np.empty((24, len(k)), dtype='<f4')
    weights = np.empty_like(scores); attention = np.empty_like(q)
    denominators = np.empty(24, dtype='<f4')
    for h in range(24):
        kv = h // 12
        for c in ids:
            scores[h, c] = score_candidate(q[h], k[c, kv])
        maximum = np.float32(scores[h].max())
        lanes = np.zeros(32, dtype='<f4')
        for c in ids:
            weight = ops.call('native_exp', [np.float32(scores[h, c] - maximum)],
                              'head'+str(h)+'.cell'+str(c)+'.softmax')
            weights[h, c] = weight; lanes[c] = weight
        chunk_sum = np.float32(xor32(lanes))
        chunk_weight = ops.call('native_exp', [np.float32(0)], 'head'+str(h)+'.chunk0.weight')
        total = ops.call('fma', [chunk_sum, chunk_weight, 0], 'head'+str(h)+'.chunk0.denominator')
        denominators[h] = total
        require(total > 0, 'Own softmax denominator must be positive')
        for d in range(256):
            accum = np.float32(0)
            for c in ids:
                accum = ops.call('fma', [weights[h, c], v[c, kv, d], accum],
                                 'head'+str(h)+'.dim'+str(d)+'.cell'+str(c))
            accum = ops.call('fma', [accum, chunk_weight, 0],
                             'head'+str(h)+'.dim'+str(d)+'.chunk0')
            attention[h, d] = ops.call('divide', [accum, total],
                                      'head'+str(h)+'.dim'+str(d)+'.attention')
    gated = np.empty_like(q)
    for h in range(24):
        for d in range(256):
            e = ops.call('native_exp', [np.float32(-g[h, d])],
                         'head'+str(h)+'.dim'+str(d)+'.gate_exp')
            sigmoid = ops.call('divide', [1, np.float32(1 + e)],
                              'head'+str(h)+'.dim'+str(d)+'.gate_sigmoid')
            gated[h, d] = np.float32(attention[h, d] * sigmoid)
    return {'scores_own': scores, 'softmax_weights_unnormalized_own': weights,
            'softmax_denominators_own': denominators, 'attention_own': attention,
            'attention_gated_own': gated, 'selected_ids_owned': ids,
            'FP16_keys_owned': k, 'FP16_values_owned': v,
            'score_candidate': SCORE_CANDIDATE, 'score_contraction_observed': False,
            'device_intrinsics_qualified': False, 'synthetic': ops.synthetic,
            'captured_operands_or_selected_IDs_used': False,
            'full_model_math_qualified': False}
