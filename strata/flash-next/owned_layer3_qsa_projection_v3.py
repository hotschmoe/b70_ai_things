"""Output-only own layer3 projection tap; original arithmetic is inherited.

Root-only actual use requires the original provider and the closed HC reference
prerequisites. This module neither loads native arrays nor accepts token states.
"""
import hashlib
from pathlib import Path
import numpy as np
from full48_owned_composition_storage_v1 import OwnedQsa
from owned_layer3_qsa_contract_v1 import require


def save_owned(root, name, value, role, position, identity, actual_source):
    a = np.asarray(value, dtype='<f4')
    require(a.size > 0 and a.nbytes <= 49152 and np.isfinite(a).all(), 'Bounded finite own QSA tensor required')
    raw = a.tobytes(); path = Path(root) / name
    require(path.parent.resolve() == Path(root).resolve(), 'Confined own QSA snapshot required')
    with path.open('xb') as stream:
        stream.write(raw)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'shape': list(a.shape), 'encoding': 'LE_F32',
            'role': role, 'position': position, 'source_identity_sha256': identity,
            'actual_original_provider': actual_source, 'captured_operand_used': False,
            'snapshot_is_math_input': False}


class OwnedLayer3ProjectionTap(OwnedQsa):
    """Save values returned by unchanged original projection/vector methods."""
    def __init__(self, provider, projector, output, identity):
        require(type(provider.actual_source) is bool and type(identity) is str
                and len(identity) == 64 and all(c in '0123456789abcdef' for c in identity),
                'Explicit original identity/provider scope required')
        self.output = Path(output).resolve(); self.output.mkdir(parents=True, exist_ok=False)
        self.identity = identity; self.own_records = []; self.active = None
        super().__init__(provider, projector, 3)

    def publish(self, kind, role, value):
        require(self.active is not None, 'Projection publication requires active own token')
        number = len(self.active[kind])
        name = 'p'+str(self.active['position'])+'-'+kind+'-'+str(number)+'.f32'
        binding = save_owned(self.output, name, value, 'blk.3.'+role,
                             self.active['position'], self.identity, self.p.actual_source)
        self.active[kind].append(binding)
        return binding

    def project(self, role, x, route):
        require(self.active is not None and route == 'verifier', 'Bounded original verifier projection only')
        before = np.asarray(x, dtype='<f4').tobytes()
        input_binding = save_owned(
            self.output, 'p'+str(self.active['position'])+'-projection-input-'+str(len(self.active['projections']))+'.f32',
            x, 'blk.3.'+role+'.own_input', self.active['position'], self.identity, self.p.actual_source)
        value = super().project(role, x, route)
        row = self.publish('projections', role, value)
        row.update(input_sha256=hashlib.sha256(before).hexdigest(),
                   own_input_snapshot=input_binding,
                   activation_storage='Q8_1' if self.ROLES[role][0] == 'Q8_0' else 'F32',
                   weight_type=self.ROLES[role][0],
                   original_projection_arithmetic_changed=False)
        return value

    def vector(self, role):
        value = super().vector(role)
        if self.active is not None:
            self.publish('vectors', role, value)
        return value

    def step(self, state, token, mixed, route='verifier', evaluate_attention=True):
        require(self.active is None and len(self.own_records) < 4 and route == 'verifier'
                and evaluate_attention is True and type(token) is int,
                'Exact four-row own verifier tap required')
        position = len(self.own_records)
        require(len(state.keys) == position and state.pos_base == 0,
                'Own zero-history sequential predecessor required')
        require(len(state.values) == position and len(state.records) == position
                and [r['token'] for r in state.records] == [r['token'] for r in self.own_records],
                'Exact own predecessor token history required')
        if position == 0:
            require(not state.pooled and state.block_pos == 0
                    and np.array_equal(state.tail, np.zeros((3, 128)))
                    and np.array_equal(state.dead, np.zeros(128)),
                    'Fresh own zero indexer state required')
        self.active = {'position': position, 'token': token, 'route': route,
                       'projections': [], 'vectors': [], 'predecessor_cells': position,
                       'source_identity_sha256': self.identity,
                       'captured_inputs_or_states_used': False,
                       'projection_or_QSA_math_changed': False}
        try:
            self.active['own_HC_mixed_input'] = save_owned(
                self.output, 'p'+str(position)+'-hc-mixed.f32', mixed,
                'blk.3.hc_attn_mixed', position, self.identity, self.p.actual_source)
            detail = super().step(state, token, mixed, route, evaluate_attention)
            expected = ['indexer.k_proj.weight', 'attn_q.weight', 'attn_k.weight',
                        'attn_v.weight', 'indexer.q_proj.weight', 'attn_output.weight']
            # The inherited CPU source orders Q,K,V before indexer. Retain its
            # exact order; it is not a claim about the engine launch schedule.
            actual = [r['role'].removeprefix('blk.3.') for r in self.active['projections']]
            require(actual == ['attn_q.weight', 'attn_k.weight', 'attn_v.weight',
                               'indexer.k_proj.weight', 'indexer.q_proj.weight',
                               'attn_output.weight'] and set(actual) == set(expected),
                    'Complete original own Q/K/V/indexer/output projection roster required')
            self.active['inherited_selected_ids_owned'] = list(map(int, detail['selected_ids_owned']))
            require(self.active['inherited_selected_ids_owned'] == list(range(position+1)),
                    'Bounded own all-cell selection required')
            self.own_records.append(self.active)
            return detail
        finally:
            self.active = None

    def scope(self):
        return {'own_QKV_indexer_projection_outputs_only': True,
                'original_projection_and_QSA_math_changed': False,
                'device_norm_RoPE_decode_observed': False,
                'captured_operand_or_selection_substitution_used': False,
                'full_model_math_qualified': False}
