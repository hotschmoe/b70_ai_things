"""Own projection recollection and preregistered device-control boundary.

No actual operation is executed here. Root must supply current original model,
source/library/flags, owned lifecycle, health, full-four and page guards.
"""
import hashlib
from pathlib import Path
import numpy as np
from full48_owned_layer3_qsa_projection_v3 import ACCEPTED_IDS
from owned_layer3_qsa_contract_v1 import require

PROJECTIONS = ('attn_q.weight', 'attn_k.weight', 'attn_v.weight',
               'indexer.k_proj.weight', 'indexer.q_proj.weight', 'attn_output.weight')
OUTPUT_DIMS = (12288, 512, 512, 128, 512, 2560)
VECTORS = ('attn_q_norm.weight', 'attn_k_norm.weight',
           'indexer.q_norm.weight', 'indexer.k_norm.weight')
VECTOR_DIMS = (256, 256, 128, 128)


def signature(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def load_own(binding, root, shape, role, position, identity, actual):
    root = Path(root).resolve(); path = Path(binding['path'])
    require(path.is_absolute() and path.parent.resolve() == root and not path.is_symlink()
            and path.is_file(), 'Confined regular original own snapshot required')
    require(binding['shape'] == list(shape) and all(type(v) is int for v in binding['shape'])
            and type(binding['bytes']) is int and binding['encoding'] == 'LE_F32'
            and binding['role'] == role and type(binding['position']) is int
            and binding['position'] == position and binding['source_identity_sha256'] == identity
            and binding['actual_original_provider'] is actual
            and binding['captured_operand_used'] is False and binding['snapshot_is_math_input'] is False,
            'Exact own snapshot provenance/role/position required')
    before = signature(path); raw = path.read_bytes(); after = signature(path)
    require(before == after and len(raw) == binding['bytes'] == int(np.prod(shape))*4
            and hashlib.sha256(raw).hexdigest() == binding['sha256'],
            'Exact consumed own snapshot bytes/SHA/stat required')
    value = np.frombuffer(raw, dtype='<f4').reshape(shape)
    require(np.isfinite(value).all() and path.read_bytes() == raw and signature(path) == after,
            'Current finite own snapshot changed during recollection')
    return value.copy()


def recollect(records, root, identity, actual):
    require(type(actual) is bool and len(records) == 4, 'Complete explicit own projection scope required')
    outputs = []
    for position, (record, token) in enumerate(zip(records, ACCEPTED_IDS)):
        require(type(record['position']) is int and record['position'] == position
                and type(record['token']) is int and record['token'] == token
                and record['route'] == 'verifier' and type(record['predecessor_cells']) is int
                and record['predecessor_cells'] == position
                and record['source_identity_sha256'] == identity
                and record['captured_inputs_or_states_used'] is False
                and record['projection_or_QSA_math_changed'] is False
                and record['inherited_selected_ids_owned'] == list(range(position+1))
                and all(type(v) is int for v in record['inherited_selected_ids_owned']),
                'Exact sequential own original projection record required')
        require(len(record['projections']) == 6 and len(record['vectors']) == 4,
                'Exact own projection/vector roster required')
        mixed = load_own(record['own_HC_mixed_input'], root, (2560,),
                         'blk.3.hc_attn_mixed', position, identity, actual)
        own = {'position': position, 'token': token, 'own_HC_mixed': mixed,
               'projections': {}, 'vectors': {}}
        for binding, role, dim in zip(record['projections'], PROJECTIONS, OUTPUT_DIMS):
            own['projections'][role] = load_own(binding, root, (dim,), 'blk.3.'+role,
                                                position, identity, actual)
            input_dim = 6144 if role == 'attn_output.weight' else 2560
            inp = load_own(binding['own_input_snapshot'], root, (input_dim,),
                           'blk.3.'+role+'.own_input', position, identity, actual)
            require(hashlib.sha256(inp.tobytes()).hexdigest() == binding['input_sha256']
                    and binding['original_projection_arithmetic_changed'] is False
                    and binding['activation_storage'] == ('F32' if role.startswith('indexer.') else 'Q8_1')
                    and binding['weight_type'] == ('BF16' if role.startswith('indexer.') else 'Q8_0'),
                    'Exact own projection storage/input binding required')
            if role != 'attn_output.weight':
                require(inp.tobytes() == mixed.tobytes(), 'Q/K/V/indexer must consume own HC mixed only')
            else:
                own['inherited_gated_attention'] = inp.reshape(24, 256)
        for binding, role, dim in zip(record['vectors'], VECTORS, VECTOR_DIMS):
            own['vectors'][role] = load_own(binding, root, (dim,), 'blk.3.'+role,
                                           position, identity, actual)
        outputs.append(own)
    return outputs


def device_contract():
    return {'schema': 3, 'actual_device_execution_observed': False,
            'input_origin': 'independently projected original Q/K/V/indexer and own gamma/zero history',
            'captured_values_allowed_as_inputs': False, 'native_targets_comparison_only': True,
            'windows': [{'position': 0, 'rows': 2}, {'position': 2, 'rows': 1}, {'position': 3, 'rows': 1}],
            'query': {'rows_per_token': 24, 'columns': 256, 'input_stride': 512, 'rotary': 64},
            'key': {'rows_per_token': 2, 'columns': 256, 'input_stride': 256, 'rotary': 64},
            'indexer_query': {'rows_per_token': 4, 'columns': 128, 'input_stride': 128, 'rotary': 64},
            'linked_symbols': ['native_qsa_rms_norm_rope', 'native_qsa_rms_norm_weighted',
                               'native_qsa_indexer_append_steps', 'qsa_decode_attn_batch', 'native_qsa_gate_apply'],
            'outputs': ['standalone_weighted_normalized', 'fused_norm_RoPE', 'FP16_KV_pool',
                        'indexer_tail_dead_pooled_block_position', 'resident_decode_attention', 'gated_attention'],
            'standalone_norm_is_fused_internal_witness': False,
            'score_softmax_internal_witness_observed': False,
            'source_score_candidate_lowering_observed': False,
            'source_operation_variants': ['unchanged linked production functions',
                                          'separate declared CPU score candidate'],
            'required_frames': ['direct', 'graph_replay_1', 'graph_replay_2'],
            'graph_state_rule': 'restore own zero history before each complete four-token route; no cross-route carry',
            'environment_absent_including_zero': ['STRATA_ROPE_TABLE', 'STRATA_NO_NORM_ROPE',
                                                'STRATA_ATTN_LANECELL', 'STRATA_VERIFY_EAGER'],
            'RopeScaling_authority': 'locked metadata plus admitted process config; no substitute defaults',
            'native_internal_norm_argument_claimed': False,
            'device_intrinsics_or_whole_model_math_qualified': False, 'runtime_ready': False}
