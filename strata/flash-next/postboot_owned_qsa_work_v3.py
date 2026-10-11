"""Root-only original projection work; native arrays remain output targets."""
import time
import numpy as np
from pathlib import Path
from postboot_owned_hc_work_v3 import OriginalWork as BaseWork
from original_tensor_rows_postboot_v1 import BoundOriginalFull48RowsPostboot as BoundOriginalFull48Rows
from explore_owned_layer0_gdn_num10_v1 import FOUNDATION
from explore_full48_original_prefix1_v4 import sha, write, require
from full48_owned_layer3_qsa_projection_v4 import OwnedLayer3ProjectionProducer
from full48_owned_layer3_qsa_projection_v3 import ACCEPTED_IDS
from owned_layer3_qsa_helper_contract_v3 import recollect, device_contract

META_KEYS = ('qwen4exp.attention.layer_norm_rms_epsilon', 'qwen4exp.rope.freq_base',
             'qwen4exp.rope.scaling.type', 'qwen4exp.rope.scaling.factor',
             'qwen4exp.rope.scaling.original_context_length')


def metadata_binding(provider):
    meta = provider.reader.files[0]['metadata']
    saved = {key: meta.get(key) for key in META_KEYS}
    eps = saved[META_KEYS[0]]
    require(eps is not None and eps['value'] == float(np.float32(1e-6)), 'Locked original QSA epsilon differs')
    for key, expected in zip(META_KEYS[1:], (1e7, 'none', 1., 262144.)):
        require(saved[key] is None or saved[key]['value'] == expected,
                'This bounded QSA helper requires unscaled original metadata '+key)
    return {'original_metadata': saved, 'effective_epsilon_F32_hex': np.float32(eps['value']).tobytes().hex(),
            'effective_RopeScaling': {'type': 'none', 'freq_base': 1e7, 'factor': 1.,
                                     'freq_scale_in': 0., 'orig_ctx': 262144., 'ext_factor': 0.,
                                     'attn_factor': 1., 'beta_fast': 32., 'beta_slow': 1.},
            'authority': 'original metadata plus pinned source defaults and no admitted CLI RoPE overrides',
            'captured_metadata_substitution_used': False}


class OriginalWork(BaseWork):
    def make_model(self, plan, env, source, device):
        provider = BoundOriginalFull48Rows(FOUNDATION, self.identity,model_association=self.model_association)
        self._models = getattr(self, '_models', [])
        model = OwnedLayer3ProjectionProducer(provider, sha(self.identity), plan['args'], env,
                                              source, self.bulk_root, device, self.output/'rs-operands',
                                              self.output/('own-qsa-phase'+str(len(self._models))))
        self._models.append((model, provider))
        return model, provider

    def __call__(self, device):
        raise ValueError('Historical postboot reader cannot produce new mathematics or runtime evidence')

    def recheck(self, report):
        result = super().recheck(report)
        own = report['own_layer3_projection']
        require(own['device_contract'] == device_contract() and own['scope'] == {
            'own_QKV_indexer_projection_outputs_only': True,
            'original_projection_and_QSA_math_changed': False,
            'device_norm_RoPE_decode_observed': False,
            'captured_operand_or_selection_substitution_used': False,
            'full_model_math_qualified': False}, 'Exact own QSA source scope changed')
        require(Path(own['root']).resolve() == (self.output/'own-qsa-phase1').resolve(),
                'Exact own prefix producer output association required')
        recollect(own['records'], own['root'], sha(self.identity), True)
        from owned_layer3_qsa_control_v4 import candidate_arrays
        candidate_arrays(own, sha(self.identity))
        provider = BoundOriginalFull48Rows(FOUNDATION, self.identity,model_association=self.model_association)
        from serial37_canonical_json_v3 import canonical
        require(canonical(metadata_binding(provider)) == canonical(own['metadata_binding']), 'Current original QSA metadata changed')
        return {**result, 'own_QKV_indexer_original_snapshot_roster_recollected': True,
                'actual_device_QSA_control_observed': False}
