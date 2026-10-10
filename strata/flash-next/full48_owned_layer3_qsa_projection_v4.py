"""Publish inherited own QSA intermediate outputs; never feed snapshots back."""
import numpy as np
from full48_owned_layer3_qsa_projection_v3 import OwnedLayer3ProjectionProducer as BaseProducer
from owned_layer3_qsa_projection_v3 import save_owned


class OwnedLayer3ProjectionProducer(BaseProducer):
    def tokens(self, token_ids):
        result = super().tokens(token_ids)
        state = result['qsa_states_owned'][3]
        self.own_candidate_outputs = []
        roles = ('q_RoPE', 'k_RoPE', 'value', 'gate', 'indexer_raw', 'iq_RoPE')
        for position, record in enumerate(state.records):
            arrays = {}
            for role, value in zip(roles, record['inputs']):
                arrays[role] = save_owned(self.qsa[3].output, 'p'+str(position)+'-candidate-'+role+'.f32',
                                         value, 'blk.3.own_candidate.'+role, position,
                                         self.identity, self.p.actual_source)
            arrays['gated'] = self.qsa[3].own_records[position]['projections'][-1]['own_input_snapshot']
            self.own_candidate_outputs.append({'position': position, 'arrays': arrays})
        self.own_candidate_final_state = {}
        for role,value in [('tail',state.tail),('dead',state.dead),('pooled',np.stack(state.pooled))]:
            self.own_candidate_final_state[role] = save_owned(
                self.qsa[3].output, 'candidate-final-'+role+'.f32', value,
                'blk.3.own_candidate.final_'+role, 3, self.identity, self.p.actual_source)
        self.own_candidate_final_state['block_pos'] = state.block_pos
        return result
