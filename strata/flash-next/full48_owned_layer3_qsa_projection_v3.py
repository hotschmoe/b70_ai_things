"""Own QSA layer3 output publication on the unchanged HC V3 reference.

This is a source producer, not a runtime qualifier. Root must admit the closed
HC V3 evidence, original provider identity, device service and lifecycle first.
"""
from full48_owned_hc_device_rs_v3 import OwnedHcDeviceRsReference
from owned_layer3_qsa_projection_v3 import OwnedLayer3ProjectionTap
from owned_layer3_qsa_contract_v1 import require

ACCEPTED_IDS = (248045, 8678, 198, 15666)


class OwnedLayer3ProjectionProducer(OwnedHcDeviceRsReference):
    def __init__(self, provider, identity, args, env, source_root, bulk_build_root,
                 device_rs, operand_root, projection_root, tile_bytes=64 << 20):
        require(provider.actual_source is True
                and provider.source_identity_sha256 == identity,
                'Exact bound original provider identity required before model construction')
        self._projection_producer_started = False
        super().__init__(provider, identity, args, env, source_root, bulk_build_root,
                         device_rs, operand_root, tile_bytes)
        self.qsa[3] = OwnedLayer3ProjectionTap(provider, self.projector,
                                             projection_root, identity)

    def tokens(self, token_ids):
        ids = list(token_ids)
        require(not self._projection_producer_started and len(ids) == 4
                and all(type(value) is int for value in ids) and tuple(ids) == ACCEPTED_IDS,
                'Fresh producer and exact preregistered prefix4 IDs required before computation')
        require(self.p.actual_source is True and self.p.source_identity_sha256 == self.identity,
                'Current original provider identity changed before own computation')
        self._projection_producer_started = True
        result = super().tokens(ids)
        result['own_layer3_projection_records'] = self.qsa[3].own_records
        result['own_layer3_projection_scope'] = self.qsa[3].scope()
        result['layer3_device_control_executed'] = False
        result['full_model_math_qualified'] = False
        return result
