import copy
import hashlib
import json
import unittest

from scale_plan import make_plan, NAMES, SOURCE_OPS, TARGET_LAYERS, WEIGHTS


class ScalePlanTests(unittest.TestCase):
    def setUp(self):
        self.config=json.dumps({'text_config':{'num_hidden_layers':64,'num_key_value_heads':4,
            'layer_types':['full_attention' if i in TARGET_LAYERS else 'linear_attention' for i in range(64)]}}).encode()
        self.artifact={'schema':'b70.qwen38-kv-scales.v2','weights':WEIGHTS,
            'model_config_sha256':hashlib.sha256(self.config).hexdigest(),
            'provenance':{'xpu_ops_sha256':SOURCE_OPS},'expected_layers':17,'headroom':1.1,
            'observations':{str(rank):{name:{label+'_amax':float(rank+1) for label in ['q','k','v']} for name in NAMES} for rank in [0,1]},
            'layers':{name:{label+'_scale':2*1.1/448 for label in ['q','k','v']} for name in NAMES}}
        self.inventory={f'model.layers.{i}.attn':i for i in TARGET_LAYERS}

    def plan(self,artifact=None,**kwargs):
        args=dict(role='target',tp_rank=0,tp_size=2,module_inventory=self.inventory)
        args.update(kwargs)
        return make_plan(json.dumps(artifact or self.artifact).encode(),self.config,**args)

    def test_same_merged_scale_on_tp1_and_both_tp2_ranks(self):
        plans=[self.plan(tp_size=1),self.plan(),self.plan(tp_rank=1)]
        self.assertTrue(all(p['rows']==plans[0]['rows'] for p in plans))
        self.assertEqual(len(plans[0]['rows']),16)

    def test_mtp_is_separate_role_not_target_layer_zero(self):
        p=self.plan(role='draft',module_inventory={'model.model.layers.0.attn':0})
        self.assertEqual(p['rows'][0]['source_layer'],'mtp.layers.0.self_attn.attn')
        self.assertEqual(len(p['rows']),1)

    def test_missing_mtp_scale_rejected_even_for_target(self):
        a=copy.deepcopy(self.artifact);del a['layers']['mtp.layers.0.self_attn.attn']
        with self.assertRaises(ValueError):self.plan(a)

    def test_invalid_scale_and_rank_merge_rejected(self):
        for value in [float('nan'),float('inf'),0,-1,True,1.0]:
            a=copy.deepcopy(self.artifact);a['layers'][next(iter(NAMES))]['k_scale']=value
            with self.assertRaises(ValueError):self.plan(a)

    def test_missing_rank_rejected(self):
        a=copy.deepcopy(self.artifact);del a['observations']['1']
        with self.assertRaises(ValueError):self.plan(a)

    def test_config_and_source_provenance_rejected(self):
        for field in ['model_config_sha256','provenance']:
            a=copy.deepcopy(self.artifact);a[field]='bad' if field!='provenance' else {}
            with self.assertRaises(ValueError):self.plan(a)

    def test_unsupported_backend_format_query_topology_rejected(self):
        for args in [dict(backend='intel_xpu'),dict(kv_dtype='fp8_e4m3fnuz'),dict(query_quantized=True),dict(tp_size=4),dict(tp_rank=2),dict(pp_size=2),dict(dcp_size=2)]:
            with self.assertRaises(ValueError):self.plan(**args)

    def test_incomplete_or_wrong_module_inventory_rejected(self):
        for inv in [{},dict(self.inventory,**{'model.layers.3.attn':0})]:
            with self.assertRaises(ValueError):self.plan(module_inventory=inv)


if __name__=='__main__':unittest.main()
