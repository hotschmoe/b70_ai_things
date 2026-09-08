"""Run inside the pinned image; scheduler-only tests, no device execution."""
from types import SimpleNamespace as NS
import unittest
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import SchedulerOffloadConfig, GroupOffloadConfig
from vllm.v1.kv_cache_interface import MambaSpec
from kv_hooks import kv_offload_group_fix


class GroupFixTest(unittest.TestCase):
    def setUp(self):
        self.original = SchedulerOffloadConfig.__dict__['from_spec']

    def tearDown(self):
        SchedulerOffloadConfig.from_spec = self.original

    def invoke(self, *, explicit=False, hybrid=True, prompt_only=True):
        groups = tuple(GroupOffloadConfig(i, 832, 832, 1, None, None if i==0 else 1, is_eagle_group=True) for i in range(4))
        fixture = SchedulerOffloadConfig(groups, 1, 832, 2, prompt_only, False)
        SchedulerOffloadConfig.from_spec = classmethod(lambda cls, *_: fixture)
        kv_offload_group_fix.install()
        cfg = NS(kv_cache_groups=[NS(is_eagle_group=explicit and i==0, kv_cache_spec=object.__new__(MambaSpec) if hybrid and i else object()) for i in range(4)])
        return fixture, SchedulerOffloadConfig.from_spec(None, None, cfg)

    def test_preserves_draft_attention_margin(self):
        _, result = self.invoke()
        self.assertEqual([g.is_eagle_group for g in result.kv_group_configs], [True, False, False, False])
        self.assertFalse(result.supports_partial_tail)

    def test_preserves_explicit_annotation(self):
        original, result = self.invoke(explicit=True)
        self.assertIs(original, result)

    def test_preserves_non_hybrid(self):
        original, result = self.invoke(hybrid=False)
        self.assertIs(original, result)

    def test_rejects_decode_storage(self):
        with self.assertRaises(ValueError):
            self.invoke(prompt_only=False)


if __name__ == '__main__':
    unittest.main()
