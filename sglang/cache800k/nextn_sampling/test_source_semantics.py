"""CPU-only source gates and exact target-only speculation reference.

No SGLang imports, image changes, or device initialization.
"""
import ast
from collections import defaultdict
from fractions import Fraction as F
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-main-refresh/sources/sglang')

class SourceSemantics(unittest.TestCase):
    def test_candidate_fail_closed_host_checks(self):
        from xpu_target_sampling import validate_target_only
        base = dict(is_all_greedy=False, sampling_seed=None, is_any_greedy=False,
                    need_min_p_sampling=False, has_custom_logit_processor=False,
                    acc_additive_penalties=None, acc_scaling_penalties=None,
                    penalizer_orchestrator=NS(is_required=False))
        options = dict(tree_topk=1, rejection_sampling=False, simulate_acc_len=0,
                       return_logprob=False)
        validate_target_only(NS(**base), **options)
        for field, value in [('sampling_seed', object()), ('is_any_greedy', True),
                             ('need_min_p_sampling', True),
                             ('has_custom_logit_processor', True),
                             ('acc_additive_penalties', object()),
                             ('acc_scaling_penalties', object()),
                             ('penalizer_orchestrator', NS(is_required=True))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_target_only(NS(**(base | {field:value})), **options)
        for field,value in [('tree_topk',2), ('rejection_sampling',True),
                            ('simulate_acc_len',1), ('return_logprob',True), ('request_has_seed',True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_target_only(NS(**base), **(options | {field:value}))
        # Greedy deterministic controls retain their existing behavior.
        validate_target_only(NS(**(base | {'is_all_greedy':True, 'sampling_seed':42})), **options)

    def test_actual_verifier_predicate(self):
        path = ROOT / 'python/sglang/srt/speculative/eagle_utils.py'
        module = ast.parse(path.read_text())
        function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'eagle_sample')
        gate = next(n for n in ast.walk(function) if isinstance(n, ast.If) and ast.unparse(n.test) == 'sampling_info.is_all_greedy or _is_cpu or _is_hip or _is_xpu')
        predicate = compile(ast.Expression(gate.test), str(path), 'eval')
        for xpu, greedy, expected in [(True, False, True), (True, True, True), (False, False, False)]:
            for rejection in (False, True):
                self.assertEqual(eval(predicate, {'sampling_info': NS(is_all_greedy=greedy), '_is_cpu': False, '_is_hip': False, '_is_xpu': xpu, 'speculative_use_rejection_sampling': rejection}), expected)
        self.assertIn('torch.argmax(next_token_logits', ast.unparse(gate.body[0]))

    def test_actual_rejection_startup_block_accepts_xpu(self):
        path = ROOT / 'python/sglang/srt/arg_groups/speculative_hook.py'
        tree = ast.parse(path.read_text())
        gate = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == 'cfg.speculative_use_rejection_sampling' and n.lineno > 900)
        cfg = NS(speculative_use_rejection_sampling=True, speculative_algorithm='EAGLE', speculative_eagle_topk=1, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, enable_deterministic_inference=False)
        logs = []
        scope = {'cfg':cfg, 'server_args': NS(device='xpu'), 'resolved_view': lambda _: NS(enable_multi_layer_eagle=False), 'logger':NS(info=logs.append)}
        exec(compile(ast.Module(body=[gate], type_ignores=[]), str(path), 'exec'), scope)
        self.assertTrue(logs)

    def test_argmax_is_not_temperature_sampling(self):
        # Equal finite logits at T=.7 have p=(.5,.5); argmax always token 0.
        p = [F(1, 2), F(1, 2)]
        argmax_distribution = [F(1), F(0)]
        self.assertNotEqual(p, argmax_distribution)

    def test_exact_sampled_id_walker_distribution(self):
        # Draft proposes token 0. Pre-sample root and its proposed child.
        # A mismatch emits the sampled root token; next iteration samples its
        # own conditional distribution. Enumerate all RNG outcomes exactly.
        p_root = [F(2, 5), F(3, 5)]
        p_after = [[F(1, 4), F(3, 4)], [F(4, 5), F(1, 5)]]
        speculative = defaultdict(F)
        for root, child, fallback in itertools.product(range(2), repeat=3):
            probability = p_root[root] * p_after[0][child] * p_after[1][fallback]
            sequence = (root, child if root == 0 else fallback)
            speculative[sequence] += probability
        ordinary = {(a,b): p_root[a] * p_after[a][b] for a,b in itertools.product(range(2), repeat=2)}
        self.assertEqual(dict(speculative), ordinary)
        self.assertEqual(sum(speculative.values()), 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
