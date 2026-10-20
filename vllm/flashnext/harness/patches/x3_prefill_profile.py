"""Opt-in event spans on real scheduled prefill; one sync per profiled step."""
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time


def install():
    target = 'vllm.v1.worker.gpu_model_runner'
    def patch(module):
        cls = module.GPUModelRunner
        original = cls.execute_model
        if getattr(original, '_x3', False):
            return
        state = {'active': False, 'hooks': False, 'events': []}
        def execute(self, scheduler_output, *args, **kwargs):
            import torch
            control = os.environ.get('B70_PREFILL_CONTROL')
            if control:
                with open(control) as f:
                    os.environ['B70_PREFILL'] = f.read().strip()
            if not state['hooks']:
                selected = {'linear_attn', 'self_attn', 'experts', 'shared_expert',
                            'gate', 'input_layernorm', 'post_attention_layernorm',
                            'in_proj_qkvz', 'in_proj_ba', 'out_proj', 'qkv_proj',
                            'o_proj', 'norm', 'gate_up_proj', 'down_proj'}
                for name, sub in self.model.named_modules():
                    if name.rsplit('.', 1)[-1] not in selected:
                        continue
                    stack = []
                    def pre(mod, inp, name=name, stack=stack):
                        if state['active']:
                            ev = torch.xpu.Event(enable_timing=True); ev.record()
                            stack.append(ev)
                    def post(mod, inp, out, name=name, stack=stack):
                        if state['active']:
                            ev = torch.xpu.Event(enable_timing=True); ev.record()
                            state['events'].append((name, stack.pop(), ev))
                    sub.register_forward_pre_hook(pre)
                    sub.register_forward_hook(post)
                state['hooks'] = True
            t = scheduler_output.total_num_scheduled_tokens
            state['active'] = t >= 512
            state['events'] = []
            start = time.perf_counter()
            result = original(self, scheduler_output, *args, **kwargs)
            if state['active']:
                torch.xpu.synchronize()
                spans = [{'name': name, 'ms': a.elapsed_time(b)} for name, a, b in state['events']]
                groups = {}
                for row in spans:
                    key = row['name'].rsplit('.', 1)[-1]
                    groups[key] = groups.get(key, 0) + row['ms']
                row = dict(T=t, wall_ms=(time.perf_counter()-start)*1000, groups=groups, spans=spans)
                with open(os.environ['B70_PREFILL_PROFILE'], 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print('X3_PREFILL', json.dumps({k:v for k,v in row.items() if k != 'spans'}), flush=True)
            state['active'] = False
            return result
        execute._x3 = True
        cls.execute_model = execute
    if target in sys.modules:
        patch(sys.modules[target]); return
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target_module=None):
            if fullname != target:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            old = spec.loader.exec_module
            def run(module):
                old(module); patch(module)
            spec.loader.exec_module = run
            return spec
    sys.meta_path.insert(0, Finder())
