"""Opt-in event spans on real scheduled prefill; one sync per profiled step."""
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time


def install():
    targets = ('vllm.v1.worker.gpu_model_runner', 'vllm.v1.worker.gpu.model_runner')
    def patch(module):
        cls = module.GPUModelRunner
        original = cls.execute_model
        if getattr(original, '_x3', False):
            return
        if os.environ.get('B70_PREFILL_ALLOC_FRACTION'):
            profile = cls.profile_run
            def limited_profile(self, *args, **kwargs):
                from b70_gguf_xpu.prefill_memory import profile_allocator
                with profile_allocator():
                    return profile(self, *args, **kwargs)
            cls.profile_run = limited_profile
        state = {'active': False, 'hooks': False, 'events': [], 'enabled': True, 'payloads': []}
        def measured(name, fn, *args, **kwargs):
            if not state['active']:
                control=os.environ.get('B70_PREFILL_PROFILE_CONTROL')
                if control:
                    with open(control) as stream:
                        flags=stream.read().split()
                    state['enabled']=bool(flags) and flags[0]=='1'
            if not state['enabled']:
                return fn(*args,**kwargs)
            remote = False
            if name in ('ep_allgather','ep_reduce_scatter'):
                sizes = kwargs.get('sizes',args[3] if len(args)>3 else None)
                remote = sizes is not None and sum(sizes) >= 512
            elif name == 'ep_compute':
                value=kwargs.get('x',args[2] if len(args)>2 else None)
                remote = value is not None and value.shape[0] >= 512
            if not state['active'] and not remote:
                return fn(*args, **kwargs)
            if name in ('ep_allgather','ep_reduce_scatter'):
                values=args[1] if len(args)>1 else kwargs.get('input_')
                values=values if isinstance(values,list) else [values]
                state['payloads'].append(dict(op=name,sizes=sizes,
                    shapes=[list(v.shape) for v in values],dtypes=[str(v.dtype) for v in values]))
            state['remote_prefill'] = state.get('remote_prefill',False) or remote
            import torch
            a,b = torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
            a.record();result=fn(*args,**kwargs);b.record()
            state['events'].append((name,a,b))
            return result
        def execute(self, scheduler_output, *args, **kwargs):
            import torch
            profile_control=os.environ.get('B70_PREFILL_PROFILE_CONTROL')
            if profile_control:
                with open(profile_control) as f:
                    mode=f.read().split()
                    state['enabled']=bool(mode) and mode[0]=='1'
                    if len(mode)>1:
                        os.environ['B70_PREFILL_EP_RS']=mode[1]
                    state['capture_enabled']=len(mode)>1 and mode[0]=='1'
            control = os.environ.get('B70_PREFILL_CONTROL')
            if control:
                with open(control) as f:
                    os.environ['B70_PREFILL'] = f.read().strip()
            if not state['hooks']:
                selected = {'linear_attn', 'self_attn', 'experts', 'shared_expert',
                            'gate', 'input_layernorm', 'post_attention_layernorm',
                            'in_proj_qkvz', 'in_proj_ba', 'out_proj', 'qkv_proj',
                            'o_proj', 'norm', 'gate_up_proj', 'down_proj', 'ple', 'embed_tokens'}
                for name, sub in self.model.named_modules():
                    if name.endswith('layers.3.self_attn') and hasattr(sub,'_project_qkv_gate'):
                        project=sub._project_qkv_gate
                        def capture(*a,_fn=project,**kw):
                            result=_fn(*a,**kw)
                            path=os.environ.get('B70_PREFILL_CAPTURE')
                            if (path and state.get('capture_enabled') and not state.get('captured')
                                    and result[0].shape[0]>=12288):
                                from pathlib import Path
                                state['captured']=True
                                q,k,v,gate=result
                                hidden=state.pop('capture_input',None)
                                positions=kw.get('positions',a[1] if len(a)>1 else None)
                                if hidden is None or positions is None:
                                    print('X3_CAPTURE_SKIPPED missing inputs',flush=True)
                                    return result
                                payload=dict(hidden=hidden,q=q,k=k,v=v,gate=gate,positions=positions)
                                payload={key:value.detach().cpu() for key,value in payload.items()}
                                Path(path).parent.mkdir(parents=True,exist_ok=True)
                                torch.save(payload,path+'.tmp')
                                Path(path+'.tmp').replace(path)
                                print('X3_CAPTURE',os.getpid(),path,tuple(q.shape),flush=True)
                            return result
                        sub._project_qkv_gate=capture
                    if sub.__class__.__name__ == 'HyperConnection':
                        for method in ('mix','combine'):
                            original_method = getattr(sub,method)
                            def call(*a,_fn=original_method,_name='hc_'+method,**kw):
                                return measured(_name,_fn,*a,**kw)
                            setattr(sub,method,call)
                    if name == '':
                        name = 'model_total'
                    elif sub.__class__.__name__ in ('Qwen4ExpDecoderLayer','Qwen3NextDecoderLayer'):
                        name += '.layer_total'
                    elif name.rsplit('.', 1)[-1] not in selected:
                        continue
                    stack = []
                    def pre(mod, inp, name=name, stack=stack):
                        if state['active']:
                            if name.endswith('layers.3.self_attn.qkv_proj') and state.get('capture_enabled') and not state.get('captured'):
                                state['capture_input']=inp[0]
                            ev = torch.xpu.Event(enable_timing=True); ev.record()
                            stack.append(ev)
                    def post(mod, inp, out, name=name, stack=stack):
                        if state['active']:
                            ev = torch.xpu.Event(enable_timing=True); ev.record()
                            state['events'].append((name, stack.pop(), ev))
                    sub.register_forward_pre_hook(pre)
                    sub.register_forward_hook(post)
                from vllm.models.qwen4_exp.gguf.moe import B70GGUFMoEMethod
                from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
                from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
                for cls,method,label in ((B70GGUFMoEMethod,'apply','ep_compute'),
                        (XpuCommunicator,'all_gatherv','ep_allgather'),
                        (XpuCommunicator,'reduce_scatterv','ep_reduce_scatter'),
                        (FlashAttentionImpl,'forward','attention_core'),
                        (FlashAttentionImpl,'do_kv_cache_update','kv_cache_write')):
                    old = getattr(cls,method)
                    def call(*a,_fn=old,_name=label,**kw):
                        return measured(_name,_fn,*a,**kw)
                    setattr(cls,method,call)
                state['hooks'] = True
            t = scheduler_output.total_num_scheduled_tokens
            state['active'] = t >= 512 and state['enabled']
            state['events'] = []
            state['payloads'] = []
            state['remote_prefill'] = False
            start = time.perf_counter()
            result = original(self, scheduler_output, *args, **kwargs)
            if state['active'] or state['remote_prefill']:
                torch.xpu.synchronize()
                spans = [{'name': name, 'ms': a.elapsed_time(b)} for name, a, b in state['events']]
                groups = {}
                for row in spans:
                    key = row['name'].rsplit('.', 1)[-1]
                    groups[key] = groups.get(key, 0) + row['ms']
                row = dict(T=t, pid=os.getpid(), remote_prefill=state['remote_prefill'],
                           wall_ms=(time.perf_counter()-start)*1000, groups=groups, spans=spans,
                           payloads=state['payloads'])
                path = os.environ['B70_PREFILL_PROFILE']
                if os.environ.get('B70_PREFILL_PROFILE_RANKED') == '1':
                    path += f'.{os.getpid()}.jsonl'
                with open(path, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print('X3_PREFILL', json.dumps({k:v for k,v in row.items() if k != 'spans'}), flush=True)
            state['active'] = False
            return result
        execute._x3 = True
        cls.execute_model = execute
    for target in targets:
        if target in sys.modules:
            patch(sys.modules[target])
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target_module=None):
            if fullname not in targets:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            old = spec.loader.exec_module
            def run(module):
                old(module); patch(module)
            spec.loader.exec_module = run
            return spec
    sys.meta_path.insert(0, Finder())
