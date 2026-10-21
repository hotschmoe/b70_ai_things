"""Focused stdlib fake tests: observer safety/spawn and client bounds."""
import argparse
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
import threading
import types

S=Path(__file__).resolve().parent

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,S/file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def spawn_check(root):
    os.environ['B70_D1_BRANCH_OBSERVE']='1';os.environ['B70_D1_BRANCH_DIR']=root
    sys.path.insert(0,str(S))
    runpy.run_path(str(S/'api.py'),run_name='__mp_main__')
    assert 'torch'not in sys.modules
    import branch_hook as hook
    fake=types.ModuleType('torch')
    fake.ops=types.SimpleNamespace(b70_o4=types.SimpleNamespace(w8_gemv=lambda **kw:kw['value']))
    sys.modules['torch']=fake
    pkg=Path(root)/'b70_o4_xpu';pkg.mkdir()
    (pkg/'__init__.py').write_text('_LOADED=False\ndef load():\n global _LOADED\n _LOADED=True\n')
    sys.path.insert(0,root)
    import b70_o4_xpu
    b70_o4_xpu.load()
    value=object();assert fake.ops.b70_o4.w8_gemv(value=value)is value
    assert hook._SEEN=={'b70_o4.w8_gemv'}

class Trap:
    def __str__(self):raise AssertionError('str tensor')
    def __repr__(self):raise AssertionError('repr tensor')
    def __bool__(self):raise AssertionError('bool tensor')
    @property
    def shape(self):raise AssertionError('shape tensor')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve();out.mkdir(parents=True,exist_ok=False)
    hook=module('d1_branch_cpu','branch_hook.py')
    before=list(sys.meta_path);hook.install();assert sys.meta_path==before and 'torch'not in sys.modules
    fake=types.ModuleType('torch');owner=types.SimpleNamespace();fake.ops=types.SimpleNamespace(b70_o4=owner);sys.modules['torch']=fake
    calls=[];events=[];hook._emit=lambda *a,**kw:events.append((a,kw))
    def original(*args,**kw):calls.append((args,kw));return kw['value']
    owner.w8_gemv=original;hook._observe_ops('b70_o4',('w8_gemv',))
    value=Trap();assert owner.w8_gemv(value=value)is value
    assert owner.w8_gemv is original and len(calls)==1 and len(events)==1
    # Repeated library-load observation never reinstalls hotpath wrapper.
    hook._observe_ops('b70_o4',('w8_gemv',));assert owner.w8_gemv is original
    failure=ValueError('native witness')
    def bad(*a,**kw):raise failure
    owner.fail=bad;hook._observe_ops('b70_o4',('fail',))
    try:owner.fail(value=value)
    except ValueError as e:assert e is failure
    else:raise AssertionError('native exception lost')
    assert 'b70_o4.fail'not in hook._SEEN
    def sink(*a,**kw):raise OSError('log failure')
    hook._emit=sink;owner.sink=original;hook._observe_ops('b70_o4',('sink',))
    assert owner.sink(value=value)is value
    # Two threads overlap while first native op has not yet returned.
    hook._emit=lambda *a,**kw:events.append((a,kw));barrier=threading.Barrier(2)
    def concurrent(value):barrier.wait(timeout=5);return value
    owner.concurrent=concurrent;hook._observe_ops('b70_o4',('concurrent',))
    wrapped=owner.concurrent;answers=[]
    ts=[threading.Thread(target=lambda:answers.append(wrapped(value)))for _ in range(2)]
    for t in ts:t.start()
    for t in ts:t.join()
    assert len(answers)==2 and all(a is value for a in answers)
    assert sum(e[0]==('op_return','b70_o4.concurrent')for e in events)==1
    assert owner.concurrent is concurrent
    # Capture completion marker follows native returns and preserves output.
    runner=types.ModuleType(hook._RUNNER)
    class GPUModelRunner:
        def capture_model(self,*,profile_only=False):return value
    runner.GPUModelRunner=GPUModelRunner;orig=GPUModelRunner.capture_model;hook._patch(runner)
    assert GPUModelRunner().capture_model(profile_only=True)is value
    assert GPUModelRunner().capture_model(profile_only=False)is value
    assert GPUModelRunner.capture_model is orig
    del sys.modules['torch']
    spawn=out/'spawn';spawn.mkdir()
    c=multiprocessing.get_context('spawn').Process(target=spawn_check,args=(str(spawn),));c.start();c.join(10);assert c.exitcode==0
    clients=module('d1_clients_cpu','clients.py')
    complete=out/'complete';complete.mkdir()
    commands=[('strict',[sys.executable,'-c','raise SystemExit(3)']),
              ('later',[sys.executable,'-c','print("canonical reached")'])]
    assert clients.run_phases(complete,'fake',budget=5,commands=commands)==1
    r=json.loads((complete/'clients-status.json').read_text());assert r['completed_all_phases']and r['phases']['later']['rc']==0
    bound=out/'bound';bound.mkdir()
    assert clients.run_phases(bound,'fake',budget=.3,commands=[('sleep',[sys.executable,'-c','import time;time.sleep(60)'])])==124
    r=json.loads((bound/'clients-status.json').read_text());assert r['overall_bound_expired']and not r['completed_all_phases']
    outcome=module('d1_outcome_cpu','outcome.py')
    branch=out/'branches';branch.mkdir()
    events=[dict(event='op_return',target=n,pid=77,monotonic_ns=i)for i,n in enumerate(
       ['b70_ep.moe_ep_route','b70_o4.w8_down_act','b70_o4.w8_up_collapse'],1)]
    events.append(dict(event='capture_return',profile_only=False,pid=77,monotonic_ns=10))
    (branch/'observer-77.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in events))
    assert outcome.branch_proof(branch,'routeq8f')['pass_proof']
    assert not outcome.branch_proof(branch,'control')['pass_proof']
    events[-1]['monotonic_ns']=0
    (branch/'observer-77.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in events))
    assert not outcome.branch_proof(branch,'routeq8f')['pass_proof']
    # A completed strict repeat rejection must differ from shell0 fatalengine
    # or overall-bound expiration. Minimal complete host artifact fixture.
    raw=out/'mock-run';outer=out/'mock-outer';raw.mkdir();outer.mkdir();(raw/'branches').mkdir()
    (raw/'window.log').write_text('SERVE_RC:0\nENGINE_ERROR_RC:0\nHEALTH_AFTER_RC:0\nWINDOW_DONE\n')
    (raw/'engine-errors.txt').write_text('');(raw/'health-after.txt').write_text('xpu-health: HEALTHY')
    (raw/'so-after-check.txt').write_text(''.join(str(i)+': OK\n'for i in range(4)))
    (outer/'source-after.json').write_text(json.dumps({'bad':[]}))
    (outer/'status.txt').write_text('HARNESS_RC:3 SOURCE_RC:0 PRIVATE_RC:0')
    cs={'completed_all_phases':True,'overall_bound_expired':False,
        'phases':{n:{'rc':3 if n=='gate'else 0}for n in ['gate','bench','probe','canonical']}}
    (raw/'clients-status.json').write_text(json.dumps(cs))
    (raw/'gate.json').write_text(json.dumps({'checks':{n:{'pass':n!='exact_repeat'}for n in outcome.REQUIRED_GATE}}))
    proof=[dict(event='op_return',target='b70_o4.w8_gemv',pid=77,monotonic_ns=1),
           dict(event='capture_return',profile_only=False,pid=77,monotonic_ns=2)]
    (raw/'branches/observer-77.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in proof))
    for suite in ['reasoning-v1','code-v1']:
        directory=raw/'lmx'/('no-spec-'+suite);directory.mkdir(parents=True)
        (directory/'validation.json').write_text(json.dumps({'valid':True}))
        (directory/'payload.json').write_text(json.dumps({'tokSOut':61.64}))
        for name in ['warmup','iteration-1','iteration-2','iteration-3']:
            (directory/name).mkdir();(directory/name/'sample.json').write_text('{}')
    cfg={'model_config':{'max_model_len':131072},'speculative_config':None,
         'parallel_config':{'tensor_parallel_size':1},'scheduler_config':{'max_num_batched_tokens':12288,'max_num_seqs':4},
         'cache_config':{'enable_prefix_caching':False},
         'compilation_config':{'mode':0,'cudagraph_mode':[2,0],'cudagraph_capture_sizes':[1,2,3,4,8]}}
    (raw/'lmx/server-config.json').write_text(json.dumps(cfg))
    r=outcome.classify(raw,outer,'control');assert r['operational_complete']and not r['strict_all_pass']
    (raw/'engine-errors.txt').write_text('EngineCore encountered an issue')
    assert not outcome.classify(raw,outer,'control')['operational_complete']
    (raw/'engine-errors.txt').write_text('');cs['overall_bound_expired']=True;cs['completed_all_phases']=False
    (raw/'clients-status.json').write_text(json.dumps(cs))
    assert not outcome.classify(raw,outer,'control')['operational_complete']
    result={'verdict':'CPU_FAKE_PASS; no Torch/XPU runtime qualification',
            'checks':['default_off_no_torch','kwargs_and_tensor_traps','native_return_identity_once',
              'native_exception_identity','logging_failure_fail_open','thread_overlap_restore_once',
              'capture_return_order_and_restore','real_spawn_shim_before_torch','strict_phase_does_not_skip_later',
              'overall_bound_preserved_not_stall','branch_actual_return_before_capture_samePID','control_rejects_treatment_branch','strict_reject_vs_operational_closure','fatal_shell0_and_bound_never_success']}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
