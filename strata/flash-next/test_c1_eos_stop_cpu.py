#!/usr/bin/env python3
"""Exercise real C1 wrapper EOS accounting and locked stop race with CPU stubs."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
import time
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c1_trace_contract import trace_end_accepted


def tracer_tests(tmp):
    settings = {'vocab_size': 100, 'special_ids': {'tokenizer.ggml.eos_token_id': 99}}
    tokens = tmp / 'tokenizer';tokens.mkdir();(tokens/'tokenizer.json').write_text(json.dumps(settings))
    import hashlib
    manifest = tmp/'identity.json';manifest.write_text(json.dumps({'tokenizer_files': {'tokenizer.json':hashlib.sha256((tokens/'tokenizer.json').read_bytes()).hexdigest()}}))
    cfg = tmp/'config.json';cfg.write_text(json.dumps({'tokenizer':str(tokens),'artifact_identity_manifest':str(manifest)}))
    trace = tmp/'trace.jsonl'
    class FakeEngine:
        def __init__(self, count=2, read=3, finish='stop', observe=True, last_token=99, drain_error=False):
            self.proc=types.SimpleNamespace(pid=123);self.last={'generated':2,'prompt_read':3,'reused':0,'prompt_tokens':3,'finish':'stop'}
            self.count,self.read,self.finish,self.observe,self.last_token,self.drain_error=count,read,finish,observe,last_token,drain_error
        def _parse_done(self, line):
            self.last={'generated':self.count,'prompt_read':self.read,'reused':0,'prompt_tokens':3,'finish':self.finish}
        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            try:
                yield 42
                yield self.last_token
            finally:
                if self.drain_error:raise RuntimeError('real reader/drain failure')
                if self.observe:self._parse_done('DONE actual-current-call')
        def close(self):pass
    class FakeService:
        def encode_prompt(self, messages, tools, kwargs):return [1,2,3]
    server=types.ModuleType('serve.server');server.StrataEngine=FakeEngine;server.Service=FakeService;server.main=lambda:0
    serve=types.ModuleType('serve');serve.server=server
    oldmods={name:sys.modules.get(name) for name in ['serve','serve.server']};oldargv=sys.argv;oldenv=os.environ.get('B70_C1_TRACE')
    sys.modules['serve']=serve;sys.modules['serve.server']=server;sys.argv=['trace','--config',str(cfg)];os.environ['B70_C1_TRACE']=str(trace)
    try:
        try:runpy.run_path(str(HERE/'c1_api_trace.py'))
        except SystemExit as e:assert e.code==0
        cases=[('normal pinned EOS',{},False,True),('stale DONE',{'observe':False},False,False),
            ('count mismatch',{'count':3},False,False),('truncated prompt',{'read':2},False,False),
            ('non-pinned stop token',{'last_token':98},False,False),('length consumer close',{'finish':'length'},False,False),
            ('cancelled EOS',{},True,False),('reader/drain error',{'drain_error':True},False,False)]
        for name,args,cancelled,accepted in cases:
            engine=FakeEngine(**args);cancel=threading.Event()
            if cancelled:cancel.set()
            gen=engine.generate([1,2,3],64,{},cancel);assert next(gen)==42;next(gen)
            try:gen.close()
            except RuntimeError:assert name=='reader/drain error'
            row=json.loads(trace.read_text().splitlines()[-1]);assert trace_end_accepted(row)==accepted,(name,row)
            if accepted:assert row['normal_eos_close'] and row['error'] is None and row['done_observation']['raw']=='DONE actual-current-call'
        # Natural exhaustion also needs a newly parsed current-call DONE.
        engine=FakeEngine();list(engine.generate([1,2,3],64,{},threading.Event()));assert trace_end_accepted(json.loads(trace.read_text().splitlines()[-1]))
    finally:
        sys.argv=oldargv
        for name,value in oldmods.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
        if oldenv is None:os.environ.pop('B70_C1_TRACE',None)
        else:os.environ['B70_C1_TRACE']=oldenv
    return 9


def stop_tests(tmp):
    spec=importlib.util.spec_from_file_location('c1',HERE/'c1_serve_controller.py');ctrl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctrl)
    directory=tmp/'run';directory.mkdir();ctrl.write(directory/'launch.json',{'container':'owned','prepared_sha256':'abc'})
    rows=[{'kind':'engine_begin','engine_pid':123},{'kind':'engine_close','engine_pid':123,'exit_code':0,'clean_exit':True,'error':None}]
    (directory/'engine-token-trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    state={'live':True,'running':True,'stops':0,'removes':0};removed=threading.Event();release=threading.Event();errors=[];results=[]
    def inspect(name):
        if not state['live']:raise subprocess.CalledProcessError(1,['docker','inspect'])
        return {'Config':{'Labels':{'b70.c1.prepared':'abc'}},'State':{'Running':state['running'],'ExitCode':0,'OOMKilled':False}}
    def command(cmd,**kw):
        if cmd[:2]==['docker','stop']:state['running']=False;state['stops']+=1
        elif cmd[:2]==['docker','rm']:
            state['live']=False;state['removes']+=1;removed.set();assert release.wait(5)
        return types.SimpleNamespace(returncode=0)
    ctrl.inspected=inspect;ctrl.run=command;ctrl.absent=lambda name:not state['live']
    def stopper():
        try:results.append(ctrl.owned_stop(directory))
        except BaseException as e:errors.append(e)
    def supervisor():
        try:results.append(ctrl.poll_owned_run(directory))
        except BaseException as e:errors.append(e)
    first=threading.Thread(target=stopper);first.start();assert removed.wait(5)
    # Container is absent now but stop.json is not published: this was the live race.
    assert not (directory/'stop.json').exists()
    second=threading.Thread(target=stopper);poll=threading.Thread(target=supervisor);second.start();poll.start();time.sleep(.1)
    assert second.is_alive() and poll.is_alive();release.set()
    for thread in [first,second,poll]:thread.join(5);assert not thread.is_alive()
    assert not errors,errors
    assert state['stops']==1 and state['removes']==1 and len(results)==3
    stop=ctrl.read(directory/'stop.json');assert stop['removed'] and stop['clean_exit'] and stop['engine_clean_exit_proven']
    assert any(r.get('stopped') for r in results)
    # Exit-zero API alone cannot hide a failed/killed native child.
    rows[-1].update(exit_code=-9,clean_exit=False);(directory/'engine-token-trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert not ctrl.native_close_proven(directory)
    (directory/'engine-token-trace.jsonl').unlink();assert not ctrl.native_close_proven(directory)
    parent_spec=importlib.util.spec_from_file_location('c1_parent',HERE/'qualify_c1_serving.py');parent=importlib.util.module_from_spec(parent_spec);parent_spec.loader.exec_module(parent)
    for code in [0,1,-15,None]:
        result={};receipt=parent.record_supervisor_exit(directory,result,code);assert receipt['passed']==(type(code) is int and code==0);assert ('supervisor_error' in result)==(code!=0)
    return 7


def main():
    with tempfile.TemporaryDirectory(prefix='c1-eos-stop-cpu-') as temp:
        root=Path(temp);trace_cases=tracer_tests(root);race_cases=stop_tests(root)
    print(json.dumps({'mode':'CPU_PROTOCOL_AND_DOCKER_STUB','trace_cases':trace_cases,'stop_race_and_native_exit_cases':race_cases,'gpu_operations':False,'passed':True},sort_keys=True))


if __name__=='__main__':main()
