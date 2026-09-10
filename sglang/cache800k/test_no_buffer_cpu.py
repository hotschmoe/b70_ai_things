#!/usr/bin/env python3
"""CPU-only actual-source XPU cache policy and launcher delta checks."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

RAW=Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


new=load(Path(__file__).with_name('tp1_viability.py'),'new')
old=load(RAW/'sglang14ee-fp8-cache-plan-v2/lifecycle-before-no-buffer.py','old')
old.REPO=new.REPO
args=NS(out=Path('/tmp/sg-test'),image=new.IMAGE,card=1,port=18235,attention_backend='triton',prefix_cache=True)
baseline=old.command(args,'test')
assert new.command(args,'test')==baseline
args.mamba_cache_strategy='no_buffer';expected=baseline.copy()
expected[expected.index('--mamba-radix-cache-strategy')+1]='no_buffer'
expected+=['--page-size','1']
assert new.command(args,'test')==expected
src=RAW/'sglang-main-refresh/sources/sglang/python/sglang/srt/arg_groups'
ns={'Any':object,'get_platform':lambda:NS(is_xpu=True)}
for file,name in [('overrides.py','supports_mamba_cache_extra_buffer'),('mamba_hook.py','validate_mamba_no_buffer')]:
    tree=ast.parse((src/file).read_text())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src/file),'exec'),ns)
view=NS(page_size=1,disable_overlap_schedule=True,attention_backend='triton')
assert ns['supports_mamba_cache_extra_buffer'](view,'Qwen3_5ForConditionalGeneration') is False
ns['validate_mamba_no_buffer'](view,'Qwen3_5ForConditionalGeneration')
for delta in [dict(page_size=128),dict(disable_overlap_schedule=False)]:
    try:
        ns['validate_mamba_no_buffer'](NS(**(vars(view)|delta)),'Qwen3_5ForConditionalGeneration')
        raise RuntimeError('invalid config accepted')
    except AssertionError:
        pass
print('PASS default argv unchanged, no_buffer/page1-only delta, actual XPU validation and rejection cases')
