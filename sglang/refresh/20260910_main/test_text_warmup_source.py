"""Execute the pinned warmup guard on CPU with no HTTP or GPU calls."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

SOURCE = Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-main-refresh/sources/sglang/python/sglang/srt/entrypoints/http_server.py')
tree = ast.parse(SOURCE.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_wait_and_warmup')
module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[]))
for skip in (False, True):
    calls = []
    def warmup(_):
        calls.append('warmup')
        return True
    state = NS(tokenizer_manager=NS(server_status=None))
    scope = dict(get_model=lambda: NS(checkpoint_engine_wait_weights_before_ready=False, delete_ckpt_after_loading=False),
        get_exec=lambda: NS(moe=NS(is_ep_scale_joiner=False)),
        get_serving=lambda: NS(skip_server_warmup=skip),
        get_observability=lambda: NS(debug_tensor_dump_input_file=None),
        _execute_server_warmup=warmup, _global_state=state, ServerStatus=NS(Up='up'),
        _freeze_gc_after_server_warmup=lambda _: calls.append('freeze'),
        logger=NS(info=lambda *a: None, debug=lambda *a: None))
    exec(compile(module, str(SOURCE), 'exec'), scope)
    scope['_wait_and_warmup'](object())
    assert calls == (['freeze'] if skip else ['warmup', 'freeze'])
    if skip:
        assert state.tokenizer_manager.server_status == 'up'
# Confirm the normal warmup owns the observed image request construction.
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_server_warmup')
assert any(isinstance(n, ast.Constant) and n.value == 'has_image_understanding' for n in ast.walk(fn))
assert any(isinstance(n, ast.Constant) and n.value == 'image_url' for n in ast.walk(fn))
assert any(isinstance(n, ast.Constant) and n.value == 'Describe the image.' for n in ast.walk(fn))
print('PASS actual warmup guard: enabled executes default warmup; skipped bypasses it and sets Up')
