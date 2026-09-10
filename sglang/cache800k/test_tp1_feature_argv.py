"""CPU-only feature isolation checks; command construction never starts Docker."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tp1', HERE / 'tp1_viability.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
plan = json.loads((HERE.parent / 'refresh/20260910_main/tp1_baseline_plan.json').read_text())


def build(**changes):
    values = dict(out=Path('/tmp/cpu-only-argv-no-create'), card=1, port=18137,
        image=plan['command'][plan['command'].index('--image') + 1],
        attention_backend='triton', prefix_cache=False, decode_graph=False, mtp_steps=0)
    values.update(changes)
    command = module.command(SimpleNamespace(**values), 'cpu-only-no-container')
    return command[command.index('sglang.launch_server') + 1:]


def value(argv, flag):
    return argv[argv.index(flag) + 1]


baseline = build()
assert baseline == plan['server_argv'], 'Default server argv changed from reviewed baseline'
for steps in (1, 3):
    argv = build(mtp_steps=steps)
    assert value(argv, '--speculative-num-steps') == str(steps)
    assert value(argv, '--speculative-num-draft-tokens') == str(steps + 1)
    assert value(argv, '--speculative-eagle-topk') == '1'
    assert value(argv, '--speculative-draft-model-path') == value(argv, '--model-path') == '/model'
    assert value(argv, '--speculative-draft-attention-backend') == 'triton'
    assert '--disable-cuda-graph' in argv and '--disable-radix-cache' in argv
    assert argv[:len(baseline)] == baseline

graph = build(decode_graph=True)
assert '--disable-cuda-graph' not in graph
assert value(graph, '--cuda-graph-backend-decode') == 'full'
assert value(graph, '--cuda-graph-backend-prefill') == 'disabled'
assert graph[-4:] == ['--cuda-graph-bs-decode', '1', '2', '4']
assert '--speculative-algorithm' not in graph and '--disable-radix-cache' in graph
both = build(prefix_cache=True, decode_graph=True, mtp_steps=3)
assert value(both, '--mamba-radix-cache-strategy') == 'extra_buffer'
assert '--disable-radix-cache' not in both and '--disable-cuda-graph' not in both
assert value(both, '--served-model-name') == 'hotschmoe-dd'
assert value(both, '--quantization') == 'gptq' and value(both, '--tp-size') == '1'
for change in [dict(mtp_steps=2), dict(mtp_steps=1, attention_backend='intel_xpu'),
               dict(decode_graph=True, attention_backend='intel_xpu')]:
    try:
        build(**change)
    except ValueError:
        pass
    else:
        raise AssertionError('Unreviewed arm accepted: ' + str(change))
# Parser rejection happens before lease acquisition or output creation.
run = subprocess.run([sys.executable, str(HERE / 'tp1_viability.py'), '--out',
    '/tmp/cpu-only-argv-no-create', '--card', '1', '--mtp-steps', '1', '--dry-run'],
    capture_output=True, text=True)
assert run.returncode == 2 and 'require --attention-backend triton' in run.stderr
print('PASS baseline golden argv, MTP1/3 budgets, graph isolation, combination identity, invalid-arm guards')
