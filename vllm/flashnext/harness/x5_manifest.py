"""Record targeted source identity without importing a GPU runtime."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
build = Path('/mnt/vm_8tb/b70/build/vllm-flashnext')
fork = build / 'vllm'
files = list((fork / 'vllm/models/qwen4_exp/gguf').glob('*.py'))
files += [fork / 'vllm/v1/worker/gpu/model_runner.py',
          fork / 'vllm/v1/worker/gpu/cudagraph_utils.py',
          fork / 'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py',
          repo / 'vllm/flashnext/harness/patches/d3_gdn.py',
          repo / 'vllm/flashnext/harness/x5_overlay/sitecustomize.py',
          repo / 'vllm/flashnext/harness/x5_pack_linear.py',
          repo / 'vllm/flashnext/harness/x5_q8_linear.py',
          repo / 'vllm/flashnext/harness/x5_quality.py',
          repo / 'vllm/flashnext/harness/x5_eventless_serve.sh',
          repo / 'vllm/flashnext/harness/patches/x5_eventless_stub.c',
          build / 'x5/eventless-lazy/libx5_eventless_stub.so',
          build / 'x5/eventless-lazy/libx5_eventless_helper.so',
          build / 'x1/eventless/libx1_eventless.so',
          build / 'x5/wikitext2-prefix.json',
          repo / 'vllm/flashnext/harness/x5_serve_proxy.sh',
          repo / 'vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.o3g.so',
          build / 'x5/gdn-bf16/libb70_gdn_xpu.so']
ident = dict(kernel=platform.release(), fork_head=subprocess.check_output(
    ['git', '-C', str(fork), 'rev-parse', 'HEAD'], text=True).strip(),
    environment={k: v for k, v in os.environ.items() if k.startswith(('B70_', 'MTP_', 'QUANT', 'DTYPE'))},
    sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()})
Path(sys.argv[1]).write_text(json.dumps(ident, indent=2) + '\n')
