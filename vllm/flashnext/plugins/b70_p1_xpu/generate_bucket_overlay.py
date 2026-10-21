"""Generate the narrow P1 bucket overlay from one frozen F1 source version."""
from pathlib import Path
import argparse
import hashlib
import json

BASE_SHA256 = 'cc0374a7a46108aaa37a4b1c86f170ab7d6e7663fe451aabcd0b208e7b800abd'
ROOT = Path('/mnt/vm_8tb/github/b70_ai_things')
BASE = ROOT / 'vllm/flashnext/plugins/b70_gguf_xpu/csrc/bucket_moe.cpp'


def generate(out: Path) -> None:
    raw = BASE.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != BASE_SHA256:
        raise RuntimeError(f'Frozen bucket source changed: expected {BASE_SHA256}, got {actual}')
    s = raw.decode('ascii')

    def replace(old: str, new: str) -> None:
        nonlocal s
        if s.count(old) != 1:
            raise RuntimeError(f'Expected exactly one source anchor: {old[:80]}')
        s = s.replace(old, new, 1)

    replace('#include "dequant_wide.hpp"', '#include <ATen/core/dispatch/Dispatcher.h>')
    replace('namespace {', '''namespace b70_p1_bucket {
// Dispatch avoids a new DT_NEEDED dependency on either loaded control or P1 SO.
using WideSignature = at::Tensor(at::Tensor, int64_t, int64_t, int64_t, at::Tensor, int64_t, int64_t);
using SelectedSignature = at::Tensor(at::Tensor, int64_t, int64_t, int64_t, at::Tensor, int64_t, int64_t, at::Tensor, int64_t);
at::Tensor dispatch_wide(at::Tensor pack, int64_t type, int64_t rows, int64_t cols,
                         at::Tensor out, int64_t mode, int64_t pair) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("b70_gguf::prefill_dequant_wide", "").typed<WideSignature>();
    return op.call(pack, type, rows, cols, out, mode, pair);
}
at::Tensor dispatch_selected(at::Tensor pack, int64_t type, int64_t rows, int64_t cols,
                             at::Tensor out, int64_t mode, int64_t pair, at::Tensor sel, int64_t expert_rows) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("b70_p1::prefill_dequant_selected", "").typed<SelectedSignature>();
    return op.call(pack, type, rows, cols, out, mode, pair, sel, expert_rows);
}
''')
    replace('int64_t tile, int64_t maxpad) {', 'int64_t tile, int64_t maxpad, bool selected_dq) {')
    replace('''        at::Tensor g_rows = at::index_select(gate2, 0, sel);
        at::Tensor u_rows = at::index_select(up2, 0, sel);
        at::Tensor d_rows = at::index_select(down2, 0, sel);''', '''        at::Tensor g_rows, u_rows, d_rows;
        if (!selected_dq) {
            g_rows = at::index_select(gate2, 0, sel);
            u_rows = at::index_select(up2, 0, sel);
            d_rows = at::index_select(down2, 0, sel);
        }''')
    replace('''        b70_prefill_dequant_wide_op(g_rows.view({-1}), type_gu, (int64_t)n * F, H, gu_flat, 1, F);
        b70_prefill_dequant_wide_op(u_rows.view({-1}), type_gu, (int64_t)n * F, H, gu_flat, 2, F);
        b70_prefill_dequant_wide_op(d_rows.view({-1}), type_down, (int64_t)n * H, F, dn_flat, 0, 0);''', '''        if (selected_dq) {
            dispatch_selected(gate, type_gu, (int64_t)n * F, H, gu_flat, 1, F, sel, F);
            dispatch_selected(up, type_gu, (int64_t)n * F, H, gu_flat, 2, F, sel, F);
            dispatch_selected(down, type_down, (int64_t)n * H, F, dn_flat, 0, 0, sel, H);
        } else {
            dispatch_wide(g_rows.view({-1}), type_gu, (int64_t)n * F, H, gu_flat, 1, F);
            dispatch_wide(u_rows.view({-1}), type_gu, (int64_t)n * F, H, gu_flat, 2, F);
            dispatch_wide(d_rows.view({-1}), type_down, (int64_t)n * H, F, dn_flat, 0, 0);
        }''')
    replace('}  // namespace', '}  // namespace b70_p1_bucket')
    marker = 'TORCH_LIBRARY_FRAGMENT(b70_gguf, m) {'
    if s.count(marker) != 1:
        raise RuntimeError('Unexpected registration block')
    s = s[:s.index(marker)] + '''TORCH_LIBRARY_FRAGMENT(b70_p1, m) {
    m.def("prefill_moe_bucketed(Tensor x, Tensor ids, Tensor w, Tensor gate, Tensor up, Tensor down, int type_gu, int type_down, int H, int F, int tile, int maxpad, bool selected_dq) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_p1, XPU, m) {
    m.impl("prefill_moe_bucketed", b70_p1_bucket::prefill_moe_bucketed_op);
}
'''
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s, encoding='ascii')
    identity = {'base': str(BASE), 'base_sha256': actual,
                'generated_sha256': hashlib.sha256(s.encode('ascii')).hexdigest(),
                'kernel_namespace': 'b70_p1_bucket', 'op': 'b70_p1::prefill_moe_bucketed'}
    out.with_suffix('.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('/mnt/vm_8tb/b70/build/vllm-flashnext/p1/bucket_overlay.cpp'))
    generate(p.parse_args().out)
