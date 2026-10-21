"""Generate the isolated BF16 D3 variant with compact ragged token indices."""
from pathlib import Path
import sys

src = Path(sys.argv[1]).read_text()


def change(before, after):
    global src
    assert src.count(before) == 1, before[:100]
    src = src.replace(before, after)


change('template <bool IsSpec>\nvoid launch_decode(', r'''
// Snapshot the three input history rows before any value-head group writes
// shared q/k history. Work-groups have no cross-group barrier. Only this
// small history is copied, not the recurrent FP32 matrix or the cache pages.
template <bool IsSpec> class X5HistoryCopy;
template <bool IsSpec>
void snapshot_history(sycl::queue& queue, sycl::half* dst,
    const sycl::half* conv, const int* slot, const int* accepted,
    const bool* has_initial, int nseq, int slot_stride,
    int64_t stride_slot, int64_t stride_row) {
  const size_t count = static_cast<size_t>(nseq) * 3 * kConv;
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<X5HistoryCopy<IsSpec>>(
      sycl::nd_range<1>((count + 255) / 256 * 256, 256),
      [=](sycl::nd_item<1> it) {
        size_t i = it.get_global_id(0);
        if (i >= count) return;
        int seq = static_cast<int>(i / (3 * kConv));
        int r = static_cast<int>((i / kConv) % 3);
        int c = static_cast<int>(i % kConv);
        int s = IsSpec ? slot[seq * slot_stride] : slot[seq];
        int offset = 0;
        if constexpr (IsSpec) offset = sycl::max(accepted[seq] - 1, 0);
        bool valid = s >= 0 && (IsSpec || has_initial == nullptr || has_initial[seq]);
        dst[i] = valid ? conv[static_cast<int64_t>(s) * stride_slot +
                             static_cast<int64_t>(offset + r) * stride_row + c]
                       : f2h(0.0f);
      });
  });
}

template <bool IsSpec>
void launch_decode(''')
change('    sycl::half* conv,\n    float* ssm,',
       '    sycl::half* conv,\n    const sycl::half* history,\n    float* ssm,')
change('      const bool active = IsSpec ? (slot[seq * slot_stride] >= 0) : (slot[seq] >= 0);',
       '      const bool active = (IsSpec ? (slot[seq * slot_stride] >= 0) : (slot[seq] >= 0))\n'
       '                          && cu[seq + 1] > cu[seq];')
change('        ntok = nspec;', '        base = cu[seq];\n        ntok = cu[seq + 1] - base;')
change('          global_t = token_indx[seq * nspec + t];',
       '          global_t = token_indx[base + t];')
change('''          const sycl::half* cp =
              conv + static_cast<int64_t>(conv_slot) * conv_stride_slot;''',
       '''          const sycl::half* cp = history + static_cast<int64_t>(seq) * 3 * kConv;''')
change('''                cp + static_cast<int64_t>(init_row + r) * conv_stride_row;''',
       '''                cp + static_cast<int64_t>(r) * kConv;''')
change('          const int state_len = 3 + (nspec - 1);\n          const int hist_rows = state_len - nspec;',
       '          const int state_len = 3 + (ntok - 1);\n          const int hist_rows = 2;')
change('              const int gt = token_indx[seq * nspec + draft];',
       '              const int gt = token_indx[base + draft];')
change('  const bool is_spec = token_indx.has_value();',
       '  TORCH_CHECK(cu_seqlens.size(0) == slot.size(0) + 1, "cu vs slot");\n'
       '  const bool is_spec = token_indx.has_value();')
change('  auto go = [&](auto spec_tag) {\n    launch_decode<decltype(spec_tag)::value>(',
       '''  auto history = torch::empty({nseq, 3, kConv}, conv_state.options());
  record_use(history);
  auto go = [&](auto spec_tag) {
    snapshot_history<decltype(spec_tag)::value>(queue,
        reinterpret_cast<sycl::half*>(history.data_ptr()),
        reinterpret_cast<const sycl::half*>(conv_state.data_ptr()),
        reinterpret_cast<const int*>(slot.data_ptr()),
        is_spec ? reinterpret_cast<const int*>(num_accepted->data_ptr()) : nullptr,
        has_initial.defined() ? reinterpret_cast<const bool*>(has_initial.data_ptr()) : nullptr,
        nseq, slot_stride, conv_state.stride(0), conv_state.stride(1));
    launch_decode<decltype(spec_tag)::value>(''')
change('''        reinterpret_cast<sycl::half*>(conv_state.data_ptr()),
        reinterpret_cast<float*>(ssm_state.data_ptr()),''',
       '''        reinterpret_cast<sycl::half*>(conv_state.data_ptr()),
        reinterpret_cast<const sycl::half*>(history.data_ptr()),
        reinterpret_cast<float*>(ssm_state.data_ptr()),''')
src = src.replace('#include <sycl/sycl.hpp>',
                  '#include <sycl/sycl.hpp>\n#include <sycl/ext/oneapi/bfloat16.hpp>')
src = src.replace('sycl::half', 'sycl::ext::oneapi::bfloat16').replace('at::kHalf', 'at::kBFloat16')
src = src.replace(' fp16', ' bf16')
Path(sys.argv[2]).write_text(src, encoding='ascii')
