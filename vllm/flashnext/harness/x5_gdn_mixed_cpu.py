"""CPU-only checks of row and state ownership in mixed GDN metadata."""
from types import SimpleNamespace
import torch
from x5_gdn_mixed import split


def metadata(**updates):
    d = dict(num_prefills=1, num_prefill_tokens=5, num_decodes=1,
        num_decode_tokens=1, num_spec_decodes=0, num_spec_decode_tokens=0,
        num_actual_tokens=6, non_spec_query_start_loc=torch.tensor([0, 1, 6]),
        non_spec_state_indices_tensor=torch.tensor([7, 13]),
        has_initial_state=torch.tensor([True, False]),
        prefill_query_start_loc=torch.tensor([0, 5]),
        prefill_state_indices=torch.tensor([13]),
        prefill_has_initial_state=torch.tensor([False]),
        spec_query_start_loc=None, spec_state_indices_tensor=None,
        spec_token_indx=None, non_spec_token_indx=None, num_accepted_tokens=None)
    d.update(updates)
    return SimpleNamespace(**d)


md = metadata()
pr, pf, dr, dm = split(md, torch, 'cpu')
assert pr.tolist() == [1, 2, 3, 4, 5] and dr.tolist() == [0]
assert pf.non_spec_state_indices_tensor.tolist() == [13]
assert dm.non_spec_state_indices_tensor.tolist() == [7]
assert pf.non_spec_query_start_loc.tolist() == [0, 5]
assert dm.non_spec_query_start_loc.tolist() == [0, 1]
assert (md.num_decodes, md.num_actual_tokens) == (1, 6)
assert pf.spec_token_indx is None and pf.num_actual_tokens == 5
assert dm.has_initial_state is None and dm.num_actual_tokens == 1
for width in (3, 4):
    # Deliberately permuted global positions, with prefill before spec.
    md = metadata(num_decodes=0, num_decode_tokens=0, num_spec_decodes=1,
        num_spec_decode_tokens=width, num_actual_tokens=5 + width,
        spec_state_indices_tensor=torch.arange(20, 20 + width).reshape(1, width),
        spec_query_start_loc=torch.tensor([0, width]),
        spec_token_indx=torch.arange(5, 5 + width),
        non_spec_token_indx=torch.arange(5),
        non_spec_state_indices_tensor=torch.tensor([13]),
        num_accepted_tokens=torch.tensor([width]))
    pr, pf, dr, dm = split(md, torch, 'cpu')
    assert pr.tolist() == list(range(5)) and dr.tolist() == list(range(5, 5 + width))
    assert dm.spec_token_indx.tolist() == list(range(width))
    assert dm.spec_state_indices_tensor.tolist() == [list(range(20, 20 + width))]
    assert dm.num_accepted_tokens.tolist() == [width]
    assert pf.spec_state_indices_tensor is None and pf.num_spec_decodes == 0
    assert dm.non_spec_state_indices_tensor is None and dm.num_prefills == 0
    x = torch.arange((5 + width) * 2).reshape(5 + width, 2)
    out = torch.zeros_like(x)
    out.index_copy_(0, pr, x.index_select(0, pr))
    out.index_copy_(0, dr, x.index_select(0, dr))
    assert torch.equal(out, x)
    md.num_spec_decode_tokens -= 1
    assert split(md, torch, 'cpu') is None, 'ragged batch must keep safe fallback'
    md.num_actual_tokens -= 1
    md.spec_token_indx = md.spec_token_indx[:-1]
    md.spec_query_start_loc[-1] -= 1
    _, _, dr, dm = split(md, torch, 'cpu', ragged=True)
    assert dr.numel() == width - 1 and dm.spec_token_indx.numel() == width - 1
    assert dm.spec_query_start_loc.tolist() == [0, width - 1]
    one = metadata(num_prefills=1, num_prefill_tokens=1,
        num_decodes=0, num_decode_tokens=0, num_spec_decodes=1,
        num_spec_decode_tokens=width, num_actual_tokens=1 + width,
        prefill_query_start_loc=torch.tensor([0, 1]),
        prefill_has_initial_state=torch.tensor([True]),
        spec_state_indices_tensor=torch.arange(20, 20 + width).reshape(1, width),
        spec_query_start_loc=torch.tensor([0, width]),
        spec_token_indx=torch.arange(1, 1 + width),
        non_spec_token_indx=torch.tensor([0]),
        num_accepted_tokens=torch.tensor([width]))
    _, pf, _, _ = split(one, torch, 'cpu')
    assert (pf.num_prefills, pf.num_decodes) == (0, 1)
    assert pf.non_spec_state_indices_tensor.tolist() == [13]
    one.prefill_has_initial_state.zero_()
    _, pf, _, _ = split(one, torch, 'cpu')
    assert (pf.num_prefills, pf.num_decodes) == (1, 0), 'fresh one-token prefill must initialize state'
assert split(metadata(num_prefills=0), torch, 'cpu') is None
print('X5_GDN_MIXED_CPU_PASS nonSpec/spec2/spec3/permuted/ragged/no-mutation')
