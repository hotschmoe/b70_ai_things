"""Bounded unseeded XPU target-only speculative sampling candidate.

Install as sglang.srt.speculative.xpu_target_sampling in the pinned source.
No kernel/ABI additions. This is not a complete sampling-parity port.
"""

def validate_target_only(sampling_info, *, tree_topk, rejection_sampling,
                         simulate_acc_len, return_logprob, request_has_seed=False):
    """Host-metadata checks only; call before mutating verify logits."""
    if sampling_info.is_all_greedy:
        return
    unsupported = []
    if tree_topk != 1:
        unsupported.append('draft tree topk must equal 1')
    if rejection_sampling:
        unsupported.append('rejection-sampling flag must be disabled')
    if sampling_info.sampling_seed is not None or request_has_seed:
        unsupported.append('seeded non-greedy sampling')
    if sampling_info.is_any_greedy:
        unsupported.append('mixed greedy/non-greedy batches')
    if sampling_info.need_min_p_sampling:
        unsupported.append('min-p sampling')
    if sampling_info.has_custom_logit_processor:
        unsupported.append('custom logit processors')
    if (sampling_info.acc_additive_penalties is not None
            or sampling_info.acc_scaling_penalties is not None
            or (sampling_info.penalizer_orchestrator is not None
                and sampling_info.penalizer_orchestrator.is_required)):
        unsupported.append('dynamic penalties (including min-new-tokens)')
    if simulate_acc_len > 0:
        unsupported.append('simulated acceptance')
    if return_logprob:
        unsupported.append('logprob reporting')
    if unsupported:
        raise ValueError('Experimental XPU target-only sampler does not support: '
                         + '; '.join(unsupported))


def sample_target_ids(logits, sampling_info, *, batch_size, verify_width):
    """Sample each target row, using the ordinary PyTorch sampler helpers.

    Input logits already contain static bias and any verify grammar mask.
    Do not call Sampler.forward: it would repeat preprocessing and expects
    request-row metadata, while verify logits contain request * width rows.
    """
    import torch
    from sglang.srt.layers.sampler import (
        sampling_from_probs_torch,
        top_k_top_p_min_p_sampling_from_probs_torch,
    )
    if logits.ndim != 2 or logits.shape[0] != batch_size * verify_width:
        raise ValueError('Unexpected target verify logits shape')
    def expand(tensor):
        if tensor.shape[0] != batch_size:
            raise ValueError('Unexpected request sampling metadata shape')
        return tensor.repeat_interleave(verify_width, dim=0)
    temperatures = expand(sampling_info.temperatures).reshape(-1, 1)
    probs = torch.softmax(logits / temperatures, dim=-1)
    if not sampling_info.need_top_k_sampling and not sampling_info.need_top_p_sampling:
        sampled = sampling_from_probs_torch(probs, sampling_seed=None, positions=None)
    else:
        sampled = top_k_top_p_min_p_sampling_from_probs_torch(
            probs, expand(sampling_info.top_ks), expand(sampling_info.top_ps),
            expand(sampling_info.min_ps), False, None, None,
        )
    return sampled.reshape(batch_size, verify_width)
