"""Port of vllm-ascend PR #7148 ("Fixed speculative decoding in FULL cudagraph mode") to vLLM-XPU 0.23.0.

WHY: single-card MTP on stock v0230 = PIECEWISE 1.79x; FULL/FULL_DECODE_ONLY capture CRASHES. The MTP campaign saw
`spec_query_start_loc must have size [num_spec_decodes + 1]` in the gdn_attention op. The upstream-shared root is in
vllm/v1/cudagraph_dispatcher.py `_create_padded_batch_descriptor`, which on v0230 HARD-ASSERTS:
    if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
        num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        assert num_tokens_padded % uniform_decode_query_len == 0      # <-- fails for spec: uniform_decode_query_len = 1+num_spec
With MTP spec=4, uniform_decode_query_len = 5, and the capture sizes (1,2,4,8,...) are not multiples of 5 -> the
batch descriptor is mis-built (or asserts), feeding gdn_attention a wrong-sized spec_query_start_loc.

THE FIX (matches #7148's intent): do NOT hit the hard assert. When num_tokens_padded is not divisible by the spec
query len, fall back to non-uniform padding (uniform_decode=False) instead of crashing. This lets FULL_DECODE_ONLY +
MTP build a valid captured batch.

HOW TO USE: mount this file into the v0230 container as a sitecustomize.py (or import it before the engine starts),
e.g. `-v $PWD/88_patch_cudagraph_xpu.py:/opt/sitecustomize/sitecustomize.py -e PYTHONPATH=/opt/sitecustomize`.
Then serve with cudagraph_mode=FULL_DECODE_ONLY + MTPTOK=4 + capture sizes that INCLUDE 1+spec (e.g. 1,2,4,5,8,16,32,64).

CAVEAT (the experiment's open question): our crash was inside the XPU gdn_attention op, not this dispatcher assert
directly. If this patch alone does NOT clear the FULL_DECODE_ONLY MTP crash, the assert is KERNEL-side (in
vllm_xpu_kernels' gdn_attention) -> file a vllm-xpu-kernels issue with the repro (none exists upstream). Either
outcome localizes the bug -> progress.
"""
import os


def _install():
    try:
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor
    except Exception as e:  # pragma: no cover -- only meaningful inside the vLLM venv
        print(f"[patch_cudagraph_xpu] vLLM not importable yet ({e}); skipping")
        return

    def _create_padded_batch_descriptor(self, num_tokens, uniform_decode, has_lora, num_active_loras=0):
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]
        # #7148 port: gate on divisibility INSTEAD of asserting it -> no crash when 1+num_spec doesn't divide the size.
        if (uniform_decode
                and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
                and num_tokens_padded % uniform_decode_query_len == 0):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)
        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor
    print("[patch_cudagraph_xpu] installed #7148 port: _create_padded_batch_descriptor (assert -> graceful fallback)")


_install()
