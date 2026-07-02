# XPU MTP target-verify graph notes

Date: 2026-07-02

## Target captured, draft eager config

For target verify capture on this fork, the target worker must resolve:

- `B70_XPU_CUDAGRAPH=1`
- no global `--disable-cuda-graph`
- no `--disable-decode-cuda-graph`
- `--cuda-graph-backend-decode full`
- `--cuda-graph-backend-prefill disabled`
- `--cuda-graph-bs-decode <bs-list>` and `--cuda-graph-max-bs-decode <max-bs>`
- `--speculative-eagle-topk 1`
- `--speculative-num-draft-tokens <draft>` with `draft > 0`
- target attention backend `intel_xpu` so `sglang/patches/xpu_cudagraph.py` patches `XPUAttentionBackend`
- draft attention backend can stay `triton`, for example `--speculative-draft-attention-backend triton`

Observed resolved examples in local logs:

- captured target verify: `cuda_graph_config.decode.backend='full'`, `decode.bs=[1]`, `decode.max_bs=1`, `prefill.backend='disabled'`, `disable_cuda_graph=False`
- fully eager: `cuda_graph_config.decode.backend='disabled'`, `prefill.backend='disabled'`, `disable_cuda_graph=True`

Draft eager is controlled by `EagleDraftWorker._capture_cuda_graphs`: it returns when
`check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)` is true. In the source
snapshot inspected here, that is the same decode phase gate used by the target
`ModelRunner.init_decode_cuda_graph`, so a single global CLI decode backend of
`disabled` disables target capture too. The exact desired state is therefore
per-worker/per-phase:

- target worker: `cuda_graph_config.decode.backend='full'`
- draft worker `_capture_cuda_graphs`: `check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED) == True`

If the serve wrapper cannot set those values separately, keep `B70_XPU_MTP` unset
and add/keep a draft-capture skip override; otherwise decode `full` will enter
`_capture_cuda_graphs`, and section 3 of `xpu_cudagraph.py` is the separate opt-in
that opens XPU draft graph capture.

## Target-verify capture hazards

- `DecodeCudaGraphRunner` captures `ForwardMode.TARGET_VERIFY` when spec is on, with `num_tokens_per_bs = get_num_tokens_per_bs_for_target_verify(...)`. `build_replay_fb_view` keeps `batch_size=bs` as the sequence count and uses `num_tokens=bs*draft` only for token buffers. The XPU metadata branch must use `forward_batch.batch_size` as sequence count.
- `max_seq_len_k` is data-dependent and is assigned from `seq_lens.max().item() + draft` outside the captured region. Capture seeds `seq_lens` to `seq_len_fill_value`; replay can change the Python metadata value before graph replay, but any kernel launch geometry already baked into the graph will not change. Keep the capture fill value at the maximum length needed by the bucket.
- `custom_mask` is live in `spec_info`. `decode_cuda_graph_runner` creates graph-time `spec_info.custom_mask` from static buffers, and `eagle_worker_v2.verify` calls `update_verify_buffers_to_fill_after_draft` after draft output is available. XPU topk<=1 target verify does not consume `custom_mask`; topk>1 and draft-extend stay eager.
- MTP/EAGLE capture sets `spec_info.hidden_states` to a fresh zero tensor during graph setup. The graph stores that address. Runtime hidden-state mode must match the captured mode; `can_run_graph` rejects mismatched `capture_hidden_mode`.
- `eagle_prepare_for_verify` can rebind draft-dependent tensors such as `draft_token` and `out_cache_loc`; the verifier records streams and keeps `verify_forward_batch` alive in `GenerationBatchResult.extra_keep_alive_refs`. Do not let replay metadata point at short-lived tensors when adding more XPU graph branches.
- The XPU eager target-verify branch has topk>1 custom-mask/table construction with fresh `arange`, `cumsum`, `sort`, and `gather` tensors. This patch intentionally leaves that path eager.

## GDN and mamba wrapper notes

- `model_runner._get_attention_backend_from_str` always wraps the selected full-attention backend with `attn_backend_wrapper`. For hybrid/GDN/mamba models, `model_runner` also records attention layers that can be direct attention modules or mamba-style layers with `_forward_mamba`.
- `DecodeCudaGraphRunner` only enables graph-owned `mamba_track_indices` and `mamba_track_mask` when `server_args.enable_mamba_extra_buffer()` and `spec_algorithm.is_none()`. Under MTP target verify this is false, so those graph buffers are not allocated for the captured verify path.
- Speculative mamba state is handled outside the target graph after verification by `commit_mamba_states_after_verify(...)` in `eagle_worker_v2.verify`. The XPU stack has separate patches for XPU mamba state scatter guards.
- `MambaPool` allocates speculative state tensors up front (`SpeculativeState`, including intermediate SSM and conv-window storage). That looks capture-friendly. The risk to recheck in a live run is any wrapper path that calls `XPUAttentionBackend._init_local_attn_metadata` or translates SWA page tables inside the captured region; those paths can allocate fresh tensors or move data through CPU helpers.
