# sitecustomize.py -- NVFP4-on-XPU shim (vllm/nvfp4 experiment, 2026-07-04).
#
# vLLM v0.24.0 ships a complete ModelOpt NVFP4 load path (ModelOptNvFp4LinearMethod)
# but _POSSIBLE_NVFP4_KERNELS in vllm/model_executor/kernels/linear/__init__.py has
# NO PlatformEnum.XPU entry, so any NVFP4 checkpoint dies at engine init with
# "Failed to find a kernel that can implement the NVFP4 linear layer".
# The in-tree EmulationNvFp4LinearKernel is device-generic (LUT nibble unpack +
# float8_e4m3fn view + pure-torch math; its triton fast paths are gated behind
# current_platform.is_cuda_alike() and never run on XPU), so the minimal unlock
# is a registry entry -- plus a faster dequant-at-load variant defined here.
#
# Modes (env NVFP4_XPU_MODE):
#   emul    - stock EmulationNvFp4LinearKernel: weight dequant EVERY forward and
#             activation fake-quant to nvfp4 (true W4A4 emulation; slow; the
#             numerics reference).
#   dequant - (default) one-time NVFP4 -> BF16 weight dequant at load; forward is
#             a plain F.linear (W4A16-style: full-precision activations; fast;
#             weights cost bf16 bytes in VRAM, ~15GB for 8B).
import os
import sys

try:
    import torch
    from vllm.model_executor.kernels import linear as _linmod
    from vllm.model_executor.kernels.linear.nvfp4.emulation import (
        EmulationNvFp4LinearKernel,
    )
    from vllm.model_executor.utils import replace_parameter
    from vllm.platforms.interface import PlatformEnum

    _MODE = os.environ.get("NVFP4_XPU_MODE", "dequant")
    _F8_SCALE_M_MAX = int(os.environ.get("B70_NVFP4_F8_SCALE_M_MAX", "0"))

    class XPUDequantAtLoadNvFp4LinearKernel(EmulationNvFp4LinearKernel):
        """One-time NVFP4 -> BF16 weight dequant at load; plain F.linear after.

        At process_weights time the ModelOpt linear method has already renamed
        scales: layer.weight [N, K/2] uint8 (2 nibbles/byte), layer.weight_scale
        [N, K/16] float8_e4m3fn, layer.weight_global_scale scalar fp32.
        dequant = e2m1_lut[nibble] * e4m3(block_scale) * weight_global_scale.
        """

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
                dequantize_to_dtype,
                kE2M1ToFloat_handle,
            )

            kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(layer.weight.device)
            w = dequantize_to_dtype(
                layer.weight.data.view(torch.uint8),
                layer.weight_scale.data,
                layer.weight_global_scale.data.to(torch.float32),
                dtype=torch.bfloat16,
                block_size=16,
                swizzle=False,
            )
            replace_parameter(layer, "weight", w)
            # Free the block scales (bf16 weight no longer needs them).
            replace_parameter(
                layer,
                "weight_scale",
                torch.zeros(1, dtype=torch.float32, device=w.device),
            )

        def apply_weights(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    # E2M1 magnitude * 2 -> exact int8 codes (sign applied separately).
    _E2M1_INT8 = [0, 1, 2, 3, 4, 6, 8, 12]

    class XPUInt8XmxNvFp4LinearKernel(EmulationNvFp4LinearKernel):
        """W4A16-via-INT8-XMX: repack NVFP4 -> s8 weight + per-16-K-group bf16
        scale at load, then oneDNN int8_gemm_w8a16 (INT8 XMX) each forward.

        Needs the K-group-fixed nvfp4_kernel/_xpu_C.abi3.so mounted (the stock op
        applies square {g,g} groups and is numerically wrong for NVFP4). Weights
        stay int8 in VRAM (~half the bf16 dequant footprint).
        """

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            import vllm_xpu_kernels._xpu_C  # noqa: F401 (registers the op)

            dev = layer.weight.device
            packed = layer.weight.data  # [N, K/2] uint8
            N = packed.shape[0]
            lut = torch.tensor(_E2M1_INT8, dtype=torch.int8, device=dev)
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            both = torch.stack([lo, hi], dim=-1).reshape(N, -1)  # [N,K] low-first
            sign = torch.where((both & 0x8) != 0, -1, 1).to(torch.int8)
            w_int8 = sign * lut[(both & 0x7).long()]             # [N,K] exact s8
            wt = w_int8.t().contiguous()                          # [K,N] s8

            # scale [N, K/16] f8e4m3 -> [K/16, N] bf16, folded * ws2 / 2
            ws2 = layer.weight_global_scale.data.to(torch.float32)
            g = (layer.weight_scale.data.to(torch.float32) * ws2 / 2.0)  # [N,K/16]
            g = g.t().contiguous().to(torch.bfloat16)             # [K/16, N]

            replace_parameter(layer, "weight", wt)
            replace_parameter(layer, "weight_scale", g)

        def apply_weights(self, layer, x, bias=None):
            out_shape = (*x.shape[:-1], layer.weight.shape[1])
            x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
            y = torch.ops._xpu_C.int8_gemm_w8a16(
                x2, layer.weight, layer.weight_scale, bias
            )
            return y.reshape(out_shape)

    if _MODE == "emul":
        _kern = [EmulationNvFp4LinearKernel]
    elif _MODE == "dequant":
        _kern = [XPUDequantAtLoadNvFp4LinearKernel]
    elif _MODE == "int8xmx":
        _kern = [XPUInt8XmxNvFp4LinearKernel]
    elif _MODE == "fused":
        # 27B MLP is W4A16_NVFP4 (handled by the W4A16 fused path below); this
        # registry entry only fires for any W4A4 layer (none in this ckpt) --
        # keep a coherent emul fallback so it never crashes.
        _kern = [EmulationNvFp4LinearKernel]
    else:
        raise ValueError(
            f"NVFP4_XPU_MODE={_MODE!r} not in (emul, dequant, int8xmx, fused)"
        )

    _linmod._POSSIBLE_NVFP4_KERNELS[PlatformEnum.XPU] = _kern
    print(
        f"[nvfp4-shim] registered {_kern[0].__name__} for PlatformEnum.XPU "
        f"(NVFP4_XPU_MODE={_MODE})",
        file=sys.stderr,
        flush=True,
    )

    # ---- (1b) W4A16_NVFP4 path (MIXED_PRECISION 27B) --------------------------
    # The mixed checkpoint's MLP is W4A16_NVFP4 (weight-only 4-bit, bf16 acts),
    # routed to ModelOptNvFp4W4A16LinearMethod -- which HARDCODES a CUDA-only
    # MarlinNvFp4LinearKernel (modelopt.py:1277) instead of consulting the XPU
    # registry, so it asserts is_supported() on XPU. We replace its kernel with
    # an XPU one. To keep weights 4-bit resident (~22GB fits one card; a bf16
    # dequant-at-load would balloon the MLP to ~37GB and NOT fit), we dequant
    # per-forward (emul-class, slow but small). After the method's
    # process_weights the layer carries: weight uint8 [N,K/2], weight_scale
    # f8e4m3 [N,K/gs] block scale, weight_global_scale fp32 scalar.
    # Signed E2M1 LUT indexed by the full 4-bit nibble (bit3 = sign, bits0-2 =
    # magnitude {0,.5,1,1.5,2,3,4,6}).
    _E2M1_SIGNED = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    class _XPUW4A16NvFp4Kernel:
        """XPU W4A16_NVFP4: weights stay 4-bit-packed resident in VRAM (~22GB
        fits the 27B on one card). apply_weights dequants per-forward with a
        COMPACT, N-TILED unpack (no fat f32/int64 intermediates -> bounded
        transient, ~0.3GB), so it fits alongside the resident 4-bit weights.
        Slow (full-weight materialize per forward) but coherent + correct --
        the reference until the fused in-register kernel lands.
        """

        def process_weights_after_loading(self, layer):
            dev = layer.weight.device
            layer._lut = torch.tensor(_E2M1_SIGNED, dtype=torch.bfloat16, device=dev)
            K = layer.weight.shape[1] * 2
            layer._gs = K // layer.weight_scale.shape[1]      # group_size (16)
            # block scale -> bf16 once (fp8 [N, K/gs]); global scale folded in.
            g = layer.weight_global_scale.data.to(torch.float32)
            layer._wscale = (layer.weight_scale.data.to(torch.float32) * g).to(
                torch.bfloat16
            )
            if _MODE == "fused":
                # weights stay 4-bit resident; prep the [K/16, N] bf16 scale in the
                # op's NT layout ONCE. The weight NT view is free at apply time
                # (layer.weight is [N, K/2] uint8 -> .t() = [K/2, N] strides [1,K/2]).
                import vllm_xpu_kernels._xpu_C  # noqa: F401 (registers the op)
                layer._wscale_nt = layer._wscale.t().contiguous()   # [K/16, N] bf16
                if _F8_SCALE_M_MAX > 0:
                    if not hasattr(
                        torch.ops._xpu_C, "nvfp4_gemm_w4a16_f8scale"
                    ):
                        raise RuntimeError(
                            "B70_NVFP4_F8_SCALE_M_MAX needs an _xpu_C build "
                            "with nvfp4_gemm_w4a16_f8scale"
                        )
                    # Decode keeps checkpoint-native E4M3 block scales. The
                    # existing folded-BF16 NT copy remains for larger prefill
                    # matrices where it is faster. Delete row-major staging
                    # tensors after both dispatch layouts are ready.
                    layer._wscale_f8_nt = layer.weight_scale.data.t().contiguous()
                    layer._wglobal = (
                        layer.weight_global_scale.data.to(torch.float32)
                        .reshape(-1)
                        .contiguous()
                    )
                    if layer._wglobal.numel() != 1:
                        raise RuntimeError("NVFP4 dense weight global scale is not scalar")
                    del layer._wscale
                    del layer.weight_scale
                    del layer.weight_global_scale
                elif os.environ.get("B70_NVFP4_COMPACT_SCALES", "0") == "1":
                    # Fused inference reads only _wscale_nt. Do not retain the
                    # row-major BF16 staging copy or the now-folded checkpoint
                    # E4M3/global scale parameters. Across Qwen3.6-27B's 193
                    # FP4 linears this reclaims about 3.21 GiB/card without
                    # changing the oneDNN inputs or arithmetic.
                    del layer._wscale
                    del layer.weight_scale
                    del layer.weight_global_scale

        def apply_weights(self, layer, x, bias=None):
            if _MODE == "fused":
                # bit-exact NVFP4 weight-decompression matmul on INT4/f4_e2m1 XMX
                # path: weights read at 4-bit, dequant in the oneDNN JIT gemm.
                x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
                if _F8_SCALE_M_MAX > 0 and x2.shape[0] <= _F8_SCALE_M_MAX:
                    y = torch.ops._xpu_C.nvfp4_gemm_w4a16_f8scale(
                        x2,
                        layer.weight.data.t(),
                        bias,
                        layer._wscale_f8_nt,
                        layer._wglobal,
                        layer._gs,
                    )
                else:
                    y = torch.ops._xpu_C.nvfp4_gemm_w4a16(
                        x2, layer.weight.data.t(), bias, layer._wscale_nt, layer._gs
                    )
                return y.reshape(*x.shape[:-1], layer.weight.shape[0])
            wp = layer.weight.data          # [N, K/2] uint8
            N, Kh = wp.shape
            K = Kh * 2
            gs = layer._gs
            lut = layer._lut
            wscale = layer._wscale          # [N, K/gs] bf16 (incl global)
            x2 = x.reshape(-1, x.shape[-1])
            outs = []
            TILE = 2048
            for n0 in range(0, N, TILE):
                n1 = min(n0 + TILE, N)
                p = wp[n0:n1]                             # [t, K/2] uint8
                lo = (p & 0x0F).to(torch.long)
                hi = (p >> 4).to(torch.long)
                nib = torch.stack([lo, hi], dim=-1).reshape(n1 - n0, K)  # low-first
                w = lut[nib]                              # [t, K] bf16 (signed mag)
                s = wscale[n0:n1].repeat_interleave(gs, dim=1)           # [t, K] bf16
                w = w * s
                outs.append(torch.nn.functional.linear(x2, w))
            y = torch.cat(outs, dim=-1)
            if bias is not None:
                y = y + bias
            return y.reshape(*x.shape[:-1], N)

    # ---- (1c) register_fake for the custom op so PIECEWISE graph capture can
    # trace it (same move as the int8 ops in contrib/vllm_int8_xpu/xpu_int8.py;
    # dynamo/inductor need a FakeTensor meta impl for torch.ops._xpu_C.*).
    # Schema: nvfp4_gemm_w4a16(Tensor A, Tensor B, Tensor? bias, Tensor B_scale,
    #                          int group_size) -> Tensor
    #   A [M,K] bf16, B [K/2,N] uint8 (NT view), out [M, B.shape[1]] A-dtype.
    def _register_nvfp4_fake():
        register_fake = getattr(torch.library, "register_fake", None) or getattr(
            torch.library, "impl_abstract", None
        )
        if register_fake is None:
            return
        try:
            import vllm_xpu_kernels._xpu_C  # noqa: F401 (defines the op schema)
        except Exception:
            return
        if not hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16"):
            return

        def _fake_nvfp4_gemm(A, B, bias, B_scale, group_size):
            return A.new_empty((A.shape[0], B.shape[1]), dtype=A.dtype)

        def _fake_nvfp4_gemm_f8scale(
            A, B, bias, B_scale, B_global_scale, group_size
        ):
            return A.new_empty((A.shape[0], B.shape[1]), dtype=A.dtype)

        try:
            register_fake("_xpu_C::nvfp4_gemm_w4a16", _fake_nvfp4_gemm)
            print(
                "[nvfp4-shim] registered fake for _xpu_C::nvfp4_gemm_w4a16",
                file=sys.stderr,
                flush=True,
            )
        except (RuntimeError, ValueError) as e:
            print(
                f"[nvfp4-shim] register_fake(nvfp4_gemm_w4a16) skipped: {e}",
                file=sys.stderr,
                flush=True,
            )
        if hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16_f8scale"):
            try:
                register_fake(
                    "_xpu_C::nvfp4_gemm_w4a16_f8scale",
                    _fake_nvfp4_gemm_f8scale,
                )
                print(
                    "[nvfp4-shim] registered fake for "
                    "_xpu_C::nvfp4_gemm_w4a16_f8scale",
                    file=sys.stderr,
                    flush=True,
                )
            except (RuntimeError, ValueError) as e:
                print(
                    "[nvfp4-shim] "
                    f"register_fake(nvfp4_gemm_w4a16_f8scale) skipped: {e}",
                    file=sys.stderr,
                    flush=True,
                )

    if _MODE == "fused":
        _register_nvfp4_fake()

    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4W4A16LinearMethod as _W4A16,
    )

    def _w4a16_init_xpu(self, quant_config):
        self.quant_config = quant_config
        self.marlin_input_dtype = None
        self.kernel = _XPUW4A16NvFp4Kernel()

    _W4A16.__init__ = _w4a16_init_xpu
    if _MODE == "fused" and _F8_SCALE_M_MAX > 0:
        print(
            "[nvfp4-shim] native E4M3 block-scale decode ON "
            f"(M <= {_F8_SCALE_M_MAX}; folded BF16 scale above threshold)",
            file=sys.stderr,
            flush=True,
        )
    if (
        _MODE == "fused"
        and _F8_SCALE_M_MAX == 0
        and os.environ.get("B70_NVFP4_COMPACT_SCALES", "0") == "1"
    ):
        print(
            "[nvfp4-shim] compact fused scales ON "
            "(retain folded BF16 NT only)",
            file=sys.stderr,
            flush=True,
        )
    print(
        "[nvfp4-shim] ModelOptNvFp4W4A16LinearMethod now uses XPU 4-bit-resident "
        "dequant kernel (was CUDA-only Marlin)",
        file=sys.stderr,
        flush=True,
    )

    # ---- (2) tolerate shard_id on KV-cache scale loads ------------------------
    # qwen2.py load_weights routes k_proj.k_scale/v_proj.v_scale through the
    # stacked-params (qkv fusion) branch, which calls
    # weight_loader(param, weight, shard_id) -- but KVCacheScaleParameter's
    # loader is (param, weight) only -> TypeError on any qwen2-family ModelOpt
    # checkpoint that carries FP8 KV scales. Scales are all 1.0 in this ckpt
    # (and unused with kv_cache_dtype=auto), so dropping shard_id is lossless.
    from vllm.model_executor.layers.quantization.kv_cache import (
        KVCacheScaleParameter,
    )

    _orig_kv_loader = KVCacheScaleParameter.weight_loader

    @staticmethod
    def _kv_loader_tolerant(param, loaded_weight, *_shard_id):
        return _orig_kv_loader(param, loaded_weight)

    KVCacheScaleParameter.weight_loader = _kv_loader_tolerant
    print(
        "[nvfp4-shim] KVCacheScaleParameter.weight_loader now tolerates shard_id",
        file=sys.stderr,
        flush=True,
    )
except Exception as e:  # never break unrelated python processes in the container
    print(f"[nvfp4-shim] FAILED: {e!r}", file=sys.stderr, flush=True)


# ---- (3) capture-safe all_gather for MTP-on-TP2 (ported from the w8a8 shelf) ----------------------
# Root cause: the MTP spec-verify path calls vllm::all_gather, and oneCCL 2021.17's allgather scheduler
# has NO SYCL-graph-recordable impl -> if left captured it CRASHES capture; if ejected to eager it
# breaks vLLM's captured-piece input-address contract -> stale-read garbage at the boundary. Fix:
# reimplement all_gather as an ALL-REDUCE of a padded buffer (dist.all_reduce DOES record under
# CCL_ENABLE_SYCL_KERNELS=1). Then all_gather is recordable -> keep ALL collectives captured (eject
# nothing) -> coherent AND fully captured. Cost world_size x bytes (TP=2 = 2x), which MTP amortizes.
# SELF-CONTAINED + env-gated + world_size==1 no-op, so this is byte-identical on the single-card path.
# Toggle off with CSAG_DISABLE=1 to A/B test.
if os.environ.get("CSAG_DISABLE", "0") != "1":
    try:
        import torch
        import torch.distributed as dist
        from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator

        def _all_gather_via_allreduce(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
            if self.world_size == 1:
                return input_
            if dim < 0:
                dim += input_.dim()
            input_ = input_.contiguous()
            input_size = tuple(input_.size())
            # buf[r] holds rank r's input; everyone else's slot is zero -> all_reduce(sum) fills all slots.
            buf = torch.zeros((self.world_size,) + input_size, dtype=input_.dtype, device=input_.device)
            buf[self.rank_in_group] = input_
            # NOTE (2026-07-08): this all_gather runs EAGER (capturing=False, ~10.5M-elem gather_output head),
            # NOT inside the replayed graph -- instrumentation proved routing it through push-AR does NOT touch
            # the linear_stream.h:84 leak (the in-graph drafter collective is the row-parallel all_reduce, and
            # it is ALREADY push-AR). Kept on oneCCL dist.all_reduce (the proven capture-safe path). See
            # docs/20260707_dd_mtp_piecewise_neo_abort.md "ROOT CAUSE CORRECTED (2026-07-08)".
            dist.all_reduce(buf, group=self.device_group)            # RECORDABLE (CCL sycl kernels)
            # concat-style along `dim`, matching base_device_communicator.all_gather exactly
            out = buf.movedim(0, dim).reshape(
                input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1:]
            )
            return out

        XpuCommunicator.all_gather = _all_gather_via_allreduce
        print("[nvfp4-shim] (3) XpuCommunicator.all_gather -> capture-safe all-reduce-of-padded (MTP-on-TP2)",
              file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (3) all_gather patch failed:", repr(e), file=sys.stderr, flush=True)


# ---- (4) Tier F (EXPERIMENTAL, default OFF): bound the PIECEWISE graph command-stream accumulation ---
# The XPU graph REPLAY appends to a Level-Zero command list that is never reset, overflowing NEO after
# ~5000 replays (the W8A8-27B MTP crash). Recapture the whole graph set every N decode steps so the
# command buffer stays bounded. OFF unless B70_XPU_CG_RECYCLE_STEPS>0 (default byte-identical).
_RECYCLE_N = int(os.environ.get("B70_XPU_CG_RECYCLE_STEPS", "0") or "0")
if _RECYCLE_N > 0:
    try:
        import torch
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        _orig_cgw_call = CUDAGraphWrapper.__call__
        _cg = {"n": 0, "root": None, "recaptures": 0}

        def _recycling_call(self, *args, **kwargs):
            try:
                mode = getattr(getattr(self, "runtime_mode", None), "name", None)
                if mode == "PIECEWISE":
                    if _cg["root"] is None:
                        _cg["root"] = id(self)
                    if id(self) == _cg["root"]:          # ~once per decode step
                        _cg["n"] += 1
                        if _cg["n"] >= _RECYCLE_N:
                            _cg["n"] = 0
                            if hasattr(torch, "xpu"):
                                torch.xpu.synchronize()
                            cleared = False
                            if hasattr(CUDAGraphWrapper, "clear_all_graphs"):
                                CUDAGraphWrapper.clear_all_graphs(); cleared = True
                            elif hasattr(self, "clear_graphs"):
                                self.clear_graphs(); cleared = True
                            _cg["recaptures"] += 1
                            print(f"[cg-recycle] recapture #{_cg['recaptures']} after {_RECYCLE_N} steps "
                                  f"(clear_all={cleared})", file=sys.stderr, flush=True)
            except Exception as e:
                print("[cg-recycle] step error:", repr(e), file=sys.stderr, flush=True)
            return _orig_cgw_call(self, *args, **kwargs)

        CUDAGraphWrapper.__call__ = _recycling_call
        print(f"[cg-recycle] (4) Tier F ENABLED: recapture PIECEWISE graphs every {_RECYCLE_N} decode steps",
              file=sys.stderr, flush=True)
    except Exception as e:
        print("[cg-recycle] (4) setup failed:", repr(e), file=sys.stderr, flush=True)

# ---- (4b) DRAFTER-EAGER (EXPERIMENTAL, default OFF): the leak fix that keeps TARGET capture ----------
# ROOT CAUSE (2026-07-07, docs/20260707_dd_mtp_piecewise_neo_abort.md): at::xpu::XPUGraphImpl::replay
# submits the captured SYCL graph via submit_with_event with no sync, so per-replay NEO command-list
# entries accumulate; the MTP drafter's propose loop fires (spec-1) x pieces replays PER decode step with
# NO host sync between draft steps (llm_base_proposer.py:613-687) -- that sync-free burst is the dominant
# leak engine (per binary disasm). Per-step torch.xpu.synchronize does NOT reclaim it (tested), and
# recapture (block 4) is racy under load. THIS fix runs the DRAFTER EAGER (no graph capture/replay for the
# spec model) while the TARGET decode stays PIECEWISE-captured -> removes the drafter's sync-free replay
# burst (the leak) but keeps the target-decode capture speedup. Mechanism: SpecDecodeBaseProposer.initialize_cudagraph_keys
# derives the drafter's cudagraph mode from the main mode (PIECEWISE/FULL -> drafter PIECEWISE, else NONE);
# we force it to NONE so the drafter dispatcher only has NONE keys -> the drafter always runs eager.
# OFF unless B70_XPU_DRAFTER_EAGER=1 (default byte-identical). Mutually exclusive with blocks (4)/(5)/sync.
if os.environ.get("B70_XPU_DRAFTER_EAGER", "0") == "1":
    try:
        import vllm.v1.spec_decode.llm_base_proposer as _lbp
        _CGM = _lbp.CUDAGraphMode
        _orig_init_keys = _lbp.SpecDecodeBaseProposer.initialize_cudagraph_keys
        def _drafter_eager_keys(self, cudagraph_mode):
            # force the DRAFTER to NONE regardless of the target's mode -> drafter runs eager
            return _orig_init_keys(self, _CGM.NONE)
        _lbp.SpecDecodeBaseProposer.initialize_cudagraph_keys = _drafter_eager_keys
        print("[drafter-eager] (4b) ENABLED: MTP drafter forced to CUDAGraphMode.NONE (eager); "
              "target decode stays captured", file=sys.stderr, flush=True)
    except Exception as e:
        print("[drafter-eager] (4b) setup failed:", repr(e), file=sys.stderr, flush=True)


# ---- (5) NVFP4 MoE W4A16 via the emulation backend (Track 11f bring-up) ---------------------------
# The official 35B-A3B-NVFP4 routed experts are W4A16 (f4_e2m1 weights, group-16 fp8 scale, per-tensor
# f32 scale2, BF16 activations). vLLM's Nvfp4QuantizationEmulationTritonExperts._supports_quant_scheme
# HARD-gates on (kNvfp4Static, kNvfp4Dynamic) == W4A4 only, so it rejects our (kNvfp4Static, None) W4A16
# scheme -> "backend 'EMULATION' does not support ...". But the emulation apply() dequantizes weights to
# BF16 (using g1/g2_alphas = weight_scale_2, which W4A16 HAS) and runs stock TritonExperts with
# expects_unquantized_inputs=True -- it does NOT need dynamically-quantized activations. So the gate is
# stricter than the code: relax it to also accept weight-only (activation_key is None) as long as the
# weight is kNvfp4Static. This is the XPU bring-up path (dequant-on-the-fly, no cutlass/marlin).
# Gated on NVFP4_MOE_W4A16_EMUL=1 so the dense-27B serve is unaffected.
if os.environ.get("NVFP4_MOE_W4A16_EMUL", "0") == "1":
    try:
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
            Nvfp4QuantizationEmulationTritonExperts as _EmulMoE,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kNvfp4Static as _kStat,
            kNvfp4Dynamic as _kDyn,
        )

        @staticmethod
        def _supports_w4a16_or_w4a4(weight_key, activation_key):
            # original: (kNvfp4Static, kNvfp4Dynamic). Also accept weight-only W4A16.
            if (weight_key, activation_key) == (_kStat, _kDyn):
                return True
            return weight_key == _kStat and activation_key is None

        _EmulMoE._supports_quant_scheme = _supports_w4a16_or_w4a4
        print("[nvfp4-shim] (5) emulation MoE _supports_quant_scheme relaxed to accept W4A16 "
              "(kNvfp4Static, None) -- 35B MoE bring-up", file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (5) MoE W4A16 emul patch failed:", repr(e), file=sys.stderr, flush=True)


# ---- (6) XPU mamba align-mode pointer fix -- enables --enable-prefix-caching (ported from w8a8 shelf) --
# --enable-prefix-caching on the hybrid GDN model auto-switches vLLM's mamba KV cache to "align" mode.
# Combined with MTP spec-decode (MambaSpecDecodeGPUContext.initialize_from_forward_context) it packs raw
# device pointers into SIGNED int64 tensors (state_base_addrs[idx]=state.data_ptr(), block_table_ptrs[i]=
# bt.data_ptr()). CUDA pointers sit below 2**63; Intel XPU Level-Zero USM device addrs are >= 2**63, so
# torch's python-int -> int64 conversion overflows -> "Overflow when unpacking long long" at engine init.
# The stored int64 is only ever reinterpreted back to a pointer via .to(tl.pointer_type(...)) with modular
# int64 arithmetic, so storing the two's-complement signed value (ptr - 2**64 when ptr >= 2**63) is
# BIT-IDENTICAL to CUDA. Re-exec the method source with the two assignments wrapped. Default on; no-op for
# CUDA-range (< 2**63) pointers and for non-prefix-cache serves (align mode not active). Off with
# VLLM_XPU_MAMBA_PTR_FIX=0. The sibling uint64 batch_memcpy path is intentionally left untouched.
if os.environ.get("VLLM_XPU_MAMBA_PTR_FIX", "1") != "0":
    try:
        import inspect as _inspect, textwrap as _textwrap
        import vllm.v1.worker.mamba_utils as _mu

        def _wrap_i64(p):
            return p - (1 << 64) if p >= (1 << 63) else p

        _cls = _mu.MambaSpecDecodeGPUContext
        _src = _textwrap.dedent(_inspect.getsource(_cls.initialize_from_forward_context))
        assert "state.data_ptr()" in _src and "bt.data_ptr()" in _src, "mamba_utils source changed"
        _patched_src = (_src
                        .replace("state.data_ptr()", "_wrap_i64(state.data_ptr())")
                        .replace("bt.data_ptr()", "_wrap_i64(bt.data_ptr())"))
        _ns = dict(_mu.__dict__)
        _ns["_wrap_i64"] = _wrap_i64
        exec(_patched_src, _ns)
        _cls.initialize_from_forward_context = _ns["initialize_from_forward_context"]
        print("[nvfp4-shim] (6) MambaSpecDecodeGPUContext pointer packing wrapped for XPU USM (>=2**63) "
              "-- prefix caching unblocked", file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (6) mamba ptr fix failed:", repr(e), file=sys.stderr, flush=True)


# ---- (7) FUSED per-expert NVFP4 MoE (Track 11f real work) -----------------------------------------
# The emulation apply() (block 5) dequantizes the FULL stacked 256-expert weight to the compute dtype
# EVERY forward (a ~2 GiB fp32 transient) then runs stock TritonExperts -> 0.37 t/s. Only top_k=8 of
# 256 experts are active per token, so 248/256 of that dequant is wasted. This block replaces the
# emulation experts' apply() with a per-expert loop that keeps weights 4-bit resident and calls the
# dense 27B's oneDNN op `torch.ops._xpu_C.nvfp4_gemm_w4a16` (weights decompressed IN the JIT gemm,
# 2.85x bf16 at decode) ONCE PER ACTIVE EXPERT (gate_up then down, SiluAndMul between). No fp32
# materialize of inactive experts; the routed-expert GEMMs are the only work.
#
# Layout (per rank; TP shards N of w13 / K of w2 on group-16 boundaries):
#   w1 (w13)      [E, 2I, H/2] uint8   (rows [0:I]=gate, [I:2I]=up)   -> per expert B = w1[e].t() [H/2, 2I]
#   w2 (down)     [E, H,  I/2] uint8                                  -> per expert B = w2[e].t() [I/2, H]
#   w1_scale_val  [E, 2I, H/16] f8e4m3 ; g1_alphas [E] fp32 (weight_scale_2)
#   w2_scale_val  [E, H,  I/16] f8e4m3 ; g2_alphas [E] fp32
# The op wants B_scale as [K/16, N] bf16 folded (block_scale * global_scale), NO /2 (E2M1 float grid),
# exactly like the dense (1b) fused path -> precomputed once at first apply into self._s13/_s2.
# Gated on NVFP4_MOE_FUSED=1; needs the nvfp4_gemm_w4a16 op mounted (serve MODE=fused).
if os.environ.get("NVFP4_MOE_FUSED", "0") == "1":
    try:
        import torch
        import torch.nn.functional as _F
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
            Nvfp4QuantizationEmulationTritonExperts as _FMoE,
        )
        import vllm_xpu_kernels._xpu_C  # noqa: F401 (registers nvfp4_gemm_w4a16)

        if not hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16"):
            raise RuntimeError("nvfp4_gemm_w4a16 op not present (need MODE=fused .so mounted)")

        def _silu_and_mul(x):
            d = x.shape[-1] // 2
            return _F.silu(x[..., :d]) * x[..., d:]

        def _build_fused_moe_scales(self, w1, w2):
            # Fold block scale (f8e4m3) * per-expert global scale (fp32) -> bf16, transpose to the
            # op's [K/16, N] NT layout, ONCE. Stored as python lists of contiguous [K/16, N] bf16.
            g1 = self.quant_config.g1_alphas.reshape(-1).to(torch.float32)  # [E]
            g2 = self.quant_config.g2_alphas.reshape(-1).to(torch.float32)  # [E]
            s1 = self.w1_scale_val  # [E, 2I, H/16] f8e4m3
            s2 = self.w2_scale_val  # [E, H,  I/16] f8e4m3
            E = w1.shape[0]
            self._s13 = []
            self._s2 = []
            for e in range(E):
                a = (s1[e].to(torch.float32) * g1[e]).t().contiguous().to(torch.bfloat16)
                b = (s2[e].to(torch.float32) * g2[e]).t().contiguous().to(torch.bfloat16)
                self._s13.append(a)   # [H/16, 2I]
                self._s2.append(b)    # [I/16, H]
            # the replaced apply() no longer needs the fp8 block scales -> free them
            # (keeps net memory below the emulation path, which held fp8 + a per-forward
            # fp32 dequant of ALL experts; we hold only the folded bf16 NT scales).
            self.w1_scale_val = None
            self.w2_scale_val = None
            self._fused_ready = True

        def _fused_nvfp4_moe_apply(
            self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
            global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
            expert_tokens_meta, apply_router_weight_on_input,
        ):
            assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
            if not getattr(self, "_fused_ready", False):
                _build_fused_moe_scales(self, w1, w2)

            x = hidden_states.reshape(-1, hidden_states.shape[-1])
            xb = x.to(torch.bfloat16)
            output.zero_()
            out_flat = output.reshape(-1, output.shape[-1])

            # topk_ids hold GLOBAL expert ids; expert_map[g] = local idx (or -1) under EP. TP-only -> None.
            ids = topk_ids
            for g in torch.unique(ids).tolist():
                local = g
                if expert_map is not None:
                    local = int(expert_map[g].item())
                    if local < 0:
                        continue
                mask = ids == g                         # [T, top_k]
                tok_idx, slot_idx = mask.nonzero(as_tuple=True)
                if tok_idx.numel() == 0:
                    continue
                w_route = topk_weights[tok_idx, slot_idx].to(torch.bfloat16).unsqueeze(1)
                x_e = xb.index_select(0, tok_idx)       # [m, H]
                if apply_router_weight_on_input:
                    x_e = x_e * w_route
                gu = torch.ops._xpu_C.nvfp4_gemm_w4a16(
                    x_e, w1[local].t(), None, self._s13[local], 16
                )                                       # [m, 2I]
                h = _silu_and_mul(gu).to(torch.bfloat16)  # [m, I]
                dn = torch.ops._xpu_C.nvfp4_gemm_w4a16(
                    h, w2[local].t(), None, self._s2[local], 16
                )                                       # [m, H]
                if not apply_router_weight_on_input:
                    dn = dn * w_route
                out_flat.index_add_(0, tok_idx, dn.to(out_flat.dtype))

        _FMoE.apply = _fused_nvfp4_moe_apply
        print("[nvfp4-shim] (7) FUSED per-expert NVFP4 MoE apply installed "
              "(nvfp4_gemm_w4a16 per active expert; weights stay 4-bit resident)",
              file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (7) fused MoE patch failed:", repr(e), file=sys.stderr, flush=True)

# ---- (8) push-AR PREFILL overlay -- port the proven W8A8 posted-write all-reduce (3.8x prefill TTFT) --
# Track 11g: the NVFP4 TP=2 daily driver's one weakness is cold prefill (PP 666 vs single-card 1702), the
# oneCCL collective cost. The hand-rolled push all-reduce (L0-IPC posted write, ~11 GB/s vs oneCCL's staged
# path) is proven to cut W8A8 prefill TTFT 3.8x. It monkeypatches XpuCommunicator.all_reduce and does its OWN
# P2P (P2PACCESS-independent) so it dodges the H.13 wedge. Gated ENTIRELY on PUSH_AR_SO: unset (single-card
# shelf + default TP=2) -> this block is a no-op and behavior is byte-identical. With PUSH_AR_MIN_NUMEL set
# above the captured-decode all-reduce numel (<= max_num_seqs*hidden = 8*5120 = 40960) and below the prefill
# chunk numel (>> that), only large EAGER prefill all-reduces take the push path; captured decode all-reduces
# stay on oneCCL (graph-recordable). The .so dlopen is DEFERRED to first all_reduce inside the patch (J.15).
if os.environ.get("PUSH_AR_SO"):
    try:
        import importlib.util as _ilu
        _pa_path = os.environ.get("PUSH_AR_PATCH", "/opt/push_ar/_push_ar_patch.py")
        _spec = _ilu.spec_from_file_location("_push_ar_patch", _pa_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)  # its bottom `try: _install()` patches XpuCommunicator.all_reduce
        print("[nvfp4-shim] (8) push-AR prefill overlay loaded from", _pa_path,
              "MIN_NUMEL=" + os.environ.get("PUSH_AR_MIN_NUMEL", "0"), file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (8) push-AR overlay failed:", repr(e), file=sys.stderr, flush=True)


# ---- (9) graph-replay command-list RECLAIM -- re-instantiate the exec graph every N replays ---------
# ROOT CAUSE (2026-07-08, docs/20260707_dd_mtp_piecewise_neo_abort.md): replaying a captured XPUGraph that
# contains a cross-device collective (oneCCL OR push-AR -- transport-agnostic) accumulates L0 immediate-
# command-list space per replay -> NEO linear_stream.h:84 overflow (the MTP drafter's ~5-replay/step burst
# hits it at ~9-12k tok; TP=1/no-collective replays 300k clean). A queue drain does NOT reclaim it; only
# graph RE-INSTANTIATION does. leak_matrix_reclaim.py PROVED g.instantiate() (re-finalize the exec graph
# from the kept modifiable graph, keep_graph=True -- no re-trace) RESETS the accumulation with ZERO throughput
# cost (inst rate DEAD FLAT at re-instantiate-every-2000 vs baseline decaying to 0.23 by 30k; stream rotation
# does NOT reset). This block: force keep_graph=True on every XPUGraph + re-instantiate every N replays PER
# graph. Re-instantiation happens at the TOP of that graph's own replay() -> its previous replay has completed
# (engine re-entered forward) -> safe, never destroys an in-flight exec. Gated B70_XPU_CG_RECLAIM=N
# (0/unset = OFF -> byte-identical). This is the FULL-SPEED fix: keeps captured+MTP (~35-40 t/s) crash-free.
_RECLAIM_N = int(os.environ.get("B70_XPU_CG_RECLAIM", "0"))
if _RECLAIM_N > 0:
    try:
        import torch as _t
        import torch.xpu.graphs as _tgraphs
        _Base = _t.xpu.XPUGraph
        _orig_replay = _Base.replay
        _rc_counts = {}
        # keep_graph is consumed in __init__ (NOT just __new__) -- vLLM constructs via torch.cuda.CUDAGraph()
        # (rebound to torch.xpu.XPUGraph) with NO args, so a __new__-only override leaves keep_graph=False and
        # instantiate() refuses. Force it in BOTH __new__ and __init__ via a subclass (validated on box).
        class _XPUGraphReclaim(_Base):
            def __new__(cls, *a, **k):
                return _Base.__new__(cls, keep_graph=True)
            def __init__(self, *a, **k):
                super().__init__(keep_graph=True)
            def replay(self):
                k = id(self)
                n = _rc_counts.get(k, 0) + 1
                _rc_counts[k] = n
                if n % _RECLAIM_N == 0:
                    try:
                        self.instantiate()  # re-finalize exec graph -> reset the accumulated command list
                    except Exception as _e:
                        print("[nvfp4-shim] (9) instantiate() failed:", repr(_e), file=sys.stderr, flush=True)
                return _orig_replay(self)
        _t.xpu.XPUGraph = _XPUGraphReclaim      # the rebind (xpu_model_runner:53) picks this up
        _tgraphs.XPUGraph = _XPUGraphReclaim
        print(f"[nvfp4-shim] (9) XPUGraph RECLAIM ON (subclass): keep_graph=True + re-instantiate every "
              f"{_RECLAIM_N} replays/graph (full-speed captured+MTP leak fix)", file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (9) reclaim patch failed:", repr(e), file=sys.stderr, flush=True)


# ---- (10) fp8 KV cache offline calibration + calibrated-scale injection (Track 11h) --------------
# The ModelOpt NVFP4 checkpoint declares quantization_config.kv_cache_scheme (static fp8 KV) but ships
# NO k/v scales, so BaseKVCacheMethod falls back to scale=1.0 (kv_cache.py:148 warning). On this hybrid
# GDN model uncalibrated fp8 KV clips the model's massive-activation K/V channels (|x| can exceed the
# e4m3 max of 448) -> error that ACCUMULATES over generation -> degenerate repetition late. Fix = load
# per-full-attention-layer calibrated per-tensor DEQUANT scales (scale = amax/448; store = real/scale,
# read = fp8*scale). Both modes DEFAULT OFF (byte-identical when both env vars unset).
#
#   NVFP4_KV_CALIB_OUT=/path.json  -> wrap Attention.forward to record running max|K|,max|V| per
#       layer_name for fp8-KV (full-attention) layers; K/V here are post-RoPE/post-norm, exactly the
#       tensors written to cache. Dumps {layer_name:{k_amax,v_amax,n}} every NVFP4_KV_CALIB_EVERY
#       (default 40) forwards + atexit. Run under GRAPH=0 (the .item() sync breaks torch.compile).
#   NVFP4_KV_SCALES_FILE=/path.json (dict {layer_name:{k_scale,v_scale}}) -> after the stock
#       BaseKVCacheMethod.process_weights_after_loading runs (sets 1.0 + warns), overwrite the runtime
#       layer._k_scale/_v_scale (+ _float host mirrors) with the calibrated scalars. No checkpoint /
#       safetensors-index edit (weights stay read-only mounted).
_KV_CALIB_OUT = os.environ.get("NVFP4_KV_CALIB_OUT")
if _KV_CALIB_OUT:
    try:
        import torch as _tk
        import json as _jk
        import threading as _thk
        import atexit as _atk
        from vllm.model_executor.layers.attention.attention import Attention as _AttnCls
        _kv_calib = {}                      # layer_name -> [k_amax, v_amax, n]
        _kv_calib_lock = _thk.Lock()
        _kv_calib_every = int(os.environ.get("NVFP4_KV_CALIB_EVERY", "40"))
        _kv_calib_ctr = [0]
        _orig_attn_fwd = _AttnCls.forward

        def _kv_calib_dump():
            with _kv_calib_lock:
                snap = {k: {"k_amax": v[0], "v_amax": v[1], "n": v[2]} for k, v in _kv_calib.items()}
            tmp = _KV_CALIB_OUT + ".tmp"
            with open(tmp, "w") as f:
                _jk.dump(snap, f, indent=1, sort_keys=True)
            os.replace(tmp, _KV_CALIB_OUT)

        def _kv_calib_fwd(self, query, key, value, output_shape=None):
            try:
                if (key is not None and value is not None
                        and str(getattr(self, "kv_cache_dtype", "")).startswith("fp8")):
                    ka = float(key.detach().abs().amax().item())
                    va = float(value.detach().abs().amax().item())
                    ln = getattr(self, "layer_name", "?")
                    with _kv_calib_lock:
                        e = _kv_calib.get(ln)
                        if e is None:
                            _kv_calib[ln] = [ka, va, 1]
                        else:
                            if ka > e[0]:
                                e[0] = ka
                            if va > e[1]:
                                e[1] = va
                            e[2] += 1
                    _kv_calib_ctr[0] += 1
                    if _kv_calib_ctr[0] % _kv_calib_every == 0:
                        _kv_calib_dump()
            except Exception:
                pass
            return _orig_attn_fwd(self, query, key, value, output_shape)

        _AttnCls.forward = _kv_calib_fwd
        _atk.register(_kv_calib_dump)
        print("[nvfp4-shim] (10) KV CALIB mode ON -> %s (every %d fwds)" % (_KV_CALIB_OUT, _kv_calib_every),
              file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (10) KV calib patch failed:", repr(e), file=sys.stderr, flush=True)

_KV_SCALES_FILE = os.environ.get("NVFP4_KV_SCALES_FILE")
if _KV_SCALES_FILE:
    try:
        import json as _js
        from vllm.model_executor.layers.quantization.kv_cache import (
            BaseKVCacheMethod as _BKV,
        )
        with open(_KV_SCALES_FILE) as f:
            _kv_scales = _js.load(f)
        _orig_pwal = _BKV.process_weights_after_loading
        _kv_inj_ctr = [0]

        def _pwal_inject(self, layer):
            _orig_pwal(self, layer)
            try:
                ln = getattr(layer, "layer_name", None)
                rec = _kv_scales.get(ln) if ln is not None else None
                if rec is not None and hasattr(layer, "_k_scale"):
                    ks = float(rec["k_scale"])
                    vs = float(rec["v_scale"])
                    layer._k_scale.fill_(ks)
                    layer._v_scale.fill_(vs)
                    layer._k_scale_float = ks
                    layer._v_scale_float = vs
                    _kv_inj_ctr[0] += 1
                    print("[nvfp4-shim] (10) injected KV scales %s: k=%.5g v=%.5g" % (ln, ks, vs),
                          file=sys.stderr, flush=True)
            except Exception as ee:
                print("[nvfp4-shim] (10) inject failed for layer:", repr(ee), file=sys.stderr, flush=True)

        _BKV.process_weights_after_loading = _pwal_inject
        print("[nvfp4-shim] (10) KV SCALE INJECT mode ON -> %s (%d layers in file)"
              % (_KV_SCALES_FILE, len(_kv_scales)), file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (10) KV scale inject patch failed:", repr(e), file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# (11) Server-side default thinking-token budget = THINKCAP parity with sglang.
# vLLM 0.24.0 has SamplingParams.thinking_token_budget (forces the </think> end
# token once thinking exceeds N tokens; needs --reasoning-parser so the token ids
# are known -- we serve qwen3). But --override-generation-config only whitelists
# repetition_penalty/temperature/top_k/top_p/min_p/max_new_tokens, so the budget
# has NO server-default path. This injects a default into every chat request that
# does not set thinking_token_budget itself, bounding Qwen3 thinking-mode runaway
# loops (the omp.sh "thinks forever inside <think>, never emits code" failure --
# JOURNAL 2026-06-26). A request that sets its own thinking_token_budget wins.
# Opt-in via B70_THINK_BUDGET (tokens); empty/0 = off. Proven live: greedy+MTP+graph
# request that hit max_tokens with 0 code -> finish stop + real code once capped.
_TB = os.environ.get("B70_THINK_BUDGET", "").strip()
if _TB:
    try:
        _TB_N = int(_TB)
        if _TB_N > 0:
            from vllm.entrypoints.openai.chat_completion.protocol import (
                ChatCompletionRequest as _CCR,
            )
            _orig_to_sp = _CCR.to_sampling_params

            def _to_sp_budget(self, *a, **k):
                if getattr(self, "thinking_token_budget", None) is None:
                    try:
                        self.thinking_token_budget = _TB_N
                    except Exception:
                        pass
                return _orig_to_sp(self, *a, **k)

            _CCR.to_sampling_params = _to_sp_budget
            print("[nvfp4-shim] (11) default thinking_token_budget=%d injected "
                  "(chat requests that omit it)" % _TB_N, file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (11) thinking-budget patch failed:", repr(e), file=sys.stderr, flush=True)


# ---- (12) XPU aten::max.dim wrong on wide reductions: fix get_top_tokens (use_local_argmax_reduction) --
# The MTP drafter's per-draft-step FULL-VOCAB logits gather is the dominant slice of the AR-bound decode
# (43%). use_local_argmax_reduction shrinks it (vocab-parallel argmax, ~76000x fewer bytes) BUT on XPU the
# drafter's local argmax comes out WRONG -> accept collapses (0.65 vs ~4-5, pos0~53% rank-0 bias) -> decode
# 48.9 -> 25.7. ROOT CAUSE (research/profiling/localargmax_accept_rootcause.md): get_top_tokens uses
# torch.max(dim=-1) at logits_processor.py:136, and aten::max.dim returns a WRONG per-shard max VALUE over
# the ~124160-wide bf16 vocab shard on this box, while aten::argmax (used by the healthy full path) is correct.
# FIX: derive both the per-shard value + index from argmax+gather instead of max(dim=-1) (keeps the O(2*tp)
# comm win; byte-identical on TP=1). Default OFF (LOCALARGMAX_ARGMAX_FIX=1). LOCALARGMAX_VERIFY=1 counts
# argmax!=max.dim disagreements to prove the op bug empirically.
if os.environ.get("LOCALARGMAX_ARGMAX_FIX", "0") == "1":
    try:
        import torch
        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_gather,
        )
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        _LAR_VERIFY = os.environ.get("LOCALARGMAX_VERIFY", "0") == "1"
        _lar_n = [0]

        def _get_top_tokens_argmax(self, lm_head, hidden_states, embedding_bias=None):
            if self.scale <= 0.0 and self.scale != 1.0:
                raise ValueError("local argmax reduction needs positive logit scale")
            tp_size = get_tensor_model_parallel_world_size()
            logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                logits = logits * self.scale
            num_pad = lm_head.shard_indices.num_org_vocab_padding
            if num_pad > 0:
                logits[..., -num_pad:] = -float("inf")
            # FIX: argmax (correct on XPU) instead of max(dim=-1) (wrong value/idx on wide bf16 shard).
            local_max_indices = logits.argmax(dim=-1)
            local_max_vals = logits.gather(-1, local_max_indices.unsqueeze(-1)).squeeze(-1)
            if _LAR_VERIFY:
                _bad_v, _bad_i = logits.max(dim=-1)
                _nmis = int((_bad_i != local_max_indices).sum().item())
                _lar_n[0] += 1
                if _nmis or _lar_n[0] <= 8:
                    print("[localargmax-verify] call=%d argmax!=max.dim mismatches=%d/%d"
                          % (_lar_n[0], _nmis, local_max_indices.numel()), file=sys.stderr, flush=True)
            vocab_start = lm_head.shard_indices.org_vocab_start_index
            global_indices = local_max_indices + vocab_start
            if tp_size == 1:
                return global_indices
            local_pair = torch.stack([local_max_vals.float(), global_indices.float()], dim=-1)
            gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
            gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
            max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
            top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
            return top_tokens.squeeze(-1).to(torch.int64)

        LogitsProcessor.get_top_tokens = _get_top_tokens_argmax
        print("[nvfp4-shim] (12) get_top_tokens -> argmax-based (XPU max.dim fix)", file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (12) localargmax fix failed:", repr(e), file=sys.stderr, flush=True)

# ---- (13) embed_tokens INT8-at-load (per-row symmetric) -- frees ~1.18 GiB of bf16 for KV ----------
# The NVFP4 ckpt quantizes everything big EXCEPT embed_tokens (2.37 GiB BF16 -- GDN is already fp8,
# lm_head already u8). Storing the embedding int8-per-row (amax/127 scale, rel err ~0.4% of row max)
# and dequantizing at gather costs one cheap index+mul and frees ~1.18 GiB -- the difference between
# MTP@48k and MTP@90-100k at the same UTIL (JOURNAL 2026-07-21 i/j). Opt-in: B70_EMBED_INT8=1.
# Mechanics: wrap Worker.determine_available_memory -> quantize EVERY VocabParallelEmbedding found via
# gc (catches the MTP drafter's shared-weight instance; dedup by weight data_ptr) BEFORE the profiling
# run + memory measurement, pop the bf16 Parameter, empty_cache so the freed bytes reach the KV budget.
# Forward fast path is TP=1-only (the DP replica case); TP>1 modules are left untouched (masking path).
# Capture-safe: gather + to() + mul, no host sync; quantization completes before any graph capture.
if os.environ.get("B70_EMBED_INT8", "0") == "1":
    try:
        import torch
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

        _emb_orig_fwd = VocabParallelEmbedding.forward
        _emb_reg = {}  # weight data_ptr -> (i8, scale)

        def _emb_fwd(self, input_):
            i8 = getattr(self, "_b70_emb_i8", None)
            if i8 is None:
                return _emb_orig_fwd(self, input_)
            w = i8[input_]
            return w.to(self._b70_emb_dtype) * self._b70_emb_scale[input_]

        def _emb_quantize_all():
            import gc
            n_q, n_shared, freed = 0, 0, 0
            mods = [o for o in gc.get_objects() if isinstance(o, VocabParallelEmbedding)]
            for m in mods:
                w = getattr(m, "weight", None)
                if w is None or not isinstance(w, torch.Tensor):
                    continue
                if getattr(m, "tp_size", 1) != 1 or w.dtype not in (torch.bfloat16, torch.float16):
                    continue  # TP>1 masking path / already non-float: leave stock
                key = w.data_ptr()
                if key in _emb_reg:
                    i8, sc = _emb_reg[key]
                    n_shared += 1
                else:
                    # CHUNKED quantize: the whole-tensor fp32 path allocates ~6 GiB of transients and
                    # OOMs a loaded card (observed UR_OUT_OF_RESOURCES); 8192-row blocks peak <200 MB.
                    V = w.shape[0]
                    CH = 8192
                    amax = torch.empty((V, 1), dtype=torch.float32, device=w.device)
                    for r0 in range(0, V, CH):
                        amax[r0:r0 + CH] = w[r0:r0 + CH].abs().amax(dim=1, keepdim=True).float()
                    amax.clamp_(min=1e-8)
                    sc = (amax / 127.0).to(w.dtype)
                    i8 = torch.empty(w.shape, dtype=torch.int8, device=w.device)
                    for r0 in range(0, V, CH):
                        blk = w[r0:r0 + CH].float().div_(amax[r0:r0 + CH]).mul_(127.0)
                        i8[r0:r0 + CH] = blk.round_().clamp_(-127, 127).to(torch.int8)
                        del blk
                    _emb_reg[key] = (i8, sc)
                    n_q += 1
                    freed += w.numel() * w.element_size() - i8.numel() - sc.numel() * sc.element_size()
                m._b70_emb_i8 = i8
                m._b70_emb_scale = sc
                m._b70_emb_dtype = w.dtype
                m._parameters.pop("weight", None)
                m.weight = None
            if n_q:
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            print("[nvfp4-shim] (13) embed INT8: quantized %d embedding(s) (+%d shared ref), freed ~%.2f GiB"
                  % (n_q, n_shared, freed / 2**30), file=sys.stderr, flush=True)
            return freed

        VocabParallelEmbedding.forward = _emb_fwd

        from vllm.v1.worker.gpu_worker import Worker as _EmbWorker
        _emb_orig_dam = _EmbWorker.determine_available_memory

        def _emb_dam(self, *a, **kw):
            try:
                _freed = _emb_quantize_all()
                # vLLM's KV budget uses weights_memory=model_memory_usage RECORDED AT LOAD -- a
                # post-load free is invisible to it. Subtract what we freed so the budget sees it.
                if _freed:
                    self.model_runner.model_memory_usage -= _freed
                    print("[nvfp4-shim] (13) model_memory_usage adjusted -%.2f GiB for KV budget"
                          % (_freed / 2**30), file=sys.stderr, flush=True)
            except Exception as e:
                print("[nvfp4-shim] (13) embed INT8 quantize failed:", repr(e), file=sys.stderr, flush=True)
            return _emb_orig_dam(self, *a, **kw)

        _EmbWorker.determine_available_memory = _emb_dam
        print("[nvfp4-shim] (13) embed INT8-at-load ARMED (quantizes at determine_available_memory)",
              file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (13) embed INT8 install failed:", repr(e), file=sys.stderr, flush=True)


# ---- (14a) prefix-cache DEBUG instrumentation (MTP x fp8-KV hits=0 root-cause) ---------------------
# Logs (a) the kv-cache group table at coordinator init (spec class, dtype, block_size, page bytes,
# layer count, eagle bit) and (b) every per-manager find_longest_cache_hit call (manager, block_size,
# max_length in, drop_eagle, tokens out) plus the coordinator's final intersected hit. Runs in the
# EngineCore process -> docker logs. Diagnostic only, default OFF (B70_PC_DEBUG=1).
if os.environ.get("B70_PC_DEBUG", "0") == "1":
    try:
        import vllm.v1.core.kv_cache_coordinator as _kvc
        import vllm.v1.core.single_type_kv_cache_manager as _stm

        def _spec_str(spec):
            return "%s(block=%s dtype=%s page=%s)" % (
                type(spec).__name__, getattr(spec, "block_size", "?"),
                getattr(spec, "dtype", "?"), getattr(spec, "page_size_bytes", "?"))

        _orig_split = _kvc.HybridKVCacheCoordinator.verify_and_split_kv_cache_groups

        def _split_logged(self):
            _orig_split(self)
            for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
                print("[pc-debug] kv group %d: %s layers=%d %s" % (
                    i, _spec_str(g.kv_cache_spec), len(g.layer_names),
                    g.layer_names[:2] + (["..."] if len(g.layer_names) > 2 else [])),
                    file=sys.stderr, flush=True)
            for j, ag in enumerate(self.attention_groups):
                print("[pc-debug] SpecGroup %d: mgr=%s group_ids=%s use_eagle=%s %s" % (
                    j, ag.manager_cls.__name__, ag.group_ids, ag.use_eagle,
                    _spec_str(ag.spec)), file=sys.stderr, flush=True)
            print("[pc-debug] scheduler_block_size=%s hash_block_size=%s eagle_group_ids=%s" % (
                getattr(self, "scheduler_block_size", "?"), getattr(self, "hash_block_size", "?"),
                getattr(self, "eagle_group_ids", "?")), file=sys.stderr, flush=True)

        _kvc.HybridKVCacheCoordinator.verify_and_split_kv_cache_groups = _split_logged

        for _mgr in (_stm.FullAttentionManager, _stm.SlidingWindowManager,
                     _stm.ChunkedLocalAttentionManager, _stm.MambaManager,
                     _stm.CrossAttentionManager):
            if "find_longest_cache_hit" not in vars(_mgr):
                continue
            _orig_flch = vars(_mgr)["find_longest_cache_hit"].__func__

            def _flch_logged(cls, block_hashes, max_length, kv_cache_group_ids,
                             block_pool, kv_cache_spec, drop_eagle_block,
                             alignment_tokens, _o=_orig_flch, **kw):
                out = _o(cls, block_hashes, max_length, kv_cache_group_ids,
                         block_pool, kv_cache_spec, drop_eagle_block,
                         alignment_tokens, **kw)
                print("[pc-debug]   %s gids=%s block=%d max_len=%d eagle_drop=%s align=%d -> %d blk = %d tok" % (
                    cls.__name__, kv_cache_group_ids, kv_cache_spec.block_size,
                    max_length, drop_eagle_block, alignment_tokens,
                    len(out[0]), len(out[0]) * kv_cache_spec.block_size),
                    file=sys.stderr, flush=True)
                return out

            setattr(_mgr, "find_longest_cache_hit", classmethod(_flch_logged))

        _orig_hy_flch = _kvc.HybridKVCacheCoordinator.find_longest_cache_hit

        def _hy_flch_logged(self, block_hashes, max_cache_hit_length):
            print("[pc-debug] hybrid lookup: max_cache_hit_length=%d" % max_cache_hit_length,
                  file=sys.stderr, flush=True)
            blocks, hit = _orig_hy_flch(self, block_hashes, max_cache_hit_length)
            print("[pc-debug] hybrid RESULT: hit=%d tok; per-group blks=%s" % (
                hit, [len(b) for b in blocks]), file=sys.stderr, flush=True)
            return blocks, hit

        _kvc.HybridKVCacheCoordinator.find_longest_cache_hit = _hy_flch_logged
        print("[nvfp4-shim] (14a) prefix-cache DEBUG instrumentation ON", file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (14a) pc-debug install failed:", repr(e), file=sys.stderr, flush=True)


# ---- (14b) EAGLE KEEP-VERIFIED prefix-cache fix (MTP x fp8-KV hits=0 root cause) -------------------
# ROOT CAUSE (2026-07-22, proven with (14a) instrumentation): NOT a drafter dtype/group mismatch (the
# MTP drafter layer is already fp8, merged into the 17-layer FullAttention group). The zero hits are the
# EAGLE/MTP LAST-BLOCK DROP x the fp8 block size: the drafter's KV for block k depends on ONE token
# beyond the block's hash (MTP shift-by-one: position p attends embed(token[p+1])), so upstream only
# trusts block k when block k+1's hash ALSO matched, i.e. it matches one extra block and drops the last.
# With fp8 KV the attention block is 1664 tokens (page >= mamba page), so any prompt < 2*1664+1 = 3329
# tokens can NEVER hit (the 2428-tok probe and the IN=2048 bench -> hits=0), and every hit loses 1664
# tokens. bf16 KV only "worked" because its block is 832 (threshold 1665).
# FIX: the out-of-hash dependency is exactly one token, and we can verify it directly instead of
# demanding a full extra block: record next_token[(k+1)*B] per block hash at BlockPool INSERT time
# (last-writer-wins == the flavor of the drafter KV physically in the pool; re-caching after eviction
# overwrites, so a lookup that finds the block in the pool always sees the matching record), then at
# lookup keep the last matched block iff the new request's token at the boundary equals the record.
# Restores the no-MTP threshold (1665) and reclaims 1664 tok per hit. Companion mamba cap: the
# coordinator inflates every eagle group's ceiling by one block; MambaManager IGNORES drop_eagle_block
# (upstream bug, PR #48375) so mamba could claim a longer hit than the attn group validated ->
# inconsistent state. We cap mamba's ceiling at the attn group's validated length (falling back to
# upstream #48375's minus-one-block when attn has not run). Gated B70_PC_EAGLE_KEEP=1, default OFF.
if os.environ.get("B70_PC_EAGLE_KEEP", "0") == "1":
    try:
        from vllm.v1.core.block_pool import BlockPool as _EKPool
        from vllm.v1.core.kv_cache_manager import KVCacheManager as _EKMgr
        import vllm.v1.core.kv_cache_coordinator as _ekco
        import vllm.v1.core.single_type_kv_cache_manager as _ekstm

        _EK_CAP = int(os.environ.get("B70_PC_EAGLE_KEEP_CAP", "200000"))
        _ek_next = {}       # BlockHash bytes -> token id one past the block (pool-coherent)
        _ek_req = [None]    # request whose lookup is in flight (scheduler is single-threaded)
        _ek_attn_len = [None]  # attn group's validated hit length (tok) for the current lookup
        _ek_stats = {"keep": 0, "drop": 0}
        _EK_DBG = os.environ.get("B70_PC_DEBUG", "0") == "1"

        # (a) record the boundary token at pool insert time.
        _ek_orig_cfb = _EKPool.cache_full_blocks

        def _ek_cfb(self, request, blocks, num_cached_blocks, num_full_blocks,
                    block_size, kv_cache_group_id, block_mask=None):
            _ek_orig_cfb(self, request, blocks, num_cached_blocks, num_full_blocks,
                         block_size, kv_cache_group_id, block_mask)
            try:
                if block_size != self.hash_block_size:
                    return  # hashes not at block granularity; keep-verify stays off for these
                if len(_ek_next) > _EK_CAP:
                    _ek_next.clear()
                toks = request.all_token_ids
                ntok = len(toks)
                bh = request.block_hashes
                nbh = len(bh)
                for i, k in enumerate(range(num_cached_blocks, num_full_blocks)):
                    if block_mask is not None and not block_mask[i]:
                        continue
                    nxt = (k + 1) * block_size
                    if nxt < ntok and k < nbh:
                        _ek_next[bh[k]] = toks[nxt]
            except Exception:
                pass

        _EKPool.cache_full_blocks = _ek_cfb

        # (b) expose the request to the lookup path.
        _ek_orig_gcb = _EKMgr.get_computed_blocks

        def _ek_gcb(self, request):
            _ek_req[0] = request
            try:
                return _ek_orig_gcb(self, request)
            finally:
                _ek_req[0] = None

        _EKMgr.get_computed_blocks = _ek_gcb

        # (c) reset per-lookup state (hybrid coordinator only; unitary has no cross-group risk).
        _ek_orig_hy = _ekco.HybridKVCacheCoordinator.find_longest_cache_hit

        def _ek_hy(self, block_hashes, max_cache_hit_length):
            _ek_attn_len[0] = None
            return _ek_orig_hy(self, block_hashes, max_cache_hit_length)

        _ekco.HybridKVCacheCoordinator.find_longest_cache_hit = _ek_hy

        # (d) FullAttention: match WITHOUT the drop, then keep the last block iff the boundary
        # token verifies; otherwise drop + re-align (upstream behavior).
        _ek_orig_fa = _ekstm.FullAttentionManager.find_longest_cache_hit.__func__

        def _ek_fa(cls, block_hashes, max_length, kv_cache_group_ids, block_pool,
                   kv_cache_spec, drop_eagle_block, alignment_tokens, **kw):
            out = _ek_orig_fa(cls, block_hashes, max_length, kv_cache_group_ids,
                              block_pool, kv_cache_spec, False, alignment_tokens, **kw)
            if not drop_eagle_block:
                return out
            n = len(out[0])
            req = _ek_req[0]
            keep = False
            bs = kv_cache_spec.block_size
            if n > 0 and req is not None:
                try:
                    rec = _ek_next.get(block_hashes[n - 1])
                    toks = req.all_token_ids
                    nxt = n * bs
                    keep = rec is not None and nxt < len(toks) and rec == toks[nxt]
                except Exception:
                    keep = False
            if keep:
                _ek_stats["keep"] += 1
            else:
                _ek_stats["drop"] += 1
                if n > 0:
                    for computed in out:
                        computed.pop()
                    while out[0] and (len(out[0]) * bs) % alignment_tokens != 0:
                        for computed in out:
                            computed.pop()
            _ek_attn_len[0] = len(out[0]) * bs
            if _EK_DBG:
                print("[eagle-keep] n=%d keep=%s attn_len=%d (keep=%d drop=%d)" % (
                    n, keep, _ek_attn_len[0], _ek_stats["keep"], _ek_stats["drop"]),
                    file=sys.stderr, flush=True)
            return out

        _ekstm.FullAttentionManager.find_longest_cache_hit = classmethod(_ek_fa)

        # (e) Mamba: cap the eagle-inflated ceiling at the attn group's validated length
        # (mamba has no eagle shift; it must simply never exceed the attn-validated hit).
        _ek_orig_mm = _ekstm.MambaManager.find_longest_cache_hit.__func__

        def _ek_mm(cls, block_hashes, max_length, kv_cache_group_ids, block_pool,
                   kv_cache_spec, drop_eagle_block, alignment_tokens, **kw):
            if drop_eagle_block:
                if _ek_attn_len[0] is not None:
                    max_length = min(max_length, _ek_attn_len[0])
                else:  # no attn group ran first: upstream #48375 fallback
                    max_length = max(0, max_length - kv_cache_spec.block_size)
            return _ek_orig_mm(cls, block_hashes, max_length, kv_cache_group_ids,
                               block_pool, kv_cache_spec, False, alignment_tokens, **kw)

        _ekstm.MambaManager.find_longest_cache_hit = classmethod(_ek_mm)

        print("[nvfp4-shim] (14b) EAGLE KEEP-VERIFIED prefix cache ON "
              "(boundary-token verify replaces the eagle last-block drop; mamba ceiling capped)",
              file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (14b) eagle keep-verified install failed:", repr(e), file=sys.stderr, flush=True)


# ---- (14c) PR#45477 port: mamba align chunk-splits must END on block boundaries --------------------
# Upstream OPEN PR https://github.com/vllm-project/vllm/pull/45477 (fixes #43559, APC+MTP accuracy
# drop on Qwen3.6 hybrid GDN -- upstream-verified on Qwen3.6-27B-FP8). Our image's
# Scheduler._mamba_block_aligned_split floors the chunk LENGTH (aligned only while the start is
# aligned) and falls through UNALIGNED past last_cache_position; with MTP (use_eagle) the eagle prune
# zeroes last_cache_position for short prompts so the first chunk is budget-capped unaligned. An
# unaligned non-final chunk end makes the GDN kernel leave the running recurrent state in a mid-block
# slot which cache_blocks later hashes as a BOUNDARY state -> poisoned prefix cache (wrong recurrent
# state on every same-prefix resume; measured as cold-vs-warm temp-0 divergence) or nulls the boundary
# slot (permanent APC miss). Port = upstream's rewritten split (align the chunk END, clamp to
# last_cache_position, re-align after mid-block resume, stay aligned past last_cache_position until
# the final chunk). DELTA vs upstream: the eagle prune of last_cache_position exists only to pair with
# the FullAttn eagle last-block DROP; with (14b) keep-verified the drop is usually skipped, so pruning
# would just deny the final boundary a state snapshot (capping warm hits one block short). We skip the
# prune when B70_PC_EAGLE_KEEP=1: every boundary <= aligned(num_tokens) gets a snapshot; if a lookup's
# keep-verify fails, (14b)'s mamba cap keeps the intersection coherent. Gated B70_PC_CHUNK_ALIGN=1,
# default OFF.
if os.environ.get("B70_PC_CHUNK_ALIGN", "0") == "1":
    try:
        from vllm.v1.core.sched import scheduler as _b70_sched_mod

        _B70_EK_ON = os.environ.get("B70_PC_EAGLE_KEEP", "0") == "1"

        def _b70_mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens=0,
            num_external_computed_tokens=0,
            num_uncached_common_prefix_tokens=0,
        ):
            num_computed_tokens = (
                request.num_computed_tokens
                + num_new_local_computed_tokens
                + num_external_computed_tokens
            )
            prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
            if num_computed_tokens >= prefill_end:
                return num_new_tokens  # decode phase: no splitting

            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens // block_size * block_size
            if self.use_eagle and not _B70_EK_ON:
                # pair of the FullAttn eagle last-block drop; unnecessary with keep-verified
                last_cache_position = max(last_cache_position - block_size, 0)

            chunk_end = num_computed_tokens + num_new_tokens
            if num_computed_tokens < last_cache_position:
                chunk_end = min(
                    chunk_end // block_size * block_size, last_cache_position
                )
                if num_computed_tokens % block_size != 0:
                    # resumed mid-block: stop at the next boundary to re-align
                    chunk_end = min(
                        chunk_end,
                        num_computed_tokens // block_size * block_size + block_size,
                    )
            elif chunk_end < prefill_end:
                # past the last cacheable boundary but not finishing prefill: stay aligned
                # (eagle-lookahead caching hashes one block past each aligned boundary)
                chunk_end = chunk_end // block_size * block_size
            num_new_tokens = max(chunk_end - num_computed_tokens, 0)

            # Marconi cache admission optimization (kept from stock, alignment preserved)
            if (
                num_uncached_common_prefix_tokens >= block_size
                and num_new_tokens > num_uncached_common_prefix_tokens
            ):
                num_new_tokens = (
                    num_uncached_common_prefix_tokens // block_size * block_size
                )
            return num_new_tokens

        _b70_sched_mod.Scheduler._mamba_block_aligned_split = _b70_mamba_block_aligned_split
        print("[nvfp4-shim] (14c) PR45477 mamba align chunk-split fix ON (eagle prune %s)"
              % ("SKIPPED, keep-verified pairs it" if _B70_EK_ON else "kept"),
              file=sys.stderr, flush=True)
    except Exception as e:
        print("[nvfp4-shim] (14c) PR45477 chunk-split fix failed:", repr(e), file=sys.stderr, flush=True)
