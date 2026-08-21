# K8: optional runtime RTN int4 lm_head for 3.8 W4A8 vLLM.
# Chains the 3.6 W4A8 shelf shim at /opt/mtp_shim/sitecustomize.py first
# (PYTHONPATH=/opt/w4a8_k8:/opt/mtp_shim replaces the shelf sitecustomize).
# Gated B70_W4A8_QUANT_LMHEAD=1. Default off = byte-identical.
# 3.6 sglang precedent: LMHEAD=1 g32 -> +7.9% decode, HE+ held.
# Do not rewrite 151. Do not pack GDN I8 (D12). Keep bf16 weight resident.
import os
import sys
import runpy

_chain_env = os.environ.get("B70_W4A8_CHAIN_SITECUSTOMIZE", "")
_chain_cands = []
if _chain_env:
    _chain_cands.append(_chain_env)
_chain_cands.extend((
    "/opt/compile_key_shim/sitecustomize.py",
    "/opt/push_ar/sitecustomize.py",
    "/opt/mtp_shim/sitecustomize.py",
))
_chained = False
for _chain in _chain_cands:
    if _chain and os.path.isfile(_chain):
        runpy.run_path(_chain, run_name="sitecustomize")
        print("[lmhead-int4] chained", _chain, file=sys.stderr, flush=True)
        _chained = True
        break
if not _chained:
    print("[lmhead-int4] WARNING: no chain sitecustomize found",
          file=sys.stderr, flush=True)

if os.environ.get("B70_W4A8_QUANT_LMHEAD", "0") != "1":
    sys.stderr.write("[lmhead-int4] off (B70_W4A8_QUANT_LMHEAD!=1)\n")
    sys.stderr.flush()
else:
    try:
        import torch
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        _g = int(os.environ.get("B70_W4A8_LMHEAD_GROUP", "32"))
        _has_op = hasattr(torch.ops, "_xpu_C") and hasattr(
            torch.ops._xpu_C, "int4_gemm_w4a16"
        )
        if not _has_op:
            print("[lmhead-int4] int4_gemm_w4a16 missing -> lm_head stays bf16",
                  file=sys.stderr, flush=True)
        else:
            def _quant_lmhead_w(weight, g):
                # weight [N,K] float -> qw [N,K/8] i32, sc [N,K/g] fp16.
                # sym zp=8, q in [-7,7] (3.6 woq_shim). Chunk N to bound fp32.
                N, K = weight.shape
                if K % g != 0:
                    raise ValueError("lm_head K=%s not divisible by group %s" % (K, g))
                dev = weight.device
                qw = torch.empty(N, K // 8, dtype=torch.int32, device=dev)
                sc = torch.empty(N, K // g, dtype=torch.float16, device=dev)
                shifts = torch.tensor(
                    [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=dev
                )
                step = 16384
                for r0 in range(0, N, step):
                    r1 = min(r0 + step, N)
                    Wg = weight[r0:r1].reshape(r1 - r0, K // g, g).to(torch.float32)
                    amax = Wg.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
                    scale = amax / 7.0
                    q = torch.round(Wg / scale).clamp_(-7, 7).to(torch.int32)
                    sc[r0:r1] = scale.squeeze(-1).to(torch.float16)
                    stored = (q + 8).reshape(r1 - r0, K // 8, 8)
                    qw[r0:r1] = (stored << shifts).sum(dim=-1).to(torch.int32)
                return qw, sc

            def _attach_one(name, lm):
                if lm is None or not hasattr(lm, "weight"):
                    return False
                if getattr(lm, "_b70_int4", None) is not None:
                    return True
                w = lm.weight.data
                if w is None or w.dim() != 2 or w.dtype not in (
                    torch.bfloat16, torch.float16
                ):
                    print("[lmhead-int4] skip %s dtype/shape %s %s" % (
                        name, getattr(w, "dtype", None), getattr(w, "shape", None)
                    ), file=sys.stderr, flush=True)
                    return False
                N, K = w.shape
                print("[lmhead-int4] RTN %s N=%s K=%s g=%s ..." % (name, N, K, _g),
                      file=sys.stderr, flush=True)
                qw, sc = _quant_lmhead_w(w, _g)
                qweight_t = qw.t()
                if qweight_t.stride()[0] != 1:
                    raise RuntimeError("qweight_t stride0=%s want 1" % (qweight_t.stride()[0],))
                lm._b70_int4 = {
                    "qw": qw,
                    "qweight_t": qweight_t,
                    "wscale_t": sc.t().contiguous(),
                    "wzp": torch.tensor([8], dtype=torch.int8, device=w.device),
                    "g": _g,
                }
                if hasattr(torch, "xpu"):
                    torch.xpu.empty_cache()
                print("[lmhead-int4] ready %s int4=%.2fGB bf16 kept=%.2fGB" % (
                    name, qw.numel() * 4 / 1e9, w.numel() * 2 / 1e9
                ), file=sys.stderr, flush=True)
                return True

            def _attach_int4_lmheads(model):
                n = 0
                for name, mod in model.named_modules():
                    if name == "lm_head" or name.endswith(".lm_head"):
                        if _attach_one(name, mod):
                            n += 1
                if n == 0:
                    print("[lmhead-int4] no lm_head module found",
                          file=sys.stderr, flush=True)
                return n

            def _wrap_load(cls, tag):
                orig = cls.load_model

                def _load(self, *a, **k):
                    orig(self, *a, **k)
                    try:
                        m = getattr(self, "model", None)
                        if m is not None:
                            _attach_int4_lmheads(m)
                    except Exception as e:
                        print("[lmhead-int4] attach FAILED:", repr(e),
                              file=sys.stderr, flush=True)

                cls.load_model = _load
                print("[lmhead-int4] wrapped %s.load_model" % tag,
                      file=sys.stderr, flush=True)

            try:
                from vllm.v1.worker.xpu_model_runner import XPUModelRunner
                _wrap_load(XPUModelRunner, "XPUModelRunner")
            except Exception as e:
                print("[lmhead-int4] wrap XPUModelRunner failed:", repr(e),
                      file=sys.stderr, flush=True)
            try:
                from vllm.v1.worker.xpu_model_runner import XPUModelRunnerV2
                _wrap_load(XPUModelRunnerV2, "XPUModelRunnerV2")
            except Exception as e:
                print("[lmhead-int4] wrap XPUModelRunnerV2 failed:", repr(e),
                      file=sys.stderr, flush=True)

            _orig_apply = LogitsProcessor._apply_head
            _hits = [0]

            def _apply_head_int4(self, lm_head, hidden_states, embedding_bias):
                q = getattr(lm_head, "_b70_int4", None)
                if q is not None and embedding_bias is None:
                    xf = hidden_states.reshape(-1, hidden_states.shape[-1])
                    xf = xf.to(torch.float16).contiguous()
                    out = torch.ops._xpu_C.int4_gemm_w4a16(
                        xf, q["qweight_t"], None, q["wscale_t"], q["wzp"], q["g"], None
                    )
                    if _hits[0] < 3:
                        _hits[0] += 1
                        print("[lmhead-int4] apply hit %s M=%s N=%s" % (
                            _hits[0], xf.shape[0], q["qweight_t"].shape[1]
                        ), file=sys.stderr, flush=True)
                    return out.to(hidden_states.dtype).reshape(
                        *hidden_states.shape[:-1], -1
                    )
                return _orig_apply(self, lm_head, hidden_states, embedding_bias)

            LogitsProcessor._apply_head = _apply_head_int4
            print("[lmhead-int4] ENABLED g=%s via int4_gemm_w4a16" % _g,
                  file=sys.stderr, flush=True)
    except Exception as e:
        print("[lmhead-int4] install FAILED:", repr(e), file=sys.stderr, flush=True)
