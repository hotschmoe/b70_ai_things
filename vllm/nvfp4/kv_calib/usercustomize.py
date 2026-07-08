# KV calibration hook (conflict-free: separate usercustomize, NOT the shared sitecustomize).
# When NVFP4_KV_CALIB=1, wrap vLLM Attention.forward to record per-layer running amax of
# |key| and |value| (pre-fp8-cast), and dump {layer_name:{k,v}} to NVFP4_KV_CALIB_OUT
# (default /tmp_ssd/kv_amax.json) every update + at exit. Card-free logic; runs in-process.
import os, sys, json, atexit
if os.environ.get("NVFP4_KV_CALIB") == "1":
    OUT = os.environ.get("NVFP4_KV_CALIB_OUT", "/tmp_ssd/kv_amax.json")
    _amax = {}  # layer_name -> [k_amax, v_amax]
    def _dump():
        try:
            with open(OUT, "w") as f: json.dump(_amax, f)
        except Exception as e:
            print("[kv-calib] dump failed:", e, file=sys.stderr, flush=True)
    atexit.register(_dump)
    _n = [0]
    def _install():
        Attention = None
        for modpath, cls in [
            ("vllm.model_executor.layers.attention.attention", "Attention"),
            ("vllm.attention.layer", "Attention"),
            ("vllm.attention", "Attention"),
        ]:
            try:
                m = __import__(modpath, fromlist=[cls]); Attention = getattr(m, cls); break
            except Exception:
                continue
        if Attention is None:
            print("[kv-calib] could not locate Attention class", file=sys.stderr, flush=True); return
        _orig = Attention.forward
        def forward(self, *a, **kw):
            try:
                # forward(query, key, value, ...) positional
                key = a[1] if len(a) > 1 else kw.get("key")
                value = a[2] if len(a) > 2 else kw.get("value")
                ln = getattr(self, "layer_name", None) or str(id(self))
                if key is not None and value is not None:
                    ka = float(key.detach().abs().amax().item())
                    va = float(value.detach().abs().amax().item())
                    cur = _amax.get(ln)
                    if cur is None: _amax[ln] = [ka, va]
                    else:
                        if ka > cur[0]: cur[0] = ka
                        if va > cur[1]: cur[1] = va
                    _n[0] += 1
                    if _n[0] % 256 == 0: _dump()
            except Exception as e:
                if _n[0] < 3: print("[kv-calib] hook err:", e, file=sys.stderr, flush=True)
            return _orig(self, *a, **kw)
        Attention.forward = forward
        print(f"[kv-calib] Attention.forward wrapped on {Attention.__module__}.{Attention.__name__}; OUT={OUT}",
              file=sys.stderr, flush=True)
    _install()
