# Bug B capture-recovery probe: the inductor graph partitioner raises KeyError('weight_scale') when a captured
# region MIXES W8A8 (has weight_scale) + BF16 GDN (none). codex #3: give the unquantized linears a DUMMY
# weight_scale so the partition input-collection succeeds; the BF16 forward (UnquantizedLinearMethod.apply)
# ignores it, so numerics are unchanged. This UNBLOCKS use_inductor_graph_partition=true to test whether THAT
# path captures coherently at TP=2 (the legacy IGP=false path serves garbage).
try:
    import torch
    from torch.nn import Parameter
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    _orig = UnquantizedLinearMethod.process_weights_after_loading
    def _patched(self, layer):
        r = _orig(self, layer)
        try:
            if not hasattr(layer, "weight_scale"):
                w = getattr(layer, "weight", None)
                if w is not None and w.dim() == 2:
                    layer.register_parameter(
                        "weight_scale",
                        Parameter(torch.ones(1, w.shape[0], dtype=torch.float32, device=w.device),
                                  requires_grad=False))
        except Exception as e:
            print("[dummy-wscale-shim] per-layer add failed:", repr(e))
        return r
    UnquantizedLinearMethod.process_weights_after_loading = _patched
    print("[dummy-wscale-shim] UnquantizedLinearMethod patched to add dummy weight_scale")
except Exception as e:
    print("[dummy-wscale-shim] patch failed:", repr(e))
