# b70: bypass benchmark_moe.py --tune Ray-on-XPU gap. Stock ray.init() -> available_resources()
# has no "GPU" key on XPU -> KeyError. Force ray.init(num_gpus=1): Ray then registers 1 schedulable
# GPU resource so the BenchmarkWorker actor launches; it still computes on the XPU via ZE_AFFINITY_MASK.
try:
    import ray
    _orig = ray.init
    def _patched(*a, **k):
        k.setdefault("num_gpus", 1)
        return _orig(*a, **k)
    ray.init = _patched
    print("[b70 sitecustomize] patched ray.init(num_gpus=1)", flush=True)
except Exception as e:
    print("[b70 sitecustomize] ray patch skipped:", e, flush=True)
