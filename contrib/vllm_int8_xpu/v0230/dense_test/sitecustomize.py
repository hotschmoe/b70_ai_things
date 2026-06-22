# b70 task #12 TEST: make a DENSE CompressedTensorsW8A8Int8 model use the task-c XPU int8 Triton kernel on v0230.
# v0230 stock has NO XPU entry in _POSSIBLE_INT8_KERNELS -> dense W8A8 KeyErrors / dequants. The register fn lives
# in the (mounted) quark.py but is only called from the Quark path. Here we wrap init_int8_linear_kernel (called by
# BOTH Quark and CompressedTensors W8A8 at model-load, reads the registry at call-time) to register first.
import os
os.environ.setdefault("B70_INT8_LINEAR", "triton")
try:
    import vllm.model_executor.kernels.linear as _L
    _orig = _L.init_int8_linear_kernel
    _done = {"v": False}
    def _wrapped(*a, **k):
        if not _done["v"]:
            try:
                from vllm.model_executor.layers.quantization.quark.quark import _b70_register_xpu_int8_kernel
                _b70_register_xpu_int8_kernel()
                print("[b70 int8-dense TEST] registered XPU int8 triton kernel before init_int8_linear_kernel", flush=True)
            except Exception as e:
                print("[b70 int8-dense TEST] register FAILED:", repr(e), flush=True)
            _done["v"] = True
        return _orig(*a, **k)
    _L.init_int8_linear_kernel = _wrapped
    print("[b70 int8-dense TEST] hooked init_int8_linear_kernel", flush=True)
except Exception as e:
    print("[b70 int8-dense TEST] hook setup FAILED:", repr(e), flush=True)
