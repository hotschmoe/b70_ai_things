"""Write exact allocator snapshots, cache specs, and storage ownership as JSON."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path


def _plain(snapshot):
    return {k: v for k, v in vars(snapshot).items() if isinstance(v, (int, float, str, bool))}


def census(model):
    """Visit module tensors plus the GGUF method's non-registered shards."""
    import torch
    from vllm.models.qwen4_exp.gguf.linear import PackedShard

    seen_objects = set()
    storages = {}
    references = []

    def visit(obj, name):
        if isinstance(obj, torch.Tensor):
            if obj.device.type != "xpu" or not obj.numel():
                return
            storage = obj.untyped_storage()
            key = f"{obj.device}:{storage.data_ptr()}"
            storages[key] = storage.nbytes()
            references.append(dict(name=name, storage=key, bytes=obj.numel()*obj.element_size(),
                                   shape=list(obj.shape), dtype=str(obj.dtype)))
            return
        if id(obj) in seen_objects:
            return
        seen_objects.add(id(obj))
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{name}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{name}.{i}")
        elif isinstance(obj, PackedShard):
            visit(obj.packed, name + ".packed")
            visit(obj.dense, name + ".dense")

    for name, module in model.named_modules():
        for attr, value in vars(module).items():
            if attr != "_modules":
                visit(value, f"{name}.{attr}")
        method = getattr(module, "quant_method", None)
        if method is not None:
            for attr, value in vars(method).items():
                visit(value, f"{name}.quant_method.{attr}")
    return dict(storage_bytes=storages, unique_bytes=sum(storages.values()), references=references)


def install(directory):
    import torch
    import vllm.v1.worker.gpu_worker as worker
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    def emit(stage, value):
        path = target / f"{os.getpid()}-{stage}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="ascii")
        print(f"b70_memory: {stage} evidence {path}", flush=True)

    old_load = GPUModelRunner.load_model

    def load(self, *args, **kwargs):
        result = old_load(self, *args, **kwargs)
        emit("weights", census(self.model))
        emit("allocator-after-load", torch.xpu.memory._snapshot())
        return result

    GPUModelRunner.load_model = load
    old_profile = worker.memory_profiling

    @contextmanager
    def profile(*args, **kwargs):
        with old_profile(*args, **kwargs) as result:
            yield result
        emit("profile", dict(before_create=_plain(result.before_create),
                             before_profile=_plain(result.before_profile),
                             after_profile=_plain(result.after_profile),
                             **{k: v for k, v in vars(result).items() if isinstance(v, (int, float))}))
        from b70_gguf_xpu import prefill_dq
        emit("dequant", {str(k): dict(shape=list(v.shape), bytes=v.untyped_storage().nbytes())
                         for k, v in prefill_dq._SCRATCH.items()})
        emit("allocator-after-profile", torch.xpu.memory._snapshot())

    worker.memory_profiling = profile
    old_spec = GPUModelRunner.get_kv_cache_spec

    def specs(self, *args, **kwargs):
        result = old_spec(self, *args, **kwargs)
        emit("kv-spec", {k: dict(spec=asdict(v), type=type(v).__name__,
                                 page_bytes=v.page_size_bytes) for k, v in result.items()})
        return result

    GPUModelRunner.get_kv_cache_spec = specs
    old_run = GPUModelRunner.profile_run

    def run(self, *args, **kwargs):
        operations = []
        handles = []

        def snapshot(name):
            stats = torch.xpu.memory_stats()
            free, total = torch.xpu.mem_get_info()
            return dict(name=name, allocated=stats["allocated_bytes.all.current"],
                        reserved=stats["reserved_bytes.all.current"],
                        peak=stats["allocated_bytes.all.peak"], free=free, total=total)

        def after(name):
            def hook(module, args, result):
                operations.append(snapshot(name))
            return hook

        for name, module in self.model.named_modules():
            if hasattr(module, "quant_method") or type(module).__name__ in (
                "HyperConnection", "Qwen4ExpDecoderLayer", "Qwen4ExpPLE",
            ):
                handles.append(module.register_forward_hook(after(name)))
        operations.append(snapshot("profile-start"))
        try:
            return old_run(self, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
            operations.append(snapshot("profile-end"))
            emit("operations", operations)

    GPUModelRunner.profile_run = run
    old_graph = GPUModelRunner.profile_cudagraph_memory

    def graph(self, *args, **kwargs):
        estimate = old_graph(self, *args, **kwargs)
        emit("graph", dict(estimate_bytes=estimate))
        return estimate

    GPUModelRunner.profile_cudagraph_memory = graph
