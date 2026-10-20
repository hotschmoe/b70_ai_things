"""Write exact allocator snapshots, cache specs, and storage ownership as JSON."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
import gc
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

    def device_info():
        indices = range(2) if os.environ.get("B70_EP") == "1" else [torch.xpu.current_device()]
        return {str(i): dict(native=torch.xpu.mem_get_info(i),
                             profiling=torch.accelerator.get_memory_info(i),
                             allocated=torch.xpu.memory_allocated(i),
                             reserved=torch.xpu.memory_reserved(i)) for i in indices}

    old_load = GPUModelRunner.load_model

    def load(self, *args, **kwargs):
        result = old_load(self, *args, **kwargs)
        owned = census(self.model)
        emit("weights", owned)
        extra = {}
        for obj in gc.get_objects():
            if not isinstance(obj, torch.Tensor):
                continue
            try:
                if obj.device.type != "xpu" or not obj.numel():
                    continue
                storage = obj.untyped_storage()
                key = f"{obj.device}:{storage.data_ptr()}"
                if key not in owned["storage_bytes"]:
                    extra[key] = dict(bytes=storage.nbytes(), shape=list(obj.shape),
                                      dtype=str(obj.dtype), type=type(obj).__name__)
            except (RuntimeError, ReferenceError):
                continue
        del obj
        emit("unowned-tensors", extra)
        reference = os.environ.get("B70_EMBED_REF") or str(target.parent / "embedding-reference.npz")
        if reference and os.environ.get("B70_MEMORY_FIX") == "1":
            import numpy as np
            from vllm.models.qwen4_exp.gguf.embedding import B70GGUFEmbeddingMethod
            table = next(m for n, m in self.model.named_modules() if n.endswith("embed_tokens"))
            if not isinstance(table.quant_method, B70GGUFEmbeddingMethod):
                raise RuntimeError("packed embedding method was not selected")
            ref = np.load(reference)
            ids = torch.tensor(ref["ids"], dtype=torch.int64, device=table.b70_packed.device)
            rows = table.quant_method.embedding(table, ids)
            actual = rows.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
            bad = int(np.count_nonzero(actual != ref["bf16"]))
            emit("embedding-check", dict(rows=len(ref["ids"]), values=actual.size, bad=bad))
            if bad:
                raise RuntimeError(f"packed embedding differs in {bad} BF16 values")
            del table, ref, ids, rows, actual
        emit("allocator-after-load", torch.xpu.memory._snapshot())
        emit("devices-after-load", device_info())
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
        emit("devices-after-profile", device_info())

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
        before = dict(free=torch.xpu.mem_get_info()[0],
                      reserved=torch.xpu.memory_reserved(), allocated=torch.xpu.memory_allocated())
        estimate = old_graph(self, *args, **kwargs)
        after = dict(free=torch.xpu.mem_get_info()[0],
                     reserved=torch.xpu.memory_reserved(), allocated=torch.xpu.memory_allocated())
        emit("graph", dict(estimate_bytes=estimate, before=before, after=after,
                           retained_bytes=before["free"]-after["free"]))
        return estimate

    GPUModelRunner.profile_cudagraph_memory = graph
    old_budget = worker.Worker.determine_available_memory

    def budget(self, *args, **kwargs):
        available = old_budget(self, *args, **kwargs)
        emit("budget", dict(requested_bytes=self.requested_memory,
                            available_kv_bytes=available,
                            init=_plain(self.init_snapshot),
                            consumed_bytes=getattr(self, "total_consumed", None),
                            peak_headroom_bytes=getattr(self, "peak_activation_memory", None)))
        return available

    worker.Worker.determine_available_memory = budget
    old_kv_init = GPUModelRunner.initialize_kv_cache

    def kv_init(self, *args, **kwargs):
        before = device_info()
        result = old_kv_init(self, *args, **kwargs)
        config = args[0] if args else kwargs.get("kv_cache_config")
        emit("kv-initialized", dict(before=before, after=device_info(),
                                    blocks=getattr(config, "num_blocks", None),
                                    profiling=kwargs.get("is_profiling", False)))
        emit("post-kv-storage", census(self.model))
        return result

    GPUModelRunner.initialize_kv_cache = kv_init
