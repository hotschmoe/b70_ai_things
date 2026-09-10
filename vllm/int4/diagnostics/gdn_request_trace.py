"""Opt-in host-only GDN routing evidence. No tensors are transferred or synced.

Install as vllm/b70_gdn_request_trace.py with the accompanying R276 patch.
B70_GDN_TRACE_DIR selects a private output directory. Disabled when unset.
"""
import json
import os
import time

TRACE_DIR = os.environ.get("B70_GDN_TRACE_DIR", "")
_context = None
_step = 0
_fd = None
_error = None
_origins = {}


def _cpu_list(value):
    if getattr(getattr(value, "device", None), "type", None) != "cpu":
        raise ValueError("trace refuses non-CPU tensor")
    return value.tolist()


def _emit(event):
    global _fd, _error
    if not TRACE_DIR or _error:
        return
    try:
        if _fd is None:
            os.makedirs(TRACE_DIR, exist_ok=True)
            path = os.path.join(TRACE_DIR, "gdn-trace-%d.jsonl" % os.getpid())
            _fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        event.update(pid=os.getpid(), time_ns=time.time_ns())
        os.write(_fd, (json.dumps(event, ensure_ascii=True) + "\n").encode("ascii"))
    except Exception as exc:
        _error = type(exc).__name__
        # Do not dump exception text, which could contain external data.
        print("B70_GDN_TRACE_DISABLED " + _error, flush=True)


def _safe(fn):
    def wrapped(*args, **kwargs):
        global _error, _context
        if not TRACE_DIR or _error:
            return
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            _context = None
            _emit(dict(event="trace-disabled", reason=type(exc).__name__))
            _error = type(exc).__name__
            print("B70_GDN_TRACE_DISABLED " + _error, flush=True)
    return wrapped


@_safe
def snapshot(runner, scheduler, scheduled, accepted_was_synced):
    """Called after existing count handling; never read in-flight D2H storage."""
    global _step
    if not TRACE_DIR:
        return
    _step += 1
    batch = runner.input_batch
    new = {r.req_id: int(r.num_computed_tokens)
           for r in scheduler.scheduled_new_reqs}
    for req_id in scheduler.finished_req_ids:
        _origins.pop(req_id, None)
    _origins.update(new)
    rows = []
    for i, req_id in enumerate(batch.req_ids[:batch.num_reqs]):
        computed = int(batch.num_computed_tokens_cpu[i])
        prompt = int(batch.num_prompt_tokens[i])
        # This is the runner's CPU-written working buffer, NOT the possibly
        # in-flight input_batch.num_accepted_tokens_cpu D2H destination.
        accepted = int(runner.num_accepted_tokens.np[i])
        rows.append(dict(row=i, req_id=req_id, scheduled=int(scheduled[i]),
                         scheduler_scheduled=int(scheduler.num_scheduled_tokens[req_id]),
                         computed_host=computed, prompt_tokens=prompt,
                         remaining_prompt_host=prompt-computed,
                         is_prefilling_host=computed < prompt,
                         new_request=req_id in new,
                         initial_computed_host=_origins.get(req_id),
                         accepted_working_host=accepted))
    runner._b70_trace_snapshot = dict(
        step=_step, rows=rows,
        accepted_source="existing-synchronized-working-copy" if accepted_was_synced
        else "default-one-working-copy-not-authoritative-async-count",
        async_spec_decode=bool(runner.use_async_spec_decode),
        cache_mode=runner.cache_config.mamba_cache_mode,
    )


@_safe
def bind_metadata(runner, metadata, capture):
    """Shallow cache-group copies retain the same CPU query-offset object.

    New draft or split metadata with a different CPU object is deliberately
    unmapped. The trace must not silently assign target IDs to draft rows.
    """
    global _context
    _context = None
    if not TRACE_DIR or capture:
        return
    snap = getattr(runner, "_b70_trace_snapshot", None)
    if snap is None:
        return
    offsets = _cpu_list(metadata.query_start_loc_cpu)
    phases = _cpu_list(metadata.is_prefilling)
    rows = snap["rows"]
    if len(offsets) < len(rows) + 1 or len(phases) < len(rows):
        _emit(dict(event="metadata-unmapped", reason="row-count-mismatch"))
        return
    for i, row in enumerate(rows):
        if offsets[i+1]-offsets[i] != row["scheduled"]:
            _emit(dict(event="metadata-unmapped", reason="query-length-mismatch",
                       step=snap["step"]))
            return
    _context = (metadata.query_start_loc_cpu, snap, phases)


@_safe
def classification(metadata, num_decodes, num_prefills, num_spec_decodes,
                   spec_masks_cpu, common_prefix_len):
    if not TRACE_DIR:
        return
    offsets = _cpu_list(metadata.query_start_loc_cpu)
    phases = None if metadata.is_prefilling is None else _cpu_list(metadata.is_prefilling)
    lengths = [b-a for a, b in zip(offsets, offsets[1:])]
    # Restrict disk traffic to the suspect effective-prefill length, not every
    # decode step or every full prompt. Padding has length zero and is excluded.
    suspect = [i for i, n in enumerate(lengths)
               if n == 1 and phases is not None and phases[i]]
    if not suspect:
        return
    mapped = _context is not None and _context[0] is metadata.query_start_loc_cpu
    masks = None if spec_masks_cpu is None else _cpu_list(spec_masks_cpu)
    rows = []
    for i in suspect:
        row = dict(_context[1]["rows"][i]) if mapped and i < len(_context[1]["rows"]) else dict(row=i)
        if masks is None:
            route = "recurrent-decode" if i < num_decodes else "initializing-prefill"
        elif masks[i]:
            route = "speculative-decode"
        else:
            # R276 promotes all non-spec singleton rows to prefill when any
            # speculative row exists. No extra tensor operations are needed.
            route = "initializing-prefill" if num_spec_decodes else "unresolved-nonspec"
        row.update(effective_query_len=lengths[i], is_prefilling=bool(phases[i]), route=route)
        rows.append(row)
    event = dict(event="gdn-short-prefill", mapped_target=mapped, rows=rows,
                 num_decodes=int(num_decodes), num_prefills=int(num_prefills),
                 num_spec_decodes=int(num_spec_decodes), common_prefix_len=int(common_prefix_len))
    if mapped:
        event.update({k:v for k,v in _context[1].items() if k != "rows"})
    _emit(event)
