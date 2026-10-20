#!/usr/bin/env python3
"""O4: in-process Flash-Next decode-step timeline (sidecar or DP layout).

Runs vLLM in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the
runner can be instrumented without a server. Per real step it records:

  gap_ms       previous sample end -> this execute_model start (engine)
  prep_ms      execute_model start -> PLE stage start (inputs, metadata)
  ple_*        PLE host stage: d2h (first .to(cpu), includes the GPU
               drain), hash, gather (host index + H2D + dequant enqueue)
  replay_host  host time inside run_fullgraph (both graph submits)
  replay_dev   card-0 event pair around run_fullgraph (device time)
  sample_ms    sample_tokens wall; sample_dev its card-0 event pair
  step_ms      execute start -> next execute start

Gates: greedy chat "17*19" -> 323 and "23*29" -> 667.

Env:
  O4_OUT        output directory (required)
  O4_MODE       "sidecar" (default) or "dp"
  O4_TOKENS     decode tokens per timing request (default 96)
  O4_UT_SESSION unitrace session name; resumes tracing only around one
                short C1 request (the process must run under unitrace
                --start-paused ... is NOT used: graphs must be captured
                with collection on, so tracing starts on and this script
                pauses it after load).
  O4_PLE_FIX    1 = install the O4 PLE fast path (see o4_ple.py)
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

OUT = os.environ["O4_OUT"]
os.makedirs(OUT, exist_ok=True)
_LOG = open(os.path.join(OUT, "profile.log"), "a", encoding="utf-8")


def log(msg: str) -> None:
    line = f"O4 {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    _LOG.write(line + "\n")
    _LOG.flush()


UT = "/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace"
UT_SESSION = os.environ.get("O4_UT_SESSION", "")


def ut(cmd: str) -> None:
    if not UT_SESSION:
        return
    r = subprocess.run([UT, f"--{cmd}", UT_SESSION], capture_output=True, text=True)
    log(f"unitrace {cmd} rc={r.returncode} {r.stdout.strip()[:200]} {r.stderr.strip()[:200]}")


import torch  # noqa: E402

STEPS: list[dict] = []
CUR: dict = {}
STATE = {"last_end": None, "record": False}


def _ev():
    e = torch.xpu.Event(enable_timing=True)
    e.record()
    return e


def install_hooks() -> None:
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.models.qwen4_exp.xpu import model as fm

    R = mr.GPUModelRunner
    orig_exec = R.execute_model

    def execute_model(self, so, *a, **k):
        if k.get("dummy_run") or not STATE["record"]:
            return orig_exec(self, so, *a, **k)
        CUR.clear()
        t0 = time.perf_counter()
        CUR["t0"] = t0
        CUR["ntok"] = int(getattr(so, "total_num_scheduled_tokens", -1))
        CUR["nreq"] = len(getattr(so, "num_scheduled_tokens", {}) or {})
        if STATE["last_end"] is not None:
            CUR["gap_ms"] = (t0 - STATE["last_end"]) * 1e3
        out = orig_exec(self, so, *a, **k)
        CUR["exec_ms"] = (time.perf_counter() - t0) * 1e3
        return out

    orig_sample = R.sample_tokens

    def sample_tokens(self, *a, **k):
        if not STATE["record"] or "t0" not in CUR:
            return orig_sample(self, *a, **k)
        t = time.perf_counter()
        s = _ev()
        out = orig_sample(self, *a, **k)
        e = _ev()
        t1 = time.perf_counter()
        CUR["sample_ms"] = (t1 - t) * 1e3
        CUR["sample_ev"] = (s, e)
        STATE["last_end"] = t1
        STEPS.append(dict(CUR))
        CUR.clear()
        return out

    R.execute_model = execute_model
    R.sample_tokens = sample_tokens

    orig_run = CudaGraphManager.run_fullgraph

    def run_fullgraph(self, desc):
        if not STATE["record"] or "t0" not in CUR:
            return orig_run(self, desc)
        t = time.perf_counter()
        CUR["prep_ms_to_replay"] = (t - CUR["t0"]) * 1e3
        s = _ev()
        out = orig_run(self, desc)
        e = _ev()
        CUR["replay_host_ms"] = (time.perf_counter() - t) * 1e3
        CUR["replay_ev"] = (s, e)
        CUR["graph_nt"] = int(desc.num_tokens)
        return out

    CudaGraphManager.run_fullgraph = run_fullgraph

    P = fm.Qwen4ExpPLE
    orig_stage = P.stage_graph_rows

    def stage_graph_rows(self, input_ids, attn_metadata, num_slots):
        if not STATE["record"] or "t0" not in CUR:
            return orig_stage(self, input_ids, attn_metadata, num_slots)
        t = time.perf_counter()
        CUR["prep_ms"] = (t - CUR["t0"]) * 1e3
        # Mirror of the fork body with timers. Same calls, same order.
        if self._graph_emb is None or num_slots == 0:
            return None
        qsl_gpu = getattr(attn_metadata, "non_spec_query_start_loc", None)
        state_idx_gpu = getattr(attn_metadata, "non_spec_state_indices_tensor", None)
        if qsl_gpu is None or state_idx_gpu is None:
            return None
        has_init = getattr(attn_metadata, "has_initial_state", None)
        device = self._graph_emb.device
        self._ensure_state(num_slots, device)
        input_ids_cpu = input_ids.to("cpu")
        t1 = time.perf_counter()
        qsl_cpu = qsl_gpu.to("cpu")
        state_idx_cpu = state_idx_gpu.to("cpu")
        has_init_cpu = has_init.to("cpu") if has_init is not None else None
        t2 = time.perf_counter()
        rows_cpu = self._compute_rows(input_ids_cpu, qsl_cpu, state_idx_cpu, has_init_cpu)
        t3 = time.perf_counter()
        from vllm.models.qwen4_exp.gguf.ple import gather_host_rows

        emb = gather_host_rows(self.host_table, rows_cpu.reshape(-1), device).reshape(
            input_ids_cpu.shape[0], -1
        )
        self._graph_emb[: emb.shape[0]].copy_(emb)
        t4 = time.perf_counter()
        CUR["ple_d2h_ids_ms"] = (t1 - t) * 1e3
        CUR["ple_d2h_meta_ms"] = (t2 - t1) * 1e3
        CUR["ple_hash_ms"] = (t3 - t2) * 1e3
        CUR["ple_gather_ms"] = (t4 - t3) * 1e3
        return None

    if os.environ.get("O4_PLE_FIX") != "1":
        P.stage_graph_rows = stage_graph_rows

    # X1 PLE ring (B70_MTP_RT set, MTP 0): time its host stage.
    from vllm.models.qwen4_exp.gguf import mtp_ple as ring
    from vllm.models.qwen4_exp.gguf import ple as ple_mod

    orig_ring_stage = ring.PLEHistory.stage

    def ring_stage(self, ids, positions, meta, slots):
        if not STATE["record"] or "t0" not in CUR:
            return orig_ring_stage(self, ids, positions, meta, slots)
        t = time.perf_counter()
        CUR["prep_ms"] = (t - CUR["t0"]) * 1e3
        CUR["_ring_t"] = t
        out = orig_ring_stage(self, ids, positions, meta, slots)
        CUR["ring_stage_ms"] = (time.perf_counter() - t) * 1e3
        return out

    ring.PLEHistory.stage = ring_stage
    orig_rows = P._compute_rows

    def compute_rows(self, *a, **k):
        if STATE["record"] and "_ring_t" in CUR and "ring_d2h_ms" not in CUR:
            CUR["ring_d2h_ms"] = (time.perf_counter() - CUR["_ring_t"]) * 1e3
        return orig_rows(self, *a, **k)

    P._compute_rows = compute_rows
    orig_gather = ple_mod.gather_host_rows

    def gather_host_rows(table, row_ids, device):
        t = time.perf_counter()
        out = orig_gather(table, row_ids, device)
        if STATE["record"] and "t0" in CUR:
            CUR["gather_ms"] = CUR.get("gather_ms", 0.0) + (time.perf_counter() - t) * 1e3
        return out

    ple_mod.gather_host_rows = gather_host_rows
    log("hooks installed")


def finish_steps(tag: str, steps: list[dict]) -> dict:
    torch.xpu.synchronize()
    rows = []
    for i, s in enumerate(steps):
        r = {k: v for k, v in s.items() if not k.endswith("_ev") and not k.startswith("_") and k != "t0"}
        for k in ("replay_ev", "sample_ev"):
            if k in s:
                a, b = s[k]
                r[k.replace("_ev", "_dev_ms")] = a.elapsed_time(b)
        if i + 1 < len(steps):
            r["step_ms"] = (steps[i + 1]["t0"] - s["t0"]) * 1e3
        rows.append(r)
    with open(os.path.join(OUT, f"steps-{tag}.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    dec = [r for r in rows if r.get("graph_nt") is not None and "step_ms" in r]
    summ = {"tag": tag, "n_steps": len(rows), "n_graph_steps": len(dec)}
    keys = sorted({k for r in dec for k, v in r.items() if isinstance(v, (int, float))})
    for k in keys:
        vals = [r[k] for r in dec if isinstance(r.get(k), (int, float))]
        if vals:
            summ[k] = {
                "p50": round(statistics.median(vals), 3),
                "mean": round(statistics.fmean(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
            }
    log(f"SUMMARY {tag} " + json.dumps(summ))
    return summ


def main() -> None:
    mode = os.environ.get("O4_MODE", "sidecar")
    ntok = int(os.environ.get("O4_TOKENS", "96"))
    install_hooks()
    if os.environ.get("O4_PLE_FIX") == "1":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import o4_ple

        o4_ple.install()
        log("o4_ple installed")
    from vllm import LLM, SamplingParams

    hf = "/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext"
    kwargs = dict(
        model=hf,
        served_model_name=["hotschmoe-dev", "qwen3.8-flash-next-iq3-xxs"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=int(os.environ.get("O4_MAX_LEN", "2048")),
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=float(os.environ.get("O4_UTIL", "0.98")),
        language_model_only=True,
        load_format="b70_gguf",
        quantization="b70_gguf",
        trust_remote_code=True,
        attention_backend="FLASH_ATTN",
        compilation_config={
            "mode": 0,
            "cudagraph_mode": os.environ.get("O4_CG", "FULL_DECODE_ONLY"),
            "cudagraph_capture_sizes": [1, 2, 3, 4, 8],
        },
    )
    if os.environ.get("B70_MTP_RT"):
        # X1's PLE ring requires these (mtp_ple.install).
        kwargs["enable_prefix_caching"] = False
        kwargs["mamba_cache_mode"] = "align"
    if os.environ.get("O4_EAGER") == "1":
        kwargs.pop("compilation_config")
        kwargs["enforce_eager"] = True
    log(f"mode={mode} kwargs={json.dumps({k: str(v) for k, v in kwargs.items()})}")
    t = time.perf_counter()
    llm = LLM(**kwargs)
    log(f"LLM ready in {time.perf_counter() - t:.1f}s")
    ut("pause")

    # Gates.
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    gates = []
    for content, ref in (
        ("Return only the integer: 17 * 19.", "323"),
        ("Return only the integer: 23 * 29.", "667"),
    ):
        out = llm.chat(
            [{"role": "user", "content": content}],
            sp,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        text = out[0].outputs[0].text
        gates.append({"ref": ref, "text": text, "match": text.strip() == ref})
    log("GATES " + json.dumps(gates))
    ok = all(g["match"] for g in gates)
    with open(os.path.join(OUT, "gates.json"), "w") as fh:
        json.dump(gates, fh, indent=2)

    prompt = "Write a long, detailed essay about the history of the city of Paris."
    spt = SamplingParams(temperature=0.0, max_tokens=ntok, ignore_eos=True)
    # Warm one request (first-use paths), not recorded.
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True), use_tqdm=False)

    for tag, n in (("c1", 1), ("c2", 2)):
        STEPS.clear()
        STATE["last_end"] = None
        STATE["record"] = True
        t = time.perf_counter()
        outs = llm.generate([prompt] * n, spt, use_tqdm=False)
        wall = time.perf_counter() - t
        STATE["record"] = False
        summ = finish_steps(tag, list(STEPS))
        txt = outs[0].outputs[0].text
        with open(os.path.join(OUT, f"text-{tag}.json"), "w") as fh:
            json.dump([o.outputs[0].token_ids and list(o.outputs[0].token_ids) for o in outs], fh)
        log(f"{tag} wall={wall:.3f}s tokens={ntok} text0={txt[:160]!r}")
        if n == 2:
            same = outs[0].outputs[0].text == outs[1].outputs[0].text
            log(f"c2 same_text={same}")

    if UT_SESSION:
        STEPS.clear()
        STATE["last_end"] = None
        STATE["record"] = True
        ut("resume")
        llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True), use_tqdm=False)
        torch.xpu.synchronize()
        ut("pause")
        STATE["record"] = False
        finish_steps("c1-traced", list(STEPS))

    log(f"DONE gates_ok={ok}")
    if UT_SESSION:
        # unitrace writes its trace in teardown; os._exit skips that.
        # The window passes --teardown-on-signal 10 (SIGUSR1).
        import signal

        log("unitrace teardown via SIGUSR1")
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(120)
    _LOG.close()
    sys.stdout.flush()
    os._exit(0 if ok else 3)


if __name__ == "__main__":
    main()
