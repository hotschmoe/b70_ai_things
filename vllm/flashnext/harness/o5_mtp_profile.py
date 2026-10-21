#!/usr/bin/env python3
"""O5: in-process Flash-Next MTP parity + latency probe (EP sidecar layout).

Same engine recipe as harness/o4_decode_profile.py (world size 1, EP
sidecar, FULL_DECODE_ONLY), plus an optional MTP drafter:

  O5_MTP=k      speculative tokens (0 = MTP off control)
  O5_OUT        output directory (required)
  O5_TOKENS     forced decode tokens per request (default 128)
  O5_CAPTURE    comma list of FULL capture sizes (default derived from k)
  O5_PROMPTS    number of parity prompts (default 4)

Outputs tokens-<tag>.json (token ids per prompt), steps-<tag>.jsonl and a
SUMMARY line per phase. Parity is judged offline (o5_compare.py) between
arms with identical config except O5_MTP.

Effective ms/token = (end of last step - start of first decode step) /
(tokens - 1) per request batch: the prefill step and its first token are
excluded, every later token is counted once per stream.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

OUT = os.environ["O5_OUT"]
os.makedirs(OUT, exist_ok=True)
_LOG = open(os.path.join(OUT, "profile.log"), "a", encoding="utf-8")


def log(msg: str) -> None:
    line = f"O5 {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    _LOG.write(line + "\n")
    _LOG.flush()


import subprocess  # noqa: E402

UT = "/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace"
UT_SESSION = os.environ.get("O5_UT_SESSION", "")


def ut(cmd: str) -> None:
    if not UT_SESSION:
        return
    r = subprocess.run([UT, f"--{cmd}", UT_SESSION], capture_output=True, text=True)
    log(f"unitrace {cmd} rc={r.returncode} {r.stdout.strip()[:200]} {r.stderr.strip()[:200]}")


import torch  # noqa: E402

STEPS: list[dict] = []
CUR: dict = {}
STATE = {"record": False}

PROMPTS = [
    "Write a long, detailed essay about the history of the city of Paris.",
    "Explain step by step how a hash map works, then implement one in Python with open addressing.",
    "List the first forty prime numbers and explain the sieve of Eratosthenes.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Describe the water cycle in detail for a high school science class.",
    "Write a Rust function that parses a CSV line with quoted fields, and explain it.",
]


def _ev():
    e = torch.xpu.Event(enable_timing=True)
    e.record()
    return e


def install_hooks() -> None:
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import speculator as tsp

    R = mr.GPUModelRunner
    orig_exec = R.execute_model

    def execute_model(self, so, *a, **k):
        if k.get("dummy_run") or not STATE["record"]:
            return orig_exec(self, so, *a, **k)
        if CUR:
            STEPS.append(dict(CUR))
            CUR.clear()
        CUR["t0"] = time.perf_counter()
        CUR["ntok"] = int(getattr(so, "total_num_scheduled_tokens", -1))
        CUR["nreq"] = len(getattr(so, "num_scheduled_tokens", {}) or {})
        return orig_exec(self, so, *a, **k)

    R.execute_model = execute_model

    orig_sample = R.sample_tokens

    def sample_tokens(self, *a, **k):
        if not STATE["record"] or "t0" not in CUR:
            return orig_sample(self, *a, **k)
        t = time.perf_counter()
        out = orig_sample(self, *a, **k)
        CUR["sample_wall_ms"] = (time.perf_counter() - t) * 1e3
        CUR["t_end"] = time.perf_counter()
        return out

    R.sample_tokens = sample_tokens

    orig_run = ModelCudaGraphManager.run_fullgraph

    def run_fullgraph(self, desc):
        if not STATE["record"] or "t0" not in CUR:
            return orig_run(self, desc)
        t = time.perf_counter()
        s = _ev()
        out = orig_run(self, desc)
        e = _ev()
        CUR["replay_host_ms"] = (time.perf_counter() - t) * 1e3
        CUR["replay_ev"] = (s, e)
        CUR["graph_nt"] = int(desc.num_tokens)
        return out

    ModelCudaGraphManager.run_fullgraph = run_fullgraph

    S = tsp.TargetDependentARSpeculator
    orig_prop = S.propose

    def propose(self, *a, **k):
        if not STATE["record"] or "t0" not in CUR:
            return orig_prop(self, *a, **k)
        t = time.perf_counter()
        s = _ev()
        out = orig_prop(self, *a, **k)
        e = _ev()
        CUR["propose_host_ms"] = (time.perf_counter() - t) * 1e3
        CUR["propose_ev"] = (s, e)
        return out

    S.propose = propose
    log("hooks installed")


def finish_steps(tag: str) -> dict:
    torch.xpu.synchronize()
    if CUR:
        STEPS.append(dict(CUR))
        CUR.clear()
    rows = []
    for i, s in enumerate(STEPS):
        r = {k: v for k, v in s.items() if not k.endswith("_ev") and k not in ("t0", "t_end")}
        for k in ("replay_ev", "propose_ev"):
            if k in s:
                a, b = s[k]
                r[k.replace("_ev", "_dev_ms")] = a.elapsed_time(b)
        if i + 1 < len(STEPS):
            r["step_ms"] = (STEPS[i + 1]["t0"] - s["t0"]) * 1e3
        rows.append(r)
    with open(os.path.join(OUT, f"steps-{tag}.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    dec = [r for r in rows if r.get("graph_nt") is not None]
    summ: dict = {"tag": tag, "n_steps": len(rows), "n_graph_steps": len(dec)}
    keys = sorted({k for r in dec for k, v in r.items() if isinstance(v, (int, float))})
    for k in keys:
        vals = [r[k] for r in dec if isinstance(r.get(k), (int, float))]
        if vals:
            summ[k] = {"p50": round(statistics.median(vals), 3), "mean": round(statistics.fmean(vals), 3)}
    return summ, STEPS[:]


def main() -> None:
    k = int(os.environ.get("O5_MTP", "0"))
    ntok = int(os.environ.get("O5_TOKENS", "128"))
    nprompt = int(os.environ.get("O5_PROMPTS", "4"))
    tag = os.environ.get("O5_TAG", f"mtp{k}")
    install_hooks()
    from vllm import LLM, SamplingParams

    if os.environ.get("O5_CAPTURE"):
        sizes = sorted({int(x) for x in os.environ["O5_CAPTURE"].split(",")})
    else:
        sizes = sorted({1, 2, 3, 4, 8} | ({k + 1, 2 * (k + 1)} if k else set()))
    hf = "/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext"
    kwargs = dict(
        model=hf,
        served_model_name=["hotschmoe-dev", f"qwen3.8-flash-next-iq3-xxs-sidecar-mtp{k}"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=int(os.environ.get("O5_MAX_LEN", "2048")),
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=float(os.environ.get("O5_UTIL", "0.98")),
        language_model_only=True,
        load_format="b70_gguf",
        quantization="b70_gguf",
        trust_remote_code=True,
        attention_backend="FLASH_ATTN",
        enable_prefix_caching=False,
        mamba_cache_mode="align",
        disable_log_stats=False,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": os.environ.get("O5_CG", "FULL_DECODE_ONLY"),
            "cudagraph_capture_sizes": sizes,
        },
    )
    if k:
        kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": k}
        if os.environ.get("O5_SPEC_SCHEDULE"):
            # "lo-hi-k,lo-hi-k": dynamic speculative tokens by batch size.
            kwargs["speculative_config"]["num_speculative_tokens_per_batch_size"] = [
                tuple(int(v) for v in part.split("-")) for part in os.environ["O5_SPEC_SCHEDULE"].split(",")
            ]
        if os.environ.get("B70_MTP_DRAFT_VOCAB"):
            # Restricted-vocab greedy draft head (Qwen4ExpMTP.get_top_tokens).
            kwargs["speculative_config"]["use_local_argmax_reduction"] = True
    log(f"tag={tag} k={k} kwargs={json.dumps({a: str(b) for a, b in kwargs.items()})}")
    t = time.perf_counter()
    llm = LLM(**kwargs)
    log(f"LLM ready in {time.perf_counter() - t:.1f}s")
    ut("pause")

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

    spt = SamplingParams(temperature=0.0, max_tokens=ntok, ignore_eos=True)
    llm.generate([PROMPTS[0]], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True), use_tqdm=False)

    prompts = PROMPTS[:nprompt]
    long_words = int(os.environ.get("O5_LONG_WORDS", "0"))
    if long_words:
        # Long-context parity: a shared ~long_words-word document before each
        # prompt (pushes the KV past the FA split threshold).
        base = (
            "In the archive of the old harbor city, clerks recorded every ship, cargo, tide and "
            "dispute for three centuries; the ledgers describe grain from the river valleys, salt "
            "from the southern flats, timber, wool, copper, wine and the long quarrels of guilds. "
        ).split()
        doc = " ".join(f"{w}{'' if i % 37 else ' (entry ' + str(i // 37) + ')'}" for i, w in enumerate((base * (long_words // len(base) + 1))[:long_words]))
        prompts = [f"Read this record:\n{doc}\n\nNow, unrelated: {p}" for p in prompts]
    result: dict = {"tag": tag, "k": k, "tokens": ntok, "gates": gates, "c1": [], "c2": []}
    timing: dict = {"c1": [], "c2": []}
    # C1: one prompt at a time. C2: pairs (i, i+1 mod n) so each row sees a
    # different neighbor -- batch invariance across rows and T.
    batches = {"c1": [[p] for p in prompts], "c2": [[prompts[i], prompts[(i + 1) % len(prompts)]] for i in range(0, len(prompts), 2)]}
    for phase in ("c1", "c2"):
        for bi, batch in enumerate(batches[phase]):
            STEPS.clear()
            CUR.clear()
            STATE["record"] = True
            t = time.perf_counter()
            outs = llm.generate(batch, spt, use_tqdm=False)
            wall = time.perf_counter() - t
            STATE["record"] = False
            summ, steps = finish_steps(f"{phase}-{bi}")
            ids = [list(o.outputs[0].token_ids) for o in outs]
            result[phase].append({"prompts": batch, "ids": ids})
            # First step = prefill; decode window from step 2 start to last end.
            dec_steps = [s for s in steps[1:] if "t_end" in s]
            eff = None
            if dec_steps:
                win = max(s["t_end"] for s in dec_steps) - dec_steps[0]["t0"]
                eff = win * 1e3 / max(1, ntok - 1)
            ndec = len(dec_steps)
            tok_per_step = (ntok - 1) / ndec if ndec else None
            acc = ((tok_per_step - 1) / k) if (k and tok_per_step) else None
            rec = {
                "batch": bi,
                "wall_s": round(wall, 4),
                "eff_ms_per_tok": round(eff, 3) if eff else None,
                "decode_steps": ndec,
                "tok_per_step": round(tok_per_step, 4) if tok_per_step else None,
                "accept_rate": round(acc, 4) if acc is not None else None,
                "summary": summ,
            }
            timing[phase].append(rec)
            log(
                f"{phase}[{bi}] eff={rec['eff_ms_per_tok']} ms/tok steps={ndec} "
                f"tok/step={rec['tok_per_step']} acc={rec['accept_rate']} "
                f"replay_dev={summ.get('replay_dev_ms', {}).get('p50')} "
                f"propose_dev={summ.get('propose_dev_ms', {}).get('p50')} "
                f"step={summ.get('step_ms', {}).get('p50')} text0={outs[0].outputs[0].text[:80]!r}"
            )
    for phase in ("c1", "c2"):
        effs = [r["eff_ms_per_tok"] for r in timing[phase] if r["eff_ms_per_tok"]]
        accs = [r["accept_rate"] for r in timing[phase] if r["accept_rate"] is not None]
        log(
            f"PHASE {phase} eff_ms_per_tok mean={statistics.fmean(effs) if effs else None} "
            f"median={statistics.median(effs) if effs else None} accept_mean={statistics.fmean(accs) if accs else None}"
        )
    try:
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                val = getattr(m, "value", None)
                if val is None:
                    val = getattr(m, "values", None)
                log(f"METRIC {m.name} {val}")
    except Exception as exc:  # noqa: BLE001
        log(f"metrics unavailable: {exc}")
    result["timing"] = timing
    with open(os.path.join(OUT, f"tokens-{tag}.json"), "w") as fh:
        json.dump(result, fh)
    if UT_SESSION:
        STEPS.clear()
        CUR.clear()
        STATE["record"] = True
        ut("resume")
        llm.generate([PROMPTS[0]], SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True), use_tqdm=False)
        torch.xpu.synchronize()
        ut("pause")
        STATE["record"] = False
        summ, _ = finish_steps("c1-traced")
        log("TRACED " + json.dumps(summ))
    log(f"DONE gates_ok={ok}")
    if UT_SESSION:
        import signal

        log("unitrace teardown via SIGUSR1")
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(120)
    _LOG.close()
    sys.stdout.flush()
    os._exit(0 if ok else 3)


if __name__ == "__main__":
    main()
