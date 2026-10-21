#!/usr/bin/env python3
"""O5: locate the first batch-variant module in the Flash-Next target.

Eager (cudagraph NONE) in-process load, same recipe as o5_mtp_profile.py.
Forward hooks on every target-model module record a hash of request 0's
rows (and of row 0 alone) per step. Runs:

  solo:  generate([A], max_tokens=N)        decode T = 1 (or 1+k with MTP)
  dup:   generate([A, A], max_tokens=N)     decode T = 2 (or 2+2k)
  trip:  generate([A, A, A], max_tokens=N)  decode T = 3

then reports, per pair, the first (step, module) whose request-0 hash
differs. hashes-<tag>.json keeps row-0 hashes keyed by row-0 position for
cross-arm comparison (MTP on vs off): o5_invar_compare.py.

Env: O5_OUT, O5_TAG, O5_MTP (k), O5_PROBE_TOKENS (default 4).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
OUT = os.environ["O5_OUT"]
os.makedirs(OUT, exist_ok=True)
_LOG = open(os.path.join(OUT, "probe.log"), "a", encoding="utf-8")


def log(msg: str) -> None:
    line = f"O5 {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    _LOG.write(line + "\n")
    _LOG.flush()


import torch  # noqa: E402

REC: dict = {"infwd": False, "on": False, "step": -1, "rows0": 0, "ntok": 0, "calls": [], "pos0": None}


def _h(t: torch.Tensor) -> str:
    return hashlib.md5(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def install() -> None:
    from vllm.v1.worker.gpu import model_runner as mr

    R = mr.GPUModelRunner
    orig_load = R.load_model

    def load_model(self, *a, **k):
        out = orig_load(self, *a, **k)
        names = 0
        for name, mod in self.model.named_modules():
            if not name:
                continue
            mod.register_forward_hook(_make_hook(name))
            names += 1
        top = self.model

        def post(mod, args, out):
            REC["infwd"] = False

        def pre(mod, args, kwargs):
            REC["infwd"] = True
            if REC["on"]:
                pos = kwargs.get("positions")
                if pos is None and len(args) > 1:
                    pos = args[1]
                if isinstance(pos, torch.Tensor) and pos.numel():
                    REC["pos0"] = int(pos.reshape(-1)[0].item()) if pos.dim() == 1 else int(pos[..., 0].reshape(-1)[0].item())
            return None

        top.register_forward_pre_hook(pre, with_kwargs=True)
        top.register_forward_hook(post)
        log(f"hooks on {names} target modules")
        return out

    R.load_model = load_model

    orig_exec = R.execute_model

    def execute_model(self, so, *a, **k):
        if REC["on"] and not k.get("dummy_run"):
            REC["step"] += 1
            nt = int(getattr(so, "total_num_scheduled_tokens", 0))
            per = dict(getattr(so, "num_scheduled_tokens", {}) or {})
            nreq = max(1, len(per))
            REC["ntok"] = nt
            # Identical prompts: request 0 owns the first nt/nreq rows.
            REC["rows0"] = nt // nreq
            REC["nreq"] = nreq
        return orig_exec(self, so, *a, **k)

    R.execute_model = execute_model


def _make_hook(name: str):
    def hook(mod, inp, out):
        if not REC["on"] or not REC["infwd"]:
            return
        t = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(t, torch.Tensor) or t.dim() == 0:
            return
        nt = REC["ntok"]
        if t.shape[0] != nt or nt == 0:
            return
        REC["calls"].append(
            {
                "step": REC["step"],
                "nt": nt,
                "nreq": REC["nreq"],
                "pos0": REC["pos0"],
                "mod": name,
                "h0": _h(t[:1]) + (
                    _h(out[1][:1])
                    if isinstance(out, (tuple, list)) and len(out) > 1 and isinstance(out[1], torch.Tensor)
                    and out[1].dim() and out[1].shape[0] == nt
                    else ""
                ),
                "hin": _h(inp[0][:1]) if (inp and isinstance(inp[0], torch.Tensor) and inp[0].dim() and inp[0].shape[0] == nt) else None,
                "hreq": _h(t[: REC["rows0"]]),
            }
        )

    return hook


def run(llm, sp, prompts, tag):
    REC.update(on=True, step=-1, calls=[], pos0=None)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    torch.xpu.synchronize()
    REC["on"] = False
    calls = list(REC["calls"])
    ids = [list(o.outputs[0].token_ids) for o in outs]
    log(f"{tag}: {len(calls)} module records, ids0={ids[0]}")
    return calls, ids


def first_diff(a, b, key):
    # Walk both call lists in order per step; report first hreq mismatch.
    from collections import defaultdict

    da, db = defaultdict(list), defaultdict(list)
    for c in a:
        da[c["step"]].append(c)
    for c in b:
        db[c["step"]].append(c)
    for s in sorted(set(da) & set(db)):
        for i, (x, y) in enumerate(zip(da[s], db[s])):
            if x["mod"] != y["mod"]:
                return {"step": s, "i": i, "why": "module order", "a": x["mod"], "b": y["mod"]}
            if x[key] != y[key]:
                return {"step": s, "i": i, "mod": x["mod"], "nt": (x["nt"], y["nt"]), "pos0": x["pos0"]}
    return None


def main() -> None:
    k = int(os.environ.get("O5_MTP", "0"))
    tag = os.environ.get("O5_TAG", f"probe{k}")
    ntok = int(os.environ.get("O5_PROBE_TOKENS", "4"))
    install()
    from vllm import LLM, SamplingParams

    hf = "/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext"
    kwargs = dict(
        model=hf,
        served_model_name=["hotschmoe-dev", f"qwen3.8-flash-next-iq3-xxs-sidecar-probe{k}"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.98,
        language_model_only=True,
        load_format="b70_gguf",
        quantization="b70_gguf",
        trust_remote_code=True,
        attention_backend="FLASH_ATTN",
        enable_prefix_caching=False,
        mamba_cache_mode="align",
        enforce_eager=True,
    )
    if k:
        kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": k}
    log(f"tag={tag} k={k}")
    llm = LLM(**kwargs)
    log("LLM ready")
    A = os.environ.get("O5_PROBE_PROMPT", "Write a short story about a lighthouse keeper who discovers a message in a bottle.")
    sp = SamplingParams(temperature=0.0, max_tokens=ntok, ignore_eos=True)
    llm.generate([A], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), use_tqdm=False)
    res = {}
    for name, prompts in (("solo", [A]), ("dup", [A, A]), ("trip", [A, A, A])):
        res[name] = run(llm, sp, prompts, name)
    for x, y in (("solo", "dup"), ("solo", "trip"), ("dup", "trip")):
        d = first_diff(res[x][0], res[y][0], "hreq")
        d0 = first_diff(res[x][0], res[y][0], "h0")
        log(f"FIRSTDIFF {x} vs {y}: req={json.dumps(d)} row0={json.dumps(d0)}")
    with open(os.path.join(OUT, f"hashes-{tag}.json"), "w") as fh:
        json.dump({n: {"calls": c, "ids": i} for n, (c, i) in res.items()}, fh)
    log("DONE")
    _LOG.close()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
