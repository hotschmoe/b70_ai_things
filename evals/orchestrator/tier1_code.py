"""Tier 1 — execution-graded code (HumanEval+ / MBPP+) via EvalPlus, sandboxed.

The pipeline is split so that the ONLY step which executes untrusted, model-generated code is
isolated inside a throwaway Docker container:

  1. GENERATE  (host, safe)  — our own OpenAI-client loop (reuses common.make_client), so Tier 1
                               gets the SAME determinism + thinking discipline as tiers 2/3:
                               greedy (temperature 0), fixed seed, concurrency 1, enable_thinking
                               off by default. Writes raw `<dataset>_raw.jsonl`.
  2. SANITIZE  (host, safe)  — `evalplus.sanitize` (tree-sitter text extraction, no code run) pulls
                               the clean function out of each markdown response → `*_sanitized.jsonl`.
  3. EVALUATE  (Docker)      — `evalplus.evaluate` RUNS the generated code against the +tests to get
                               pass@1. Sandboxed: --network none, non-root --user, a per-run throwaway
                               cache copy (can't poison ~/.cache/evalplus), memory/pids caps.

Why our own generator instead of `evalplus.codegen`: codegen's OpenAI provider can't pass
`chat_template_kwargs.enable_thinking`, so it would silently run thinking-ON and diverge from the rest
of the harness. We replicate EvalPlus's exact chat prompt (instruction_prefix + fenced code) so the
samples drop straight into EvalPlus's sanitize+evaluate.

Sandbox image: build once with `evals/sandbox/build.sh` (tag `evalplus-sandbox:<ver>`).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import RunContext, write_json

# EvalPlus's own default for instruct/chat models (evalplus/codegen.py). Kept verbatim so our
# pass@1 is comparable to EvalPlus's chat-backend numbers.
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)

# dataset -> (cache-file prefix, dataset loader). The cache prefix is how we find the staged
# dataset jsonl to trim for --limit smoke runs (file is e.g. HumanEvalPlus-v0.1.10.jsonl).
_DATASETS = {
    "humaneval": ("HumanEvalPlus", "get_human_eval_plus"),
    "mbpp": ("MbppPlus", "get_mbpp_plus"),
}


def run(
    ctx: RunContext,
    dataset: str = "humaneval",
    limit: int | None = None,
    think: bool = False,
    image: str = "evalplus-sandbox:0.3.1",
    allow_host_exec: bool = False,
) -> dict:
    dataset = dataset.lower()
    if dataset not in _DATASETS:
        return {"tier": 1, "skipped": True, "error": f"unknown dataset {dataset!r}; use humaneval|mbpp"}

    out_dir = ctx.out_dir / "tier1_evalplus"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- preflight: can we even grade? (do this BEFORE the expensive generation) ---
    grader = _pick_grader(image, allow_host_exec)
    if grader["error"]:
        return {"tier": 1, "dataset": dataset, "skipped": True, "error": grader["error"]}

    # 1. GENERATE (host) ------------------------------------------------------------------------
    t0 = time.time()
    problems = _load_problems(dataset)
    task_ids = sorted(problems, key=_task_sort_key)
    if limit:
        task_ids = task_ids[:limit]
    raw_path = out_dir / f"{dataset}_raw.jsonl"
    print(f"[tier1] generating {len(task_ids)} {dataset} solutions "
          f"(thinking={'on' if think else 'off'}, greedy, seed={ctx.sampling.get('seed')})")
    _generate(ctx, problems, task_ids, raw_path, think)
    gen_sec = time.time() - t0

    # 2. SANITIZE (host, safe text extraction) --------------------------------------------------
    san_path = _sanitize(raw_path)

    # 3. EVALUATE (sandboxed) -------------------------------------------------------------------
    t1 = time.time()
    proc = grader["run"](dataset, out_dir, san_path, limit)
    (out_dir / "evaluate.stdout.log").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    if proc.returncode != 0:
        return {"tier": 1, "dataset": dataset, "mode": grader["mode"],
                "error": f"evaluate exit {proc.returncode}; see tier1_evalplus/evaluate.stdout.log",
                "n_problems": len(task_ids)}
    scores = _parse_pass1(proc.stdout)
    eval_sec = time.time() - t1

    result = {
        "tier": 1, "dataset": dataset, "mode": grader["mode"], "thinking": think,
        "n_problems": len(task_ids), "limit": limit,
        "pass@1": scores,            # {"base": x, "plus": y}  (plus = base + extra tests)
        "image": image if grader["mode"] == "sandbox-docker" else None,
        "raw_samples": str(raw_path), "sanitized_samples": str(san_path),
        "eval_results": str(san_path).replace(".jsonl", "_eval_results.json"),
        "gen_sec": round(gen_sec, 1), "eval_sec": round(eval_sec, 1),
    }
    write_json(ctx.out_dir / "tier1_code.json", result)
    print(f"[tier1] {dataset} pass@1={scores} (gen {gen_sec:.0f}s, eval {eval_sec:.0f}s, {grader['mode']})")
    return result


# ----------------------------------------------------------------------------- generation
def _load_problems(dataset: str) -> dict:
    import evalplus.data as ed
    _, loader = _DATASETS[dataset]
    return getattr(ed, loader)()


def _task_sort_key(task_id: str):
    # "HumanEval/12" / "Mbpp/427" -> numeric order, not lexical
    try:
        return (task_id.split("/")[0], int(task_id.split("/")[1]))
    except (IndexError, ValueError):
        return (task_id, 0)


def _generate(ctx: RunContext, problems: dict, task_ids: list[str], raw_path: Path, think: bool) -> None:
    client = _make_client(ctx.endpoint)
    s = ctx.sampling
    extra_body = {"chat_template_kwargs": {"enable_thinking": bool(think)}}
    with open(raw_path, "w") as fh:
        for i, tid in enumerate(task_ids, 1):
            prompt = problems[tid]["prompt"]
            message = INSTRUCTION_PREFIX + f"\n```python\n{prompt.strip()}\n```"
            resp = client.chat.completions.create(
                model=ctx.model_id,
                messages=[{"role": "user", "content": message}],
                temperature=s.get("temperature", 0.0),
                top_p=s.get("top_p", 1.0),
                seed=s.get("seed"),
                max_tokens=s.get("max_tokens", 2048),
                extra_body=extra_body,
            )
            solution = resp.choices[0].message.content or ""
            fh.write(json.dumps({"task_id": tid, "solution": solution}) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(task_ids):
                print(f"[tier1]   generated {i}/{len(task_ids)}")


def _make_client(endpoint: str):
    # local import keeps `import common` cheap and mirrors common.make_client wiring
    from common import make_client
    return make_client(endpoint)


# ----------------------------------------------------------------------------- sanitize (host)
def _sanitize(raw_path: Path) -> Path:
    binary = shutil.which("evalplus.sanitize")
    # sys.executable is THIS interpreter (the venv python with evalplus installed), robust even when
    # the venv bin dir isn't on PATH (we're usually invoked as `.venv/bin/python ...`, not activated).
    cmd = [binary, str(raw_path)] if binary else [sys.executable, "-m", "evalplus.sanitize", str(raw_path)]
    print(f"[tier1] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"evalplus.sanitize failed: {proc.stderr[-500:]}")
    san = raw_path.with_name(raw_path.stem + "-sanitized.jsonl")
    if not san.exists():
        raise RuntimeError(f"sanitize produced no output at {san}")
    return san


# ----------------------------------------------------------------------------- grading backends
def _pick_grader(image: str, allow_host_exec: bool) -> dict:
    """Return {mode, run(dataset,out_dir,samples,limit)->CompletedProcess, error}."""
    if shutil.which("docker") and _image_present(image):
        return {"mode": "sandbox-docker", "error": None,
                "run": lambda ds, od, s, lim: _evaluate_docker(image, ds, od, s, lim)}
    if allow_host_exec:
        print("[tier1] WARNING: running evaluate on the HOST (unsandboxed). Trusted env only.")
        return {"mode": "host-unsandboxed", "error": None,
                "run": lambda ds, od, s, lim: _evaluate_host(ds, s)}
    if shutil.which("docker") and not _image_present(image):
        return {"mode": None, "run": None,
                "error": f"sandbox image '{image}' not built. Run: bash evals/sandbox/build.sh "
                         f"(or pass --allow-code-exec to grade on the host, UNSANDBOXED)."}
    return {"mode": None, "run": None,
            "error": "no docker for the sandbox and --allow-code-exec not set. "
                     "Tier 1 executes generated code; refusing to run ungated."}


def _image_present(image: str) -> bool:
    r = subprocess.run(["docker", "image", "inspect", image],
                       capture_output=True, text=True)
    return r.returncode == 0


def _evaluate_docker(image: str, dataset: str, out_dir: Path, samples: Path, limit: int | None):
    """Grade inside the sandbox: --network none, non-root, throwaway writable cache, caps."""
    from evalplus.data.utils import CACHE_DIR  # host cache populated by generation/sanitize

    # Throwaway cache copy so untrusted code can't touch the real ~/.cache/evalplus, yet the
    # ground-truth .pkl that evaluate writes still has somewhere to land (it can't re-download
    # under --network none).
    stage = out_dir / "_sandbox_cache"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "evalplus").mkdir(parents=True)
    if Path(CACHE_DIR).exists():
        shutil.copytree(CACHE_DIR, stage / "evalplus", dirs_exist_ok=True)

    # Smoke/dev: --limit generated only a subset, but evaluate asserts FULL dataset coverage.
    # Trim the staged (throwaway) dataset to exactly the sampled task_ids so coverage matches.
    # Only when limit is set — full runs stay unfiltered so the assertion still catches drops.
    if limit:
        _trim_staged_dataset(stage / "evalplus", dataset, samples)

    uid, gid = os.getuid(), os.getgid()
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--user", f"{uid}:{gid}",
        "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/xdgcache",
        "--memory", "8g", "--pids-limit", "512",
        "-v", f"{out_dir.resolve()}:/work",
        "-v", f"{stage.resolve()}:/xdgcache",
        image,
        "evalplus.evaluate", dataset,
        "--samples", f"/work/{samples.name}",
        "--i_just_wanna_run",
    ]
    print(f"[tier1] $ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)


def _trim_staged_dataset(cache_dir: Path, dataset: str, samples: Path) -> None:
    """Rewrite the staged dataset jsonl to only the task_ids present in `samples` (smoke runs)."""
    sample_ids = {json.loads(ln)["task_id"] for ln in samples.read_text().splitlines() if ln.strip()}
    prefix = _DATASETS[dataset][0]
    hits = list(cache_dir.glob(f"{prefix}*.jsonl"))
    if not hits:
        print(f"[tier1] WARN: no staged {prefix}*.jsonl to trim; evaluate may assert full coverage")
        return
    for ds_file in hits:
        kept = [ln for ln in ds_file.read_text().splitlines()
                if ln.strip() and json.loads(ln)["task_id"] in sample_ids]
        ds_file.write_text("\n".join(kept) + "\n")
    print(f"[tier1] trimmed staged {prefix} dataset -> {len(sample_ids)} task(s) for limited run")


def _evaluate_host(dataset: str, samples: Path):
    binary = shutil.which("evalplus.evaluate")
    cmd = ([binary] if binary else [sys.executable, "-m", "evalplus.evaluate"]) + [
        dataset, "--samples", str(samples), "--i_just_wanna_run"]
    print(f"[tier1] $ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)


# ----------------------------------------------------------------------------- parse
def _parse_pass1(stdout: str) -> dict:
    """EvalPlus prints a header line then a `pass@1` line, e.g.:
        humaneval (base tests)
        pass@1:	0.713
        humaneval+ (base + extra tests)
        pass@1:	0.665
    Track the section header to label base vs plus.
    """
    out: dict = {}
    section = None
    for line in stdout.splitlines():
        s = line.strip().lower()
        if s.startswith(("humaneval", "mbpp")):
            section = "plus" if ("+" in s or "extra" in s or "plus" in s) else "base"
        if "pass@1" in s:
            try:
                val = float(s.split("pass@1")[1].lstrip(": \t").split()[0])
            except (IndexError, ValueError):
                continue
            out[section or "base"] = val
    return out
