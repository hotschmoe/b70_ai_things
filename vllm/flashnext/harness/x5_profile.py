"""Single-card proxy timeline; uses O4's runner/event instrumentation."""
import json
import os
import signal
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

if any(os.environ.get(k) == "1" for k in ("B70_X5_GDN_BF16", "B70_X5_PACK_LINEAR", "B70_X5_Q8_LINEAR", "B70_X5_EVENTLESS_LAZY", "B70_X5_CANONICAL", "B70_X5_BATCH_LINEAR", "B70_X5_Q8_FAST")):
    from x5_api import bootstrap

    bootstrap()

import o4_decode_profile as prof
import torch
from vllm import LLM, SamplingParams


def main():
    torch.xpu.set_device(0)
    prof.install_hooks()
    llm = LLM(
        model="/mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B",
        served_model_name=["hotschmoe-dd", os.environ["X5_ALIAS"]],
        tensor_parallel_size=1, dtype="bfloat16", language_model_only=True,
        max_model_len=4096, max_num_seqs=2, max_num_batched_tokens=1024,
        gpu_memory_utilization=0.90, load_format="b70_gguf", quantization="b70_gguf",
        trust_remote_code=True, attention_backend="FLASH_ATTN", enable_prefix_caching=False,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": [1, 2, 3, 4, 6, 8]},
    )
    prof.ut("pause")
    open(prof.OUT + "/profile-ready", "w").close()
    deadline = time.monotonic() + 7200
    while not os.path.exists(prof.OUT + "/profile-go"):
        if time.monotonic() > deadline:
            raise TimeoutError("CPU quiet trigger did not arrive")
        time.sleep(0.2)
    gates = []
    for question, expected in [("17 * 19", "323"), ("23 * 29", "667")]:
        out = llm.chat([{"role": "user", "content": "Return only the integer: " + question}],
                       SamplingParams(temperature=0, max_tokens=16),
                       chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        text = out[0].outputs[0].text
        gates.append(dict(text=text, expected=expected, passed=text.strip() == expected))
    open(prof.OUT + "/gates.json", "w").write(json.dumps(gates, indent=2))
    prompt = "Write a long, detailed essay about the history of the city of Paris."
    llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
    for tag, batch, ntok in [("c1", 1, 96), ("c2", 2, 96), ("c1-traced", 1, 24)]:
        if tag == "c1-traced" and not prof.UT_SESSION:
            continue
        prof.STEPS.clear()
        prof.STATE.update(last_end=None, record=True)
        if tag == "c1-traced":
            prof.ut("resume")
        out = llm.generate([prompt] * batch,
                           SamplingParams(temperature=0, max_tokens=ntok, ignore_eos=True),
                           use_tqdm=False)
        torch.xpu.synchronize()
        if tag == "c1-traced":
            prof.ut("pause")
        prof.STATE["record"] = False
        summary = prof.finish_steps(tag, list(prof.STEPS))
        open(prof.OUT + "/summary-" + tag + ".json", "w").write(json.dumps(summary, indent=2))
        ids = [list(o.outputs[0].token_ids) for o in out]
        open(prof.OUT + "/tokens-" + tag + ".json", "w").write(json.dumps(ids))
    c1 = json.load(open(prof.OUT + "/tokens-c1.json"))[0]
    c2 = json.load(open(prof.OUT + "/tokens-c2.json"))
    batch_parity = [dict(identical=row == c1, first_mismatch=next(
        (i for i, (x, y) in enumerate(zip(c1, row)) if x != y), None)) for row in c2]
    open(prof.OUT + "/batch-parity.json", "w").write(json.dumps(batch_parity, indent=2))
    print("X5_PROFILE_DONE", flush=True)
    if prof.UT_SESSION:
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(30)
    open(prof.OUT + "/profile-done", "w").close()
    os._exit(0 if all(g["passed"] for g in gates) else 3)


if __name__ == "__main__":
    main()
