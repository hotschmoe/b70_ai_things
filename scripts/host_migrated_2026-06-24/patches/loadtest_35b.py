import os, time
from vllm import LLM, SamplingParams

def main():
    m = os.environ.get("MODEL")
    t0 = time.time()
    llm = LLM(
        model=m, trust_remote_code=True,
        max_model_len=int(os.environ.get("MAXLEN", "2048")),
        max_num_seqs=1,
        max_num_batched_tokens=int(os.environ.get("MAXBATCH", "2048")),
        gpu_memory_utilization=float(os.environ.get("UTIL", "0.95")),
        enforce_eager=True,
    )
    print(f"=== LLM constructed in {time.time()-t0:.0f}s ===", flush=True)
    o = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=24))
    print("=== GENERATION OK ===", flush=True)
    print("OUTPUT:", repr(o[0].outputs[0].text), flush=True)

if __name__ == "__main__":
    main()
