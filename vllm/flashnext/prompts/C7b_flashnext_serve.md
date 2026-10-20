You are grok agent C7b, continuing C7 (fork owner). Read vllm/flashnext/BRIEF.md first and obey it (ASCII, bin/gpu-run, bin/cpu-run, journal entries CONFIG->COMMAND->RESULT->VERDICT appended at the bottom of vllm/flashnext/JOURNAL.md, agents do not commit). Then read reports/1C7_gguf_graph_ep.md and the 11:32 C7 JOURNAL entries.

You have priority on BOTH cards now. As of 14:35 both cards are free and MemAvailable is ~112 GiB. Every other GPU agent has finished. hotschmoe-dd is stopped; do not start it.

Goal: the first coherent Flash-Next IQ3_XXS serve on vLLM.
1. Run the prepared window: bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-fn-window.sh (DP=2 EP eager, max_model_len 2048). Watch host memory. Greedy targets are the Strata strings: "17*19" -> 323, "23*29" -> 667. Record worker RSS after load. Strata held this model in ~16 GB host RAM, so explain any big excess.
2. If it is coherent, move to graphs: FULL_DECODE_ONLY + FLASH_ATTN + B70_XPU_CG_RECLAIM=500, using the C12 _C.graph.so. Measure C1/C2 ITL with harness/bench_latency.py under cpu-run --quiet.
3. If it is not coherent, bisect: per-layer dumps vs the Strata reference (harness/strata_reference.json, flashnext_ref/). Shy away from nothing. Missing kernels or features get built.
4. Keep max_model_len small until the coherence gate passes. Then try 32K so the H1d EP sidecar agent and the G4 KV-parking agent have a Flash-Next serve recipe.

Run discipline: start long GPU work detached and poll it. Do not end your run while a job you own is still running. When you finish, write reports/1C7b_flashnext_serve.md with the exact serve recipe (script path), identity, coherence text, ITL, host RSS, and health before/after. Other agents (H1d) will queue for the cards after you, so release both cards when you are done.
