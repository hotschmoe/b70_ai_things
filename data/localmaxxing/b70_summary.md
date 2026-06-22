# localmaxxing.com -- Intel Arc Pro B70 community benchmarks

Auto-pulled from `GET /api/benchmarks?gpuName=Intel Arc Pro B70` (best output tok/s per model/engine/quant/gpuCount). Regenerate with `python3 scripts/75_localmaxxing.py save`.

| Model | Engine | Quant | GPUs | tok/s out | TTFT ms | By |
|---|---|---|---|---|---|---|
| Intel/gemma-4-12B-it-int4-AutoRound | vllm | INT4 AutoRound W4A16 | x4 | 796.2 | 2530 | steveseguin |
| Jackrong/Qwopus3.6-35B-A3B-v1-MTP-GGUF | llama.cpp | Q4_K_M | x1 | 107.8 | - | Beezzle |
| nex-agi/Nex-N2-mini | llama.cpp | Q4_K_M | x1 | 106.5 | - | carolus22 |
| Qwen/Qwen3.6-35B-A3B | vllm | BF16 | x4 | 102.5 | 104 | RagingNoper |
| Qwen/Qwen3.6-35B-A3B | vllm | Quark W8A8 INT8 | x4 | 99.8 | 77 | steveseguin |
| nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8 | vllm | Quark W8A8 INT8 | x4 | 99.4 | 76 | steveseguin |
| Lasimeri/MiniMax-M2.7-int4-AutoRound | vllm | INT4 AutoRound W4A16 | x4 | 94.5 | - | steveseguin |
| Lasimeri/MiniMax-M2.7-int4-AutoRound | vllm | AutoRound INT4 W4A16 | x4 | 93.4 | - | steveseguin |
| MJPansa/MiniMax-M2.7-REAP-172B-A10B-AutoRound-W4A16 | vllm | AutoRound INT4 W4A16 | x4 | 89.5 | - | steveseguin |
| Qwen/Qwen3-Coder-Next | vllm | FP8 | x4 | 71.7 | 95 | RagingNoper |
| Qwen/Qwen3.6-35B-A3B | llama.cpp | Q4_K_M | x1 | 70.3 | 203 | wirbol |
| Qwen/Qwen3.6-35B-A3B | llama.cpp | UD-Q4_K_M | x4 | 68.8 | - | ytnszmy |
| Lasimeri/MiniMax-M2.7-int4-AutoRound | vllm | AutoRound W4A16 | x4 | 61.8 | - | steveseguin |
| Qwen/Qwen3.6-27B | vllm | fp16 | x4 | 54.2 | 79 | ytnszmy |
| vrfai/Qwen3.6-27B-FP8 | vllm | fp8 | x4 | 49.6 | - | steveseguin |
| unsloth/Qwen3.6-27B | llama.cpp | Q4_0 | x3 | 49.4 | - | steveseguin |
| Qwen/Qwen3.6-27B | llama.cpp | Q4_0 | x3 | 46.1 | - | steveseguin |
| Qwen/Qwen3.6-27B | vllm | fp8 | x4 | 43.7 | - | steveseguin |
| unsloth/Qwen3.6-27B | llama.cpp | Q4_0 | x2 | 42.1 | - | steveseguin |
| unsloth/Qwen3.6-27B-GGUF | llama.cpp | Q4_K_M | x1 | 42.0 | - | bassmaster187 |
| unsloth/Qwen3.6-27B | llama.cpp | Q4_0 | x4 | 39.2 | - | steveseguin |
| Qwen/Qwen3.6-27B | llama.cpp | Q4_K_M | x1 | 30.0 | 3030 | bassmaster187 |
| AaryanK/Qwen3.6-27B-GGUF | llama.cpp | Q4_0 | x1 | 28.8 | - | steveseguin |
| z-lab/Qwen3.6-27B-DFlash | llama.cpp | Q4_0 | x2 | 26.9 | - | steveseguin |
| unsloth/Qwen3.6-27B-GGUF | llama.cpp | Q4_K_S | x1 | 26.7 | - | steveseguin |
| unsloth/Qwen3.6-27B-GGUF | llama.cpp | IQ4_NL | x1 | 26.0 | - | steveseguin |
| unsloth/Qwen3.6-27B-GGUF | llama.cpp | Q4_0 | x1 | 25.2 | - | steveseguin |
| vrfai/Qwen3.6-27B-FP8 | vllm | FP8 compressed-tensors | x4 | 22.7 | - | steveseguin |
| stepfun-ai/Step-3.7-Flash-GGUF | llama.cpp | UD-IQ4_XS | x4 | 20.8 | - | ytnszmy |
| MiniMaxAI/MiniMax-M2.7 | vllm | INT4 AutoRound W4A16 | x4 | 20.1 | - | steveseguin |
| Qwen/Qwen3.5-27B | llama.cpp | Q4_K_XL | x1 | 18.0 | 486 | wirbol |
| MiniMaxAI/MiniMax-M2.7 | llama.cpp | UD-IQ4_XS | x4 | 17.7 | - | steveseguin |
| google/gemma-4-31B-it | llama.cpp | Q4_K_M | x1 | 16.1 | - | snnn |
