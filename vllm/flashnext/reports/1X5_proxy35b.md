# X5 -- Qwen3.6-35B-A3B proxy best-stack comparison

In progress. The eventless stock proxy completes the short-context
identity/coherence/exact-repeat/health gates. No localmaxxing result
has been submitted or fully qualified.
Non-speculative decode is the headline. Flash-Next MTP remains with O5;
X1's implementation and findings are in reports/1X1_mtp.md.

Raw evidence: /mnt/vm_8tb/b70/build/vllm-flashnext/x5/.

## Recipe and identity

`harness/x5_serve_proxy.sh` selects QUANT, MTP_TOKENS and CARD (default1),
and calls xpu-run.sh, which leases and pins the physical card. Primary
served name is hotschmoe-dd, followed by a quant/depth-specific research
alias registered in evals/configs/models.yaml. TP1/world1, BF16,
FULL_DECODE_ONLY, FLASH_ATTN, no prefix cache, maxlen4096, maxseq2,
batch1024, GPUutil0.90, B70_FUSE_SHARED=1. The retained control uses
B70_X5_EVENTLESS=1 and reclaim0; the original reclaim500 arm stalled.

MoE library: _C.o3g.so SHA256
38f7176ac290cb264268185c864da2dea1fcc36d07535fff021fb636d6538dcc.
Stock installed attention is required for this proxy; the Flash-Next
c7b overlay is not its matching binary/wrapper pair.

HF config/tokenizer: /mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B,
revision995ad96eacd98c81ed38be0c5b274b04031597b0.
GGUFs: unsloth/Qwen3.6-35B-A3B-MTP-GGUF,
revision5bc3e238d916f48a861bac2f8a1990a0e9b7e98d.
Provenance is in devmodel/MANIFEST.txt; header-only inventory is
x5/quant-inventory.json. Each file has753tensors and20MTP tensors
under blk.40, with nextn_predict_layers1. This is the official proxy,
not the Ornith/Shisa finetune.

| Format | File bytes | Trunk | Routed gate/up | Routed down |
| --- | ---: | --- | --- | --- |
| UD-IQ3_XXS | 14069266720 | mostly Q6_K | mostly IQ2_S | mostly IQ3_S |
| UD-IQ4_XS | 18209036576 | mostly Q8_0 | mostly IQ3_S | mostly IQ4_XS |
| UD-Q4_K_M | 22663387424 | mostly Q8_0 | mostly Q4_K | Q5_K/Q4_K |

These are dynamic mixed formats. A latency difference cannot be
attributed solely to expert bits. Output head is Q6_K in all three.

## Completed results so far

All numbers below are card1 only, no MTP, no prefix cache, maxlen4096.
ITL columns are SSE inter-chunk p50, not MTP effective token latency.

| Format / path | C1 p50 ms | C2 p50 ms/stream | Prefix PPL | Repeat | Health |
| --- | ---: | --- | ---: | --- | --- |
| IQ3, stock GDN, eventless | 8.83 | 11.14 / 11.02 | 9.09554 | exact C1+C2 | before/after HEALTHY |

C1 p90/p99=9.04/9.17ms,115.11tok/s,63completion tokens through EOS.
C2 stream0 p99=71.60ms includes a concurrency-admission/prefill gap;
this is not a steady-tail qualification. Eventless output also matches
all initial ordinary-control C1/C2 token IDs from before that control's
stall. Before/after source hashes match in the retained eventless arm.

The ordinary reclaim500 arm passed the initial6-request greedy gate.
X4 then completed4x96tokens in its first C1 round and timed out60s in
the second round. X5 repeat made no progress either. The owned server
and master were stopped, and card1 post-health passed. There is no
completed benchmark/PPL for that arm. No stack was captured: py-spy
was unavailable at the attempted path. Periodic graph replay is a
hypothesis; the stall is not proven to occur at the500 boundary.

The new eventless arm uses X1's Torch08187d9e shim, SHA256
b82405b40a13457c5d6e6b19c7a44d19988cf0b7e981cccb7ea181cb249a7cc3,
whose2066-replay/RNG primitive check is recorded in1X1_mtp.md.
`x5_eventless_serve.sh` primes both Qwen3.5 target/draft inspection
caches before preload, then disables periodic re-instantiation.
Both priming lines and X1_EVENTLESS_REPLAY_ACTIVE appear in the worker
log. The completed finite window is not a long-duration stability claim.

## BF16 D3 port

The existing d3_gdn.py guard requires fp16 conv state and activations;
setting B70_D3_GDN=1 on the BF16 GGUF recipe falls back to stock GDN.
Switching only the model dtype is not a clean fix: the packed embedding
and GEMV interface produce/consume BF16.

`harness/x5_build_gdn.sh` makes an isolated BF16 instantiation of D3,
replacing scalar storage/rounding types and keeping its geometry/math.
No shared D3 source or existing library was changed. CPU-only build
under cpu-run, MAX_JOBS4,60s, rc0. Library SHA256
2bb3661f2b445ba57b3af0db5dbca192550bb22ebefac24d2a7f0a1953aae328.
`x5_overlay/sitecustomize.py` changes only the corresponding dtype
guards and is opt-in through B70_X5_GDN_BF16=1. The original FP16 library
is not loaded in that process. Spec verification stays on stock GDN.
First numerical probe passed core/output/conv checks but failed12state
values at a2e-5 absolute bound (largest failing delta3.974745e-5).
That failure and the healthy teardown are retained. The same binary
is queued with1e-4absolute/0.005relative state tolerance, consistent
with the prior D3 FP16 state's reported max5.85e-5. Convolution and
32-replay equality stay exact; model greedy gates are unchanged.
R2 passed3seeds and32replays each: convolution and replay output exact,
core max3.05e-5..6.10e-5, output max0.015625 and cosine above0.999998.
SSM overall max1.50e-4..2.45e-4, mean1.31e-6..1.60e-6, cosine1.0.
The assert uses absolute PLUS relative tolerance, so1e-4 is not an
absolute maximum-error claim. Pinned post-health is HEALTHY.
This is approximate primitive equivalence, not bit equality to stock.
The stock baseline disables the old D3 wrapper: its BF16 rejection
occurs after projections, so a fallback would duplicate qkvz/ba work.

## Measurement gates

`x5_proxy_window.sh` records pinned health before/after, live identity,
four greedy C1 prompts and two C2 prompts, same-load exact repeat,
coherence, C1/C2 SSE latency, prompt-logprob quality, and MTP acceptance.
Token comparisons respect configured EOS. MTP compares with the same
quant's no-MTP token baseline. Timing clients hold CPUquiet; no quiet
lock is held while waiting for a GPU. A watchdog stops the owned server
below8GiB MemAvailable. Every arm tears down before releasing its job.

`x5_prepare_quality.py` caches the revision-pinned raw test parquet from
[Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext),
revisionb08601e04326c79dfdd32d625aee71d232d685c3, records parquet
SHA2565f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91
and the proxy tokenizer SHA256, and selects the first
2048 tokens without adding special tokens. `x5_quality.py` scores four
independent512-token chunks using actual prompt-token log probabilities,
excluding the unscored first token of each chunk (2044 scored tokens).
The result is exp(mean NLL). This is a matched test-prefix diagnostic,
not full WikiText-2 test perplexity. Four original prose/code passages
remain as a fallback only; differing corpus hashes invalidate a matched
comparison. Perplexity is measured only without MTP.

SSE inter-chunk ITL and effective ms/token must be distinguished for
MTP: one chunk can contain multiple accepted tokens. The table will use
effective latency for MTP and include wire-chunk timing separately.

## Failures and decisions

CONFIG: initial IQ3 stock-BF16 arm with the Flash-Next c7b FA overlay.
COMMAND: x5_proxy_window.sh; raw ud-iq3_xxs-mtp0/.
RESULT:733tensors loaded, then graph capture failed because that overlay
lacks paged config8,256,64,false,false,false. Its reference fallback
attempted D2H during capture, raising UNSUPPORTED_FEATURE. No API
readiness or token/latency result. Teardown and pinned post-health passed.
VERDICT: remove that overlay from X5 and use O3's installed FA path.
No device loss or GPU reset.

Claude consults requested fable-5.1 and claude-5.5; both returned
unrecognized_model and Not logged in. There was no responding model ID
and no independent model review. No commits or DD service actions.

## Requests

X5 needs a bounded card1 window after current O4/O5/F1 priority work.
Stock proxy body is about2min plus health. Trace and every subsequent
quant arm release the lease independently. CPU preparation is done,
and the master does not hold CPUquiet while deferred or queued.
