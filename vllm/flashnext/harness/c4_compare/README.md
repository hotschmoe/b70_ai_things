# C4 comparison, 2026-10-21

One shared HTTP workload for the retained Strata Flash-Next, vLLM
Flash-Next IQ3_XXS, and vLLM Qwen3.8-27B AutoRound INT4 recipes.
Results and fixtures: `/mnt/vm_8tb/b70/results/c4_compare_20261021/`.

`prepare.py` uses the campaign's CPU tokenizer to freeze four coding-review
prompts per batch, 19787 content tokens each. It refuses an existing fixture.
All endpoints receive the same messages, greedy/thinking-off, output512.
One full warmup is excluded; two cold batches are measured. Early differing
prefixes prevent prefix-cache reuse; server usage verifies input/output and
cache counts. No `ignore_eos` override differs between backends.

Run a finite arm with a fresh descriptive tag, for example:

```sh
bin/cpu-run --quiet python3 vllm/flashnext/harness/c4_compare/run.py \
  flashnext --tag vllm-qwen38-flashnext-ista-iq3xxs-mtp1-fresh
```

The launcher holds a quiet CPU window, uses the existing GPU lease launchers,
records model/config identity, checks arithmetic, captures raw SSE/text/usage,
stops its owned server, and checks per-card and compiled two-rank health.
No DD service changes, runtime builds, shelf promotion or external submission.
Run arms sequentially; inspect closure before launching the next one.

Effective PP is uncached input divided by time until every stream's first
content. Burst TG is output-minus-first divided by earliest-first-content
through last-response-end. Both include scheduling and overlapping work;
total batch wall is the direct cold-workload comparison.

`analyze.py` additionally counts output-token arrivals only while all four
streams are decoding, after all first content and before any last content.
It uses the corresponding tokenizer and requires reencoded token count to
equal server usage for every stream. This is an HTTP arrival measurement,
not a GPU event timer or a proof of generated token-ID equality.

Strata's cold512 burst had no such four-way interval. The separate
`strata-warm-fixture.json` primes the original four warmup prompts, then
replays them twice with cache fraction >99% required. It supplies only the
missing steady TG observation; retain the original cold PP/wall results.
Use `run.py strata --tag <fresh> --fixture <warm-fixture>` for that supplement.
