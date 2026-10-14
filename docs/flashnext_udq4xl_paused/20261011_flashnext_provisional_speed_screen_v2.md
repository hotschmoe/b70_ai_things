# Provisional one/pair client speed screen

CONFIG

Use new unlaunched C1401403 segmented prepared directories. The original C140
controller validates current source/model/config and owns foreground launch,
exact named stop, native clean exit and removal. The new diagnostic parent owns
the pair lease, strict per-card and compiled P2P0 health, original page guards,
kernel journals, and a new full four-shard byte identity after terminal and
post-health. It runs the original small C140 screen before timed requests, retaining
its repeat/token/coherence evidence, but does not call original finalize or publish
an original C140 qualification: extra speed requests violate the fixed trace roster.

The same two useful water-cycle messages run short/long/long/short, temperature0,
seed1, fresh=true, cacheOFF, max_tokens192, natural EOS, thinking disabled.
The longer repeated factual context targets about512 tokens; actual final API
usage is authoritative and the report records whether it falls within384..768.
No inference is used just to tune the prompt length. Short natural EOS output
below128 remains a short decode sample; never ignore EOS or invent128 tokens.

One profile ctx2048/prefill64 and pair ctx8192/prefill128 remain distinct complete
configurations. Same messages/seed do not make this a hardware-only speedup.
No SFD/P30/QSA/host/UR profiling is enabled; baseline debug/request lines and
original API token trace remain part of the declared configuration.

COMMAND

Run only after the current exclusive GPU/model actor fully retires. The parent
autoleases; do not wrap without --leased inside another gpu-run lease.

```
PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/run_provisional_speed_screen_v2.py --prepared /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/speed-screen-c140-one-postcache-prepared-v1
PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/run_provisional_speed_screen_v2.py --prepared /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/speed-screen-c140-pair-postcache-prepared-v1 --one-card-receipt /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/c140-v3-one-segmented-postboot-prepared-v3/parent-qualification.json
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s strata/flash-next -p test_provisional_speed_screen_cpu_v2.py
```

RESULT

Source/CPU only at preparation. Client raw SSE line hex/receipt timestamps are
flushed incrementally, including failures. Per-request JSON retains all events,
actual usage, finish, text, reasoning, native timings, first text arrival, text
chunk gaps and wall time to usage. Initial data/role/keepalive is not first text.
One chunk can hold multiple tokens and one token can emit no visible text.
Therefore these are client text-chunk gaps, never true per-token gaps. Completion
tokens/client-wall includes prompt preparation/read/network and is not isolated
decode speed. Client first-text time is not device prefill time or a prefill rate.

Native timings are reported separately without recomputing them from SSE.
Current source40 server.py request_timings3863..3877 maps engine DONE prompt_ms
and decode_ms to API prompt/predicted rates. generate.cpp r0 at9789 starts the
host prompt interval, prompt_ms at10685 ends after refill. d0 at10718 starts the
host decode interval, decode_ms at11620 ends after decode. These are host wall
intervals including engine work and waits, not GPU event timestamps. Keep raw
server-engine.log/DONE and original token trace for later source-bound review.
All timed requests follow the preliminary ordinary screen and are process-warm
diagnostics. First timed request and subsequent requests remain separate: cacheOFF still permits
process-local graph/compiler/expert residency changes. No causal warmup claim.

VERDICT

Report speed-parent.json success only when client and normal original supervisor
termination plus new post-health/journal/full4 gates pass. Results are diagnostic
numbers, with original math/quality failures retained and no shelf promotion.
ABBA here is request order within each launch, not interleaved hardware runs.
First one then pair gives raw numbers; a later independently prepared pair/one
repeat is required for an interleaved configuration comparison. No actual runtime
or performance result is claimed by this source proposal.
