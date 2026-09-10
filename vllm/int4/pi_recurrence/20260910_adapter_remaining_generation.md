# Remaining generation and cache-reuse failures

CONFIG -> Corrected adapter image09fc6b750415, card1 TP1/MTP3, FP8 fresh
calibration, prefix cache/FULL graphs,100K context/c4. Original concurrent-v1
runs four independent exact arrays cold/reuse with t0/seed42 and cap4096.
This report preserves the original combined gate and evaluates content and
cache reuse independently. No live inputs were changed by the audit.

COMMAND -> Read results and raw SSE, join each stream once, and independently
run the exact-array semantic check without requiring a cache hit. Then the
parent queued the frozen same-session0 payload as two serial requests, using
explicit stable/research model identity checks and separate content/cache gates.

RESULT -> Concurrent content passed14/16; cache gates passed9/16 and original
combined checks passed8/16. Seven of eight reuse requests reported cached0;
that is a cache-reuse test failure, not evidence of corrupted output. The two
semantic failures were session0's thinkingOFF512-element array. Cold stopped
after correct1..433 followed by a comma; the repeated request stopped after
correct1..432 and partial digit4. SSE joined exactly to parsed content, with
normal finish_reason=stop and no bangs. Usage2058/2055 was below cap4096, so
these are not budget-clipped completions. Raw token IDs were unavailable;
finish_reason=stop is not itself direct observation of an EOS token ID.

RESULT -> The identical canonical payload also failed both serial requests:
cold output2055 tokens/total6100/cached0, reuse2058/total6103/cached1600.
Content checks failed2/2 while cache checks passed2/2. Thus concurrency and a
cache miss are not required for the remaining premature-stop behavior. It is
not yet attributed to the model, MTP verification, graph state or a backend
kernel. These totals do not match the earlier4800/4801 bang localization.

RESULT -> Subsequent exact-session0 MTP0 serial control also failed both
requests at2055 output tokens/total6100, with identical partial4of433. Cold
cached0 and reusecached3200; both cache gates passed. MTP is therefore not
necessary for this remaining early-stop phenotype. This finding does not
identify FP8 numerical behavior or intrinsic model quality as its cause.

RESULT -> Cache pressure is substantial despite roughly4K prompt lengths.
The model has48linear and16full layers; actual scale-load receipt includes a
seventeenth full-attention MTP layer. Source grouping gives17layers per pool,
one full group and three Mamba groups with3padding layers, matching the log.
At100K, a request's maximum is63full blocks plus3*(2+3) Mamba blocks =78.
The exact source capacity formula and reported103846-token capacity imply81
shared block IDs,80usable after the null block. A4K request initially needs
3full+3*4Mamba=15 IDs; four requests use60/80=75percent. One extra full block
per request gives64/80=80percent, exactly matching observed usage. Transient
Mamba checkpoint demand can further reduce room for cached, unreferenced
blocks. The source permits eviction of those blocks without request preemption.
No preemption message was found; individual cache-eviction events were not
recorded. The log reports4.25GiB current KV memory;3.99GiB is a suggested
adjusted budget, not the actual allocation. The81-ID figure is source-derived,
not a direct live scheduler dump.

VERDICT -> Preserve both failures separately: confirmed serial premature stops
and unachieved concurrent cache-reuse expectations. Do not call either an
unfixed bang defect without further evidence. The next native5802 comparison
can reuse the exact serial payload and concurrent matrix. The adapter's bounded
composed-copy success does not establish general generation qualification.

Raw evidence under bang_recurrence_testing_20260910:

- adapter-concurrent-independent-audit.json
- adapter-cache-pressure-derivation.json
- tp1-card1-mtp3-conv-adapter1-ctx100k/concurrent-v1/
- tp1-card1-mtp3-conv-adapter1-ctx100k/serial-session0-control/
