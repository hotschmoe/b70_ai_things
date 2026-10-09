# Independent original-GGUF mathematical reference route

CONFIG -> CPU-only source audit of frozen engine0018 at
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp.
Engine source fb58e0 plus0001..0018; embedded llama/ggml source3cf03257.
Selected original four-shard UD-Q4_K_XL, unchanged tokenizer/template, MTP off.
No GPU execution, source mutation, new model artifact or weight conversion.

COMMAND -> Read consumed SYCL verifier/prefill, embedded qwen4exp graph,
dequantizers, existing scalar parity references and local qualification scope.
Record exact source hashes in independent-math-reference-source-audit-v1.json.

RESULT -> The strongest practical independent route is a bounded lazy CPU
forward from ORIGINAL GGUF bytes, with separate mathematical and explicitly
rounded implementation-contract references. It is preparation work, not an
existing qualified full-forward executable. The correct embedded architecture
is qwen4exp, not the similarly named qwen3next graph.

## Existing evidence and gaps

- Full390 covers387 HC and3 PLE original images. Its plan explicitly excludes
  complete ordinary dense/head/embedding/expert arithmetic and persistent state.
  Correct ownership, source readback and frees do not qualify those operations.
- HC composition/write and PLE fixtures have independent FP64 references but
  bounded synthetic inputs. They do not prove every invocation receives the
  right tensor, activation or history in a real forward.
- Mirror v4 qualifies ten selected original experts and table/lifetime paths,
  independently separating Q8_1 packet math from original-F32 math. It does not
  cover every layer/type/shape/real routed expert. Derived SiLU reconstruction is
  not a raw fused-hidden observation.
- Observers capture48 residuals and first raw logits, suitable for localizing
  divergence. Same-engine fresh/repeated/off/on equality shares model equations
  and cannot detect a common arithmetic error. Current all-state snapshots
  likewise require independently computed expectations.
- Frozen ref contains only load.py. model.py/gdn.py/qsa.py named in parity
  comments are absent. load.py has neither Q5_1 geometry nor decoder dispatch,
  although actual selected expert down tensors include Q5_1. Do not run it as a
  complete exact-artifact loader without a separate corrected generation.
- Existing GDN/QSA scalar parity references are useful independent equations
  and rival-reading controls, but source existence/compilation is not actual
  model-path numerical qualification. QSA parity also shares f16 conversion
  helpers with the implementation in some comparisons; isolate that dependency.

## Proposed bounded progression

1. Build a separate CPU reference loader using the independent frozen inventory
   and exact source offsets. Support every ACTUAL intake type, including Q5_1,
   with independent NumPy/gguf-py dequantizers cross-checked against pinned ggml
   scalar decoders on known packets. Do not decode compatibility copies. Enforce
   exact tensor names/shapes/bytes/SHA and fail closed on unknown types. Gather
   only requested IQ4_NL PLE rows and routed experts; never materialize the full
   expert bank or PLE table. Bound decoded cache to one layer and selected experts.
2. Implement explicit scalar/NumPy equations from the pinned qwen4exp graph,
   reviewed against Strata's independent parity equations. Start with actual
   embedding plus first GDN layer, then layer1 PLE and the first QSA layer. Use
   asymmetric labelled states to reject wrong head maps, strides, time order,
   pooled-cell rotation, norm placement and expert indexing. Model metadata,
   not stale reference comments about two shards, determines geometry.
3. Conditional layer replay consumes a recorded input residual AND recorded
   incoming persistent state, computes every projection/router/mixer/expert/
   combine independently and checks outgoing state plus residual. This is a
   local operator check only: injected GPU inputs cannot establish upstream or
   end-to-end correctness. Missing incoming state means recurrence replay is
   unavailable, not zero-initialized by assumption.
4. The decisive end-to-end lane consumes exact GEN token IDs from empty state,
   computes its OWN routing and persistent state through all48 layers, and emits
   full248320 raw logits before sampling. Begin with1/2/4/8 tokens, then16 tokens
   to cross multiple QSA pooling cells and PLE history updates. Compare every
   residual/state and vocabulary row, not generated prose alone. Choose bounded
   histories that include EOS reset and later extend beyond any pooling/top-k
   thresholds before claiming general QSA fidelity.
5. Maintain two expectation lanes: original-dequantized weights with FP64 sums,
   and independently encoded Q8_1 / exact FP16/BF16 storage at each ACTUAL operator
   boundary. Record original-math deviation separately from numerical error
   against that rounded contract. Packet identities, routing IDs and layout/
   position/hash/history metadata need exact checks. No global3e-2 gate may hide
   a wrong scale, decoder, routing choice or accumulated state error. Existing
   HC/PLE NMSE1e-6/max-normalized1e-4 gates remain unchanged. Preregister new
   operator-specific gates and cancellation-aware denominators before GPU use;
   full-forward accumulated-error limits are not established by this audit.

Fresh-process llama one-prompt controls can provide additional implementation
triangulation while its repeated-history bug is unresolved. They must retain
the same original GGUF, token IDs, state initialization and declared arithmetic
precision. A fresh run is not automatically trustworthy; require per-layer
agreement with the independent equations. Do not replace the oracle with an
unverified fresh llama output or a different BF16 checkpoint.

## Highest-priority unverified consumed arithmetic

Paths below are relative to the frozen source directory; embedded qwen4exp and
quants paths are relative to the engine's ggml-source directory.

| Path | Exact source locations | Independent gate needed |
| --- | --- | --- |
| Native dense projections and head | sycl/src/core/verify.cpp:1011-1052,1089-1090,1164,1265,1623 | Every selected type/production shape; actual Q8_1 packets; original weights; full head row |
| GDN preprocess/recurrent/output | sycl/src/core/verify.cpp:1026-1052; sycl/src/kernels/gdn_parity.cpp:47,72 | Actual projection inputs, decay-before-update, modulo pairing, labelled conv direction, full outgoing state |
| QSA indexer/selection/cache/attention | sycl/src/kernels/qsa_parity.cpp:154; src/models/qwen4exp.cpp:542,695,775 | Exact selected IDs and cache addresses; separately quantify FP16 cache cost; pool-tail and causal positions |
| Router/shared/expert combine | sycl/src/core/verify.cpp:1305,1320,1362-1400; src/models/qwen4exp.cpp:988 | Full router logits, exact top10/weights, each chosen expert, shared gate, ordered weighted sum |
| Prefill/decode precision seam | sycl/src/prefill/prefill.cpp:2806-2827 | Actual post-HC F16/BF16 boundary rounding vs verifier Q8_1, chunked state recurrence vs scalar tokens |
| PLE gather/history/complete integration | sycl/src/core/verify.cpp:872-935; src/models/qwen4exp.cpp:1185,1206 | Original IQ4_NL row hashes, EOS n-gram reset,57-row incoming/outgoing history, residual injection |

Embedded semantic anchors: qwen4exp.cpp:267 HC mix,323 combine,353 full graph,
861 GDN,988 MoE/shared,1206 PLE. Decoder anchors: ggml/src/ggml-quants.c:302
Q8_1 reference encoder and526 Q5_1 decoder. Use these as equation sources and
independent packet controls; do not invoke production fused kernels as oracle.

VERDICT -> Actionable next source task is the lazy original-GGUF loader plus
independent first-GDN/layer1-PLE/first-QSA replay, then empty-state48-layer CPU
forward. Existing C1 and self-consistency evidence do not yet establish complete
original-GGUF mathematical fidelity. Full concurrent/history/prefix lifecycle
and speed qualification remain separate. No new numerical result is claimed.
