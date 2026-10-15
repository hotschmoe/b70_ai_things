# S76: IQ3_S native/S75 cost and parallelism audit (2026-10-15)

CONFIG -> COMMAND

- SOURCE/RETAINED-EVIDENCE ONLY: read actual native/v2 source, S31 CSV and saved
  ISA/result files; calculate logical counts in source-cost-proof.json. No frontend,
  build, binary run, GPU, model payload, model consultation or frozen-stage edit.
- Fixed C6 N2560/M6144 and N6144/M2560. Exact files/hashes in source-pins.json.
  S75 passed strict gates but was about7.7-7.9x slower; that exact mapping is closed.
  No S75 barrier-elision followup is reopened here.

RESULT: retained measured scope, not a causal model

- S31 mapped159 operations over five cycles5..9. Entire primary non-head dense
  pool:468calls/13.8990344ms service per cycle. IQ3_S C6 itself is five rows,
  55calls/1.9335818ms service per cycle. Selected grids1536/640 contribute
  13calls/1.1693698ms. CSV encodes groups, not N or a specific fixture tensor;
  those grids match tested M6144/M2560 via the source row mapping.
- These are instrumented operation service sums, not current uninstrumented wall,
  additive recoverable savings, GPU idle time, critical-path proof or a diagnosis
  of S75. All detailed source byte counts below are LOGICAL, not measured traffic.

RESULT: mapping and useful work

| Shape N/M | Native groups / work-items | S75 groups / work-items | Native K32 steps/lane | S75 serial K32 steps/group |
| --- | --- | --- | --- | --- |
|2560/6144|1536 /196608|384 /6144|2-3|80|
|6144/2560|640 /81920|160 /2560|6|192|

- Native: one SG32/output row, four rows/WG128; p=lane+32*j. It decodes each
  row/p ONCE and reuses that W across six token dots. S75: one SG16/WG covers
  sixteen rows but iterates ALL p sequentially. For sixteen outputs, native uses
  sixteen SG32/512 work-items; S75 uses one SG16/16 work-items. This describes
  source parallel decomposition, not simultaneous resident work or latency cause.
- Both shapes have491520 output-row/K32 segments and the same total original
  ControlSplit loads. Per sixteen outputs/K32, native has768 logical DP4A calls
  (six tokens x sixteen outputs x eight packed dots). S75 has one logical matrix
  collective computing4096 integer MACs,3072 useful: two padded token rows add
  25percent of tile slots. Static S75 ISA has one DPAS site, not a dynamic counter.
- Native keeps six FP32 partials/thread and performs the original XOR reduction;
  no explicit shared staging/WG barrier in the selected STAGE_GRID=false path.
  Retained S68 original ASM has zero SLM sends/signal barriers. Its unused source
  uint32[4] grid declaration is not evidence of runtime SLM use.
- S75 has the same active FP partial-update arithmetic, but moves the32 partials
  per output/token through shared memory and a target-lane final tree. Its source
  has1+3*P WG barriers/group,13708B shared payload before alignment, and16-float
  private tree. Actual ISA has four static barrier/fence sites and130 static SLM
  sends; none of those static counts is a measured dynamic invocation/byte total.

RESULT: per sixteen outputs/K32 logical data references

| Quantity | Native | S75 |
| --- | ---: | ---: |
|Compressed weight field reads|256B|256B|
|IQ grid value reads|512B|512B|
|Q8 qs+ds[0] field reads|3264B|204B|
|Extra shared element references|none in selected source path|4428B|

- Weight fields per row/p are qs8+qh1+signs4+scalebyte1+half-d2=16B; distinct
  packed storage is110B/256 weights, so repeated d/scale reads are included.
  Eight grid uint32 values add32B/row/p from a512-entry/2048B codebook.
  Q8 needs32 code bytes+2B scale per token; ds[1] is unused by IQ3_S.
- S75 already amortizes logical Q8 reads across16 outputs, yet its weight decode
  count is unchanged. It writes padded A/B, scales and INT32 tile, reads that tile
  and scales, and reads/writes the ledger.4428B includes ideal logical matrix
  element accesses; compiler broadcasts/hoists and layout may change actual refs.
  Ledger init/final read adds24576B/group once. Caches/coalescing/transactions are
  unmeasured; no claim that these bytes caused the observed slowdown.
- Native lanes visit neighboring K32 segments of ONE row; S75 lanes decode
  sixteen different weight rows at one p (raw row strides1100/2640B). Equal logical
  bytes do not imply equal memory access/coalescing behavior or measured bandwidth.

RESULT: ONE source-eligible lead, temporary signed-code plane/original native map

- Retain original SG32 p-lane assignment, DP4A apply, FP statements/partial order
  and stored-lane XOR tree. Replace ONLY W loading with predecoded signed grid
  codes, oddls bytes and original half-d bits. A one-time temporary device pack
  would use the exact original ControlSplit decoder; native apply is unchanged.
  Code int32 words recreate identical g[8], ls widens exactly, and dw uses the same
  half->float conversion. No new reduction, DPAS mapping or activation substitution.
- One fixed layout: signed codes N*M bytes; oddls N*M/32 bytes; rawhalf-d
 2*N*M/256 bytes. PER SHAPE:15728640+491520+122880=16343040B (15.5859375MiB).
  Both shapes require32686080B temporary EXTRA allocation while original weights
  remain for public anchor. Expanded storage is2.41818x raw6758400B, not merely
  the9.140625MiB difference; original retention matters. No full-model commitment.
- Logical W-load fields become32code+1ls+2d=35B/row/p, replacing compressed16B
  plus32B codebook values. The codebook is only2KiB and cache reuse is unmeasured;
  moving codes into a15MiB plane can increase physical memory pressure substantially.
  Packing cost must be separately reported, not hidden in a projection-only timing.
- Distinct from S65 SWAR: eliminates complete grid lookup/sign decode each call,
  rather than rewriting a few byte-sign expressions. Same broad predecode/expanded
  weight family as rejected Q6 packing, so it inherits that risk; IQ3_S codebook
  work and preserved ORIGINAL mapping make it a distinct source contract, not proof
  of a performance win or reason to erase Q6's rejection.
- Eligibility/blockers: private two-slice scope only; no existing production IQ3_S
  pack API. New temporary owner/loader would need exact code/scale preservation,
  aligned words, guards, public/original strict initial+changed bytes and drained
  graph lifetime before free. Compiler behavior still gates compiled bit identity.

VERDICT

- One source-eligible mechanism identified; HOLD implementation/build/GPU pending
  root selection. No candidate code, packing run, format sweep, benchmark framework
  or future GPU authorization prepared. Source counts cannot establish whether
  removing decode repays expanded-weight traffic or packing; no speed forecast.
