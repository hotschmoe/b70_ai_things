# S75v2 offline compiled-image audit (2026-10-15)

CONFIG -> COMMAND

- Existing actual build only, binary18045cced6ce5539851a8e9db1e52a40bff6660a5e3aa9af4f0d019dcdae032f.
  Header8a490cc2.../leafe2605695... match actual build records and current v2 bytes.
- cpu-run bash iq3s-int8-dpas-s75-isa/decode.sh: reused bounded S68 ELF seeks,
  selected ONE DpasC6 plus public C6/control identities and metadata; decoded
  candidate only using pinned e41d59c0 image/ocloc bmg-g31, no network/no DRI.
- CPU lease9s/exit0, decoder0/empty stderr, lease released. No binary execution,
  compile, common AOT/archive rehash, GPU, payload or model consultation.

RESULT

-1088 embedded images. Candidate _ZTSN3s756DpasC6E, private-original and exact
  public IQ3_S/C6 full IDs each occur once. Public zebin is unchanged from S65:
  bad2ccded9f4136e6809825844815652eba954f059b807774c45d7916d92beb4.
- Candidate main text17664B, original16128B. Exact image/text SHA and complete
  metadata are retained in selection.json/report.json; full candidate ASM retained.
- Candidate ze_info: requested SG16/SIMD16, GRF128, has_dpas=true, barrier_count1,
  eu_thread_count8. Barrier_count is metadata resource count, NOT invocation count.
  Seven dynamic local/SLM pointer args align4/4/4/1/1/2/4 and three global args.
  No total dynamic SLM/scratch/private allocation field is emitted.13708B is
  source payload before runtime alignment, not a measured allocation or occupancy.
- Exactly ONE static signed INT8-to-INT32 instruction in the K32 loop:
  dpas.8x8 (16|M0) r13:d null:d r5:b r105.0:b
  Integer scale multiply follows, then local tile store and publish barrier.
- Four static signal-barrier, four SLM-fence and four sync.bar sites match the
  initialization plus three source loop boundaries. These are static inventory,
  not measured/dynamic barrier counts. No performance or correctness inference.
- At candidate ASM889-891, first token's ledger path has FP32 mul(dw,dx), then
  separate mul(...,sumi), then add(partial,...). This observes v2 emitted code;
  it does not prove v1 differed, establish all FP dataflow, or replace strict gates.
- Actual duplicates comprise eight DPCT memcpy/SYCL fill helper families (each2)
  plus Intel_Symbol_Table_Void_Program metadata821. These are shared helpers,
  not duplicated target math routes. Their bodies were NOT exhaustively compared.
  This is the bounded known S65/shared-helper registration risk; parent separately
  checked ordinary two wrapper constructor/unregister pairs and host link map.
  No new target-ID or public-image blocker was found; runtime gates still matter.

VERDICT

- PASS bounded static prerequisite. No blocker found for root to decide whether
  to run the ONE already reviewed strict primitive window. This does not run or
  independently authorize it. Actual local matrix behavior, outputs, full guards,
  changed graphs, DOT resources, health and confirmed removal remain pending.
- Static ISA is not proof of runtime image selection, numerical identity, speed,
  physical bandwidth, occupancy, latency cause or serving benefit. No new probe,
  compiler policy, numerical tolerance or followup implementation was added.
