# S122 actual host review (2026-10-16)

CONFIG -> Frozen S107 main 064b63ce, object e4dafa84 and native source ae47a632;
full identities in facts.json/pinned prep. S107 mode0 config SHA5883bf0b sets
STRATA_HEAD_Q5K_RPW2=0, STRATA_MMVQ_WIDE_K=1,
STRATA_MMVQ_WIDE_K_RPW=1 and STRATA_BATCH_MTP_PAIR_HEAD=1.
Shape is source-conditioned N=2560, M=248320, C=2.

COMMAND -> Read all four saved actual function bodies (including normal,
initialization, cleanup and exception tails), cross-check pinned source
2077-2140,3290-3303,3940-3978 and verify all four saved ASM hashes against
facts.json. No new ELF scan/disassembly, decoder, build or GPU invocation.

RESULT -> Static forwarding and selected launch agree with the source route:

- head symbol _ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv
  starts 0x67f90. host-2 lines15-24/104-116 show static head flag loading
  and exact getenv value "1" test (0x680f9-0x68118); "0" takes fallback
  0x68072. Lines62-70 restore M to r9d and tail-jump native_mmvq via
  relocation 0x68084. Original type, weights/input/output, N, C and stream
  forwarding is preserved; stack C/stream slots survive frame removal.
- generic symbol _ZN6strata7kernels11native_mmvqEiPKvS2_PfiiiPv
  starts 0x69260. host-0 lines11-16 show type-minus2 jump-table dispatch.
  Pinned source3972 maps case13 to Q5; lines137-148 show its call site
  0x693fd, relocation 0x693fe, moving weights/input/output/N/M/C to
  rdi/rsi/rdx/ecx/r8d/r9d and stream to stack.
- Q5 symbol _ZN6strata7kernels16native_q5_k_mmvqEPKvS2_PfiiiPv
  starts 0x6960. host-3 lines15-77 show saved inputs, validation, full
  argument forwarding at 0x69ee-0x6a00 and call 0x6a03 to try_wide at 0x70f0.
  True return exits; false proceeds to generic multi-column fallback.
  Source2130-2134 admits wideK=1, C2, N divisible256 and calls WideQ5K/C2
  with default wide_k_rpw(), source2077-2079 giving configured RPW1.
- selected symbol
  _ZN6strata7kernels12_GLOBAL__N_111launch_wideINS1_7WideQ5KELi2EEEvPKvS5_PfiiPN4sycl3_V15queueEi
  starts 0x68280. host-1 lines12-18 test RPW2/4; RPW1 reaches0x686e9.
  Lines191-227 store three pointers plus N/M into capture offsets
  0x38/0x40/0x48/0x50/0x54, construct global(1,4,32*ceil(M/4)),
  local(1,4,32). This is 62080 groups for retained M. Lines231-234 read
  selected guard/cache;326-335 initialize exact Info (.rodata+0x78048)
  and cache (.bss+0x36f0). PC32 cache addend is0x36ec, normalized+4.
  Lines248-268 pass that Info, capture descriptor and geometry to
  submit_kernel_direct_with_event_impl at 0x688d9. facts.json binds this
  Info to its exact registered name and selected 30168B image aa4d0202.

VERDICT -> PASS_STATIC_HOST_SOURCE_CONDITIONED_BINDING. Suitable target for
ONE selected-image offline ISA inspection. The try_wide body0x70f0 and
jump-table entries were not separately decoded among these four functions;
those edges use pinned source plus actual caller/callee symbols, not claimed
compiled-body coverage. No observed runtime dispatch/scalar payload, current
serving execution, numeric equivalence or speed conclusion. ISA_UNKNOWN here.
