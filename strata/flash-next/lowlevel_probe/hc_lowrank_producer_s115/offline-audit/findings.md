# Selected original HC UP/reduce offline audit (2026-10-16)

CONFIG -> Root and actual Opus/Fable authorized ONLY offline CPU extraction and
decode after reading the HC producer candidate. Authority: parent S113 post-HC
root-post-decision.json, actual models claude-opus-5-5 (55.236s) and
claude-fable-5-1 (64.737s). No private oracle/source edit/build/GPU/model CLI.

Use frozen S107 main 064b63ce75d7568db541af5cf24eee07f41d0e54ea49813af9958a816d75d5ca
and archive cf710683aec404599d1cc6173bf8364e6d30731bda82b315df286bf58aed9b38.
Full identities are reused from prior root closure/S107/S113; this audit checks
current pre/post stat continuity, not a fresh full main/archive hash. The exact
607880-byte fused_gr member was checked against its current object. The first
attempt stopped on an overly broad public-entry symbol predicate; root permitted
the specific parser fix. Retained member bytes and HOLD are under attempt1-*;
continuation rechecked the copy/object, without rereading the entire archive.

COMMAND -> B70_AGENT=HC-S115-selected-old-offline-corrected bin/cpu-run bash run.sh
One CPU lease, serialized after S114 root CPU work. Two selected-image disasm
calls in pinned e41, ocloc7a418c1e7dfc62b6fb3895bea111647fa3e1ff600bb69f6be2a3114b90c41eac,
-device bmg-g31; UID1000/networknone/cpus4/noDRI/diskTMP. CPU wrapper0 in12s;
both decoders0, each input hash unchanged. This is not a GPU experiment/retry.
Original full images and section-name-only aliases, complete metadata/options,
section inventories, decoded raw relocations and full ASM remain external.

RESULT -> Two static bindings PASS. Object host relocations resolve exact Info,
cache and guard, linked main Info/name pointer, then unique matching .ze_info
and selected executable text. UP host is launch_gr_up_multi, with its C16
normal branch; reduce host is public fused_gr_read_multi. This establishes
actual compiled host/image identity, not runtime dispatch or numerical proof.

The actual public host calls launch_gr_up_multi at0x2b2c. Normal reduce submits
at0x17af with its own Info/cache/guard. UP C16 default path tests grf=false
(0x3c60 ->0x3e41), ILP=false (0x3e57 ->0x4041), sets global0xa000/local0x100
(160 WGs/256 threads), and uses C16 normal Info at0x594a (.rodata+0x2240),
cache/guard0x5e0/0x5e8. C16 target name is gr_up_multi_cols/scalar16, not
MORE, ILP or grf256. Exact metadata and relocation facts are in selected/facts.

S107 and S113 configs and complete container Config.Env contain no STRATA_GR_*
overrides. Retained native env is only a configured-key projection, so its
absence alone does not prove full native absence. Source defaults on B70:
V3=false, down-sliced=true, MORE=false, large-GRF=false, UP cols16 and ILP=false;
server child_env inherits container env and overlays config. Source sets
nsplit=1 in normal sliced path. Existing S95 selected normal-name/grid rows
support this family lineage; no S113 device trace/dispatch has been invented.
This source-conditioned normal-default binding remains distinct from a new
runtime dispatch observation.

Selected resources (literal fields): UP GRF128/SIMD32/SLM12288/barrier_count1,
EU-thread-count8. Reduce GRF128/SIMD32/EU-thread-count8; SLM/barrier fields absent.
Spill/scratch fields absent BOTH, hence UNKNOWN, not zero/no-spill proof.
UP ISA has TWO sync.bar sites despite metadata barrier_count1; do not conflate
static barrier instructions with the metadata allocation field. Both options
are exactly -cl-fp32-correctly-rounded-divide-sqrt. Neither image has a named
.data/.bss/.rodata section. Each has two type2 REL entries at76/156 targeting
__INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; raw records retained without an
unverified relocation-type interpretation or claim of all implicit memory.

Actual UP nonlinear prologue and readiness:
- Work loop starts752; split0 load1056, retained add loaded_part +0.0 at1072.
  Later split additions remain conditional on nsplit>1/2/3; normal path uses1.
- At1720 source x/=float(HC) is ACTUALLY lowered to mul0.25. At1728 the exp
  argument uses mul -1.442695 (raw immediate0xbfb8aa3b), math.exp at1744, add1
  at1760, math.invm half-SIMD sites1792/1968 and14 static madm refinement sites.
  This is the observed compiled sequence, not permission to rewrite source
  division into multiplication or universal exp/rounding equivalence.
- Result goes to local lo for all active rows/groups at2176. Group==0 was
  compared at560 and saved, then reloaded only AFTER nonlinear evaluation;
  nonzero groups jump over global lo publication at2192 ->3008. Global store
  is2992. Thus LO_ONCE suppresses global stores, not the nonlinear prologue.
- Backedge to752 advances256 across T*320, independent of UP group except that
  store predicate. Source launch has160 WGs: repeated per-WG activation is
  present in this selected image;160 is geometry, NOT a measured speed factor.
- First barrier sequence3088..3200 includes SLM+UGM group fences, signal and
  sync.bar at3200. First w_up vector load is6448, after pointer selection.
  Its address uses source w_up base (payload source_offset56 ->arg offset80,
  r3.6:q), row stride640 and lane stride16. Earlier UGM loads are cross-thread
  payload and partial sums. Weight-data loading was not hoisted before this
  first barrier in this image. Static order proves no elapsed wait/cycle cost.

Normal reduce has80 static a64 partial-load sites and80 FP32 add sites in a
single accumulator chain (first loaded slice +0; final value to r108), retaining
its existing raw-part publication. There are no math.exp/invm or sync.bar sites.
It currently lacks the extra post-reduce +0/nonlinear chain. Adding that chain
could make its small grid slower; UP-only gains cannot establish net gain.

Both kernels literally OR CR0 with0x400004C0 (UP328, reduce280). This observation
DOES NOT decode denormal, rounding, exception, inherited/reset mode semantics.
No universal claim that reduced x cannot be -0 is accepted. Candidate must keep
extra +0, source division, nonfinite and signedzero/subnormal obligations.

Static counts through first EOT: UP6417 instructions/end82848 of83072 text bytes;
reduce1165/end16272 of16512. Labels were validated using8-byte Compacted/16-byte
other instructions. Prologue loop752..3008 contains1 math.exp,2 math.invm,
14 madm sites; later UP sigmoid/other nonlinear sites are OUTSIDE that candidate
seam and must remain. Counts include conditional/refinement sites; they are
not dynamic executed work, cycles, SFU traffic, bandwidth or removable budget.

VERDICT -> The specific mechanism survives selected original ISA inspection.
ONE narrow private oracle is worth preparing AFTER root reads this packet;
it is not implemented/authorized here. Keep norm/down/reduce/UP boundaries,
raw part2/part, injection, disjoint owner lo and normal C16/nsplit1 scope. Move
only the completed nonlinear lo publication into reduce; UP loads it into SLM.
No generic public-alias guarantee or other HC path follows.

Before GPU, bind old exact images and new unique private pair, then compare
ordered FP dataflow/CR0/options and unchanged UP body, with scheduling/register
changes explicit. Raw image/text byte equality is a SEPARATE stronger claim:
short aliases preserve selected text byte-for-byte; future changed kernels
will not inherit raw identity merely because their arithmetic ledger matches.
Preserve store/load boundary rounding or prove it with the strict oracle.
Use retained bounded T1/2/6 x apply0/1 x inject0/1 snapshots/guards plus small
+0/-0/subnormal/nonfinite epilogue fixtures and changing A-B-A eager/captured
inputs. Candidate lo-before-UP compares to baseline lo-after-UP; raw parts and
all consumed finals/tails/read-only buffers stay exact. Mismatch means HOLD,
no tolerance/zero/retry. Only a later balanced whole-read measurement, if
separately authorized and strict checks pass, decides usefulness. No forecast
from S95 family times, source160-fold counts or this offline static audit.
