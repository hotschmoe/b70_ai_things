# Paired original MTP HC T2 compile

CONFIG -> Original SDK52, original publisher MTP V6 paired MAIN, paired HC T2
candidate V2. Strict default OFF; expert-pair reuse OFF. No ABI layout change.

COMMAND -> Prepare source inputs with the pinned genuine SDK52 receipt. Root
compiles four fresh objects: MAIN, Verifier, HC projection and HC composition.
Copy the original engine/kernel archives, require exactly one old member for each
named removal, delete verify.cpp.o and the two HC members in copies only, and
require zero remaining. Link fresh objects before the archive group. Other engine,
core, prefill, spec and CPU kernels remain original. Exact compile/link flags come
from the genuine production Ninja recipe. Actual object and ELF receipts are
required before a matched root-run OFF/ON model trial.

RESULT -> Source-only modular recipe. Existing leaf operator passed independently;
that does not qualify the paired caller, model outputs, latency or speed.

VERDICT -> Root must check full output equivalence, identity, health, kernel faults,
normal exit/EOF/removal and equal resolved residency before interpreting benefit.
No model payload, compiler or GPU execution is performed by this preparer.
