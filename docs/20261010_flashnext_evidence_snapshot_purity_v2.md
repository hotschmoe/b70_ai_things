# Evidence snapshot V2 supported purity contract

CONFIG

NEW unintegrated successor to V1 evidence DAG prototype. Root's independent
same-code make(1)/make(2) closure control demonstrated stale V1 semantic reuse:
the key bound code/source/arguments but not captured cells. No actual runtime
integration or successful model/source qualification used that prototype.

COMMAND

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_operation_evidence_snapshot_cpu_v2

RESULT

Ten tiny CPU controls pass. V2 rejects closures, positional/keyword defaults,
bound or non-Python callbacks, imports/global/deref mutation instructions,
private/reflection attributes, and unknown/impure/shadowed builtins. Nested
code objects are inspected too, preventing a comprehension from hiding a
LOAD_GLOBAL dependency. All referenced nonbuiltin globals need explicit
complete typed bindings and must be immutable primitive/tuple values; mutable
containers, module objects and function/object globals are refused. Changed
immutable global binding cannot borrow previous success. Boolean/numeric and
signed-zero distinctions remain exact.

The validator receives a restricted immutable byte-input view plus a deep
copy of the complete keyed parameters. It cannot receive live epoch counters,
phase, mutable memo or unrelated inputs. Returned results are copied and
preserve original data types. Source SHA/code identity/complete roster,
entry-predevice-post bytes, ownership/lifetime and declared unobserved intervals
remain mandatory. No saved proof/result can be imported as current admission.

This is a conservative supported subset, not a claim that arbitrary Python
functions are pure. The original logical-free adapter imports parser/json
modules and therefore is explicitly NOT admitted as a cached V2 validator.
A future actual parser port needs a reviewed complete dependency contract or
source-indexed refactoring into this supported interface, plus unchanged
positive/negative/destructor/owner predicates. Current live source/model/page/
health/time/lease/container guards remain outside all result reuse.

VERDICT

Frozen SOURCE/CPU10 only; no runtime integration, predicate omission or speed
claim. Preserve the V1 counterexample and current H49 evidence. Broader parser
reuse requires another deliberate reviewed source generation, not accepting
mutable globals under a vague same-file code hash.
