# Serial V3: strict canonical JSON recovery interface

CONFIG

Root's actual V2 preparation failed before GPU launch because the first49
adjudicator's in-memory metadata contains tuples and integer object keys, while
its saved JSON necessarily contains lists and string keys. Root confirmed exact
full-field equality after JSON roundtrip, with no artifact or numerical change.
All executed V1/95, V2/103, first49 receipts and failed partial preparation remain
immutable. V3 uses NEW consumer/selector/controller/parent43/reader names.

COMMAND

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s strata/flash-next -p test_serial37_canonical_json_cpu_v3.py -q

Root-only preparation: batch_serial_source37_v3.py prepare with the prior explicit
arguments and --origin-plan PRESERVED_GENUINE_V40_PLAN --first-job-adjudication ACTUAL_FIRST49_RECEIPT.
Root-only parent: qualify_batch_serial_source37_v3.py --plan NEW_PLAN --output NEW_RUN.
Readonly API: validate_batch_serial_source37_v3.finalized_binding(NEW_RUN).

RESULT

26 CPU controls pass, including exact tuple/integer-key JSON representation,
boolean-versus-integer and integer-versus-float distinction, missing/changed
fields, duplicate serialized keys, nested duplicate saved keys, unsupported key
types, nonfinite values and the actual origin-reuse branch. No model/GPU execution.

The codec encodes with allow_nan=false, rejects duplicate serialized object keys
before normalizing, and compares exact sorted encoded JSON values. It allows only
JSON's tuple/list and integer-key representation boundary; no field is discarded,
boolean/integer equality is never Python True==1, and int1 plus string1 key
collisions reject. Saved receipts are decoded with duplicate-key rejection.

--origin-plan reruns genuine current V40 manifest admission and records the exact
preserved origin path/SHA plus complete unchanged nested preparation, avoiding a
repeat origin.prepare. Actual first49 core admission still reruns. V3 retains V2
prospective EOF/log/error/retirement, raw-file equality and pre/post source joins.

VERDICT

SOURCE/CPU only. First49 remains historical PIN0/original failed parent and has
no observed old supervisor EOF producer stamp. Remaining3 must be actual fresh
absent-PIN under new parent43. No full-group absent-PIN, fullmodel/math, quality,
cache/concurrency or latency qualification follows from source preparation.
