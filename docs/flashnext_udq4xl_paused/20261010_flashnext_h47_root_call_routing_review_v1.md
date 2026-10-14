# H47 draft root call-routing review

CONFIG -> Draft H47 operation-local pack witness; old H46 preparation remains
live and unmodified. Read-only root review; no model/GPU work.
COMMAND -> Inspect operation_pack_hash_witness_v2.py and explicit pack_epoch
call sites in H47 controller, parent, native/serial runners and gate ports.
RESULT -> Found unsupported pack_epoch keyword on subprocess.run docker rm
in both native and serial runners; found old serial37_selection_v3 dynamically
imported with new unsupported pack_epoch argument in serial_reader_pack47_v1.
Reported to author before freeze. Requested explicit call-routing regression,
serial predevice seal and full parent/child lifetime ordering review.
At observation PID956195 was live at23:28 and rchar1076157493484 bytes,
physical read_bytes345731072. Cached logical read amplification continues;
no matched optimized preparation speed measurement exists.
VERDICT -> Draft held for corrections. Preserve full current-byte boundaries
and original provenance; no runtime qualification from AST/import success.

CONFIG -> Original operation-local witness V1, tiny temporary CPU fixtures.
COMMAND -> PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_operation_pack_hash_witness_cpu_v1
RESULT -> 10 controls PASS in 0.007s. These exercise three complete byte-read
boundaries, mutation/roster/receipt/ownership/expiry rejection and closure.
V2 changes sealed-phase semantic consumption and requires separate controls;
V1 tests do not qualify the H47 integration. Serial runner signature and
predevice seal were inspected and are present in the draft.
VERDICT -> Source review progress only. Actual optimized preparation/runtime,
matched dev-loop speed and the full campaign goal remain unqualified.
