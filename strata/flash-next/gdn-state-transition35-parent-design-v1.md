# Owned GDN35 component runtime parent

CONFIG -> exact leaf plan d50affd3050aa64db17485357dc5db742e0e0ed6fa7849fe886cd6976cb37c4f,
source35 SDK and freshly compiled leaf ELF. No source/backend/model arithmetic
changes. Static archives are currently pinned, not retrospectively certified
by the original eight-ELF SDK receipt.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s
strata/flash-next -p 'test_qualify_gdn_state_transition35_cpu_v1.py'.

RESULT -> 17 CPU tests PASS. Compile/health/Docker lifecycle metadata is mocked;
actual tiny /tmp raw files exercise the frozen27-pair collector, modified-state
negative and chronological UR allocation/free parser plus its negative controls.
Dynamic failed-child tests retain post-health, removal and newfull4 chronology;
failed compile admission performs no launch or model scan. No actual leaf/GPU/
Docker/model payload/runtime execution by this agent.

VERDICT -> frozen CPU parent source READY only. Root owns actual execution:

```sh
python3 strata/flash-next/qualify_gdn_state_transition35_v1.py --compile-receipt <fresh-successful-compile/receipt.json> --output <NEW-owned-component-parent>
```

The parent acquires both cards via bin/gpu-run for strict percard+compiledP2P0
pre/post health. The leaf validates inherited card0 FD8 and pins physicalcard0;
FD8/9 remain inherited/open until named container exit/removal and post-health.
Current bin/gpu-run cannot nest safely inside a held pair lease: it reopens FD8
and reacquires the parent's lock. No bin change or nested reacquisition occurs.

Fresh compile admission checks exact controller/plan snapshots, canonical
compiler command and explicit image ENTRYPOINT /bin/bash, clean compile exit,
removed container, ELF/binary/source/archive/all8 SDK targets and leaf hashes.
Runtime explicitly uses /bin/bash -c and mounts binarydirRO/output only; it does
not mount model weights or SDK. EAGER presence is refused, UR2 remains enabled.
Only exact named/image/labeled ownership is removed. Unexpected live ownership
retains the lease; absent-before-inspection cannot fabricate terminal evidence.

After leaf terminal/removal, retain post-health/journal and NEW complete four
publisher hashes, with both known shard3 pages before/after and failure copies.
Recheck all code/compile/source/archive/ELF/leaf/binary bindings after runtime.
Collector bitwise27 and modified-state negative plus chronological UR logical
allocation/free must all pass. Any synthetic numerical failure remains a failed
diagnostic even when health/teardown succeeds. Physical backing reclamation,
model allocation ownership, real row3 inputs, original math/model quality,
concurrency and speed remain unqualified. Parent PASS means this synthetic
component/lifecycle control only, never full model fidelity.
