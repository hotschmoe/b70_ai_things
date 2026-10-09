# Bounded CPU RAM discriminator

CONFIG ->64GiB anonymous CPU memory, eight full-byte patterns00/ff/55/aa/01/fe/df/20.
Pinned39992 image, ordinary gcc C11 compilation, no SYCL/GPU libraries in the
probe. Container exposes no devices or model source mount, has80GiB memory
limit; VmSwap remains zero. Source cache pages are checked outside the probe.
Kernel baseline remains7.1.0-070100; no driver/firmware/host settings changed.

COMMAND -> Compile host_ram_pattern_probe_v1.c. Separate16MiB control injects
byte2796 XOR0x20 and must return1 with the correct changed word. Run the actual
64GiB allocation without injection and retain its container inspection/log/state.

RESULT -> Injected control detected. Actual64GiB eight-pattern sweep PASS,
zero mismatches in all eight scans, VmRSS67110696KiB andVmSwap0 throughout.
Owned container exits0 without OOM and is removed; both known source pages match
before/after. Raw evidenceF09/host-ram-pattern-probe-v1/preparation.json, run.json,
negative-control.log andreal64GiB.log. CPU test writes only anonymous memory.

VERDICT -> The cached-source bit change did not reproduce in this bounded CPU
allocation. Physical page coverage/PFNs are unobserved; these are not necessarily
the pages that held the corrupted file cache. Device/driver behavior and the
entire host memory are unqualified. This result cannot assign or clear hardware,
DMA, runtime or storage cause, and cannot turn the failed pair intoPASS.
A fresh all-four buffered model scan follows the memory-pressure exposure before
any new source-upload GPU test. No speed or host-stability claim.

Probe source SHA256 6e4cd981e12ba08909e0a69f5f3a354c325c5b19be70acb744832884531dadfb.
Run receipt SHA256 57647183c5ad023f702e04db7edc3818a6dc2283189ed7e5189a9181a5ba7c3d.

Follow-up: fresh all-four buffered publisher hash PASS after probe removal,
F09/model-identity-after-ram-v1/receipt.json SHA256
76aa2d22bfc0e331fdd66fca68665d71fca6fd87889db44a75857d5027fa7abc.
