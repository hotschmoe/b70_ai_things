# S74 recorded INT8 matrix compile/static-ISA prerequisite

One signed INT8 8x32 times32x16 matrix expression, INT32 output, SG16.
Actual CPU build accepted the BMG-G31 target and offline disassembly contains
one signed-byte DPAS. No binary execution/GPU/numerical/performance result.
The optional --run path in the source was NOT executed in S74.

Exact source/build recipe copies preserve original historical paths/hashes.
Original preparation/build:
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/int8-xmx-capability-s74-prep/
Root build logs: sibling int8-xmx-capability-s74-root/.
Offline decoder/extractor: sibling int8-xmx-capability-s74-isa/.

Do not execute these copies in place or overwrite frozen outputs. build.sh
rejects an existing output directory. isa-extract.py and isa-decode.sh are
record copies named extract.py/decode.sh in the original ISA directory;
the decoder expects that recorded layout. A future replay needs a new directory
and explicit path bindings. No generic extraction framework is installed.

The build uses cpu-run, the pinned offline image/no DRI/network, caller UID,
disk-backed temporary files and four AOT link jobs. It never runs the binary.
Source and six installed matrix header hashes are checked before and after.

Static resources/DPAS are not runtime capability, correctness, occupancy or
speed. No IQ decode, odd scales or FP32 reduction is exercised by this source.
See docs/20261015_strata_int8_xmx_capability.md from the repository root.
