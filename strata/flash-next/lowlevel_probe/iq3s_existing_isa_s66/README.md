# S66 recorded offline ISA audit

Exact experiment sources for the
[dated findings](../../../../docs/20261015_strata_iq3s_existing_isa.md).
They retain their actual frozen-input/raw-output paths. Prepare a new external
output directory for another use; do not overwrite the recorded evidence.

The extractor reads one existing device image and metadata, not model payload.
The alias script changes only one section name in a COPY to work around the
ocloc output filename limit. That copy is for offline disassembly only and
must never become a link/runtime input. The original image and instruction
bytes remain the source of evidence.

Use the pinned compiler image without DRI/network and bin/cpu-run for offline
tool work. Verify expected output files as well as tool exit status. No GPU,
new compilation or candidate speed conclusion belongs to this audit.
