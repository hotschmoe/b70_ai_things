# Bounded readonly recurrence observation

CONFIG -> Root supplies one aligned 4096-byte page, its expected direct hash,
and an exclusive evidence output. Initial buffered and direct bytes are saved
before the optional --single-page-dontneed-after-preservation action. This
action uses an O_RDONLY descriptor and exactly one POSIX_FADV_DONTNEED range;
no global flush, model write, GPU action or full model identity claim occurs.
Anonymous aligned memory is used only for the direct read buffer.

COMMAND -> Run probe_readonly_page_recurrence_v1.py --path ABSOLUTE_SHARD
--page-offset 33530449920 --expected-page-sha256 EXPECTED --output NEW_OUTPUT
--duration 60 --interval 1. First run the idle control without --phase-file or
DONTNEED. Root may separately choose the explicit single-page action after
preservation, then a named --phase-file. Each phase is exactly name, argv,
script_sha256, timeout. Only absolute /usr/bin/python3 and the pinned current
admit_postboot_original_qsa_reference_v4.py are admitted; no inference or Docker
phase is accepted. Preserve original OMP/MKL/OPENBLAS=1 and NUMEXPR absence.

RESULT -> The watcher hashes only 4096 bytes per sample, with bounded duration
and sample count. It saves the first changed buffered page and a direct page,
original source stat5 excluding atime, each phase before/after page, command,
actual child PID/start/end/return code and both owned pipe EOF receipts. CPU
phase output is capped at8MiB per stream while excess bytes are drained and
cause failure. Timeout or interruption terminates the owned process group and
requires reader retirement; neither becomes PASS.

VERDICT -> Observed page transition and phase timing only. Idle/reader/hash or
later GPU-fixture arms must be separate root-coordinated experiments. A page
hash match does not qualify all four shards or assign the recurring discrepancy
to CPU readers, a driver, hardware, a decoder, or a model operation.
