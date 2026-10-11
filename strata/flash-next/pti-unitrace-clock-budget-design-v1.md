# Fixed clock metadata budget successor

CONFIG -> The accepted clock bridge source remains immutable. This source-only
successor raises its per-process metadata ceiling from 8 MiB to exactly 512 MiB.
The 4096-byte row ceiling, exclusive per-PID file, serializer, mutex, original
queries, query brackets, clock samples, error exit86, and normal close are
unchanged. Default OFF behavior is unchanged. The overlay targets only
`tools/unitrace/src/pti_unitrace_clock_export_v1.h` in the explicitly pinned
bridge source. Full byte undo returns the accepted header.

COMMAND -> Run the declared six CPU controls. The tiny writer fixture uses a
separate 32-byte reference limit; it never changes the production header.
Actual git apply/check/reverse operates on a fresh temporary header copy.
Fresh PTI overlay/build source identity, actual compiler/dependency/source
receipts, installed ELF hashes, metadata smoke, and root-only leased runtime
qualification are required later. Do not patch the accepted source or install.

RESULT -> CPU controls exercise exact production boundary arithmetic without
allocating 512 MiB, row overflow, full-file overflow, exclusive bounded CPU
file writes retaining the completed prefix, typed extents, exact apply/undo,
and unchanged surrounding native bytes. These are source/CPU controls, not
execution of the C++ writer or native library.

VERDICT -> SOURCE ONLY. No new build, GPU query, model read, runtime, device
clock calibration, latency, or observer-cost qualification. 512 MiB bounds
only this process's clock metadata file, not Chrome total disk, all collector
memory, all process files, or whole-campaign output. Model observer cost is
unmeasured. Metadata overflow retains the original fatal refusal; it cannot
qualify a truncated capture. The later actor must separately preregister and
own a total raw disk budget, retain failure artifacts, and drain original
launch/session ownership before exclusion release.
